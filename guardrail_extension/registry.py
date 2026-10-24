from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import CheckSpec, Policy, RegexRule
from .company import enforce_company, load_company, service_root
from .native_catalog import NativeCheck, NativeRule, compile_native, load_deployments
from .policy_conflicts import PolicyContext, PolicyReviewer
from .policy_state import PolicyState


class RegistryError(ValueError):
    """Validation error containing only a static, safe operator message."""


class ContentSafetyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["content_safety"]
    mode: Literal["custom", "default"]
    prompt: str | None = Field(default=None, min_length=1, max_length=8000)

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode == "custom" and (not self.prompt or not self.prompt.strip()):
            raise ValueError("custom mode requires a nonempty prompt")
        if self.mode == "default" and "prompt" in self.model_fields_set:
            raise ValueError("default mode must omit prompt")
        return self


class RegexCheck(RegexRule):
    type: Literal["regex"]


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input: tuple[str, ...] = Field(min_length=1)
    output: tuple[str, ...] = Field(min_length=1)


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: dict[str, str] = Field(min_length=1)
    profiles: dict[str, Profile] = Field(min_length=1)
    checks: dict[str, ContentSafetyCheck | RegexCheck | NativeCheck] = Field(min_length=1)


@dataclass(frozen=True)
class Snapshot:
    version: int
    agents: Mapping[str, str]
    profiles: Mapping[str, Mapping[str, tuple[CheckSpec, ...]]]
    checks: Mapping[str, Policy | RegexRule | NativeRule]
    policy_contexts: Mapping[tuple[str, ...], PolicyContext]


class ProfileRegistry:
    def __init__(self, root: Path, policy_reviewer=None):
        self.root = root
        self.policy_reviewer = policy_reviewer if policy_reviewer is not None else PolicyReviewer()
        self._reload_lock = Lock()
        self._state = PolicyState(root)
        candidate = self._candidate(1)
        self._review(candidate, previous=self._state.load())
        self._state.save(candidate.policy_contexts)
        self.snapshot = candidate

    def _candidate(self, version):
        company_prompt, baseline = load_company(service_root())
        data = RegistryFile.model_validate(yaml.safe_load((self.root / "guardrails.yaml").read_text()))
        if any(not name.strip() or name.startswith("__") for group in (data.agents, data.profiles, data.checks) for name in group):
            raise RegistryError("registry names cannot be empty")
        for profile in data.agents.values():
            if profile not in data.profiles:
                raise RegistryError("agent references missing profile")
        deployments = load_deployments()
        definitions, specs = {}, {}
        for name, check in data.checks.items():
            if isinstance(check, NativeCheck):
                try:
                    definitions[name] = compile_native(check, deployments)
                except Exception:
                    raise RegistryError("native rail preset is unavailable or invalid") from None
                specs[name] = CheckSpec(rail=check.type, config_id="native_shared", policy_id=name)
                continue
            # Runtime selection is fixed in code, never supplied by the registry.
            config_id = {"content_safety": "content_safety_shared", "regex": "regex_shared"}[check.type]
            if not (self.root / config_id / "config.yml").is_file():
                raise RegistryError("config.yml missing")
            specs[name] = CheckSpec(rail=check.type, config_id=config_id, policy_id=name)
            definitions[name] = (Policy(mode=check.mode, instructions=check.prompt)
                                 if isinstance(check, ContentSafetyCheck)
                                 else RegexRule(deny_patterns=check.deny_patterns))
        contexts = {}
        # Validate definitions even if they are not yet attached to a profile.
        for name, rule in definitions.items():
            if isinstance(rule, Policy) and rule.mode == "custom":
                contexts[("check", name)] = PolicyContext(company_prompt, None, (rule.instructions,))
        profiles = {}
        for name, profile in data.profiles.items():
            stages = {}
            for stage, names in (("before_model", profile.input), ("after_model", profile.output)):
                if any(check not in specs for check in names):
                    raise RegistryError("profile references missing check")
                direction = "input" if stage == "before_model" else "output"
                for check in names:
                    rule = definitions[check]
                    if isinstance(rule, NativeRule) and direction not in dict(rule.surfaces):
                        raise RegistryError("rail does not support this stage")
                stages[stage] = tuple(specs[check] for check in names)
            for stage, selected in stages.items():
                policies = [definitions[s.policy_id] for s in selected if s.rail == "content_safety"]
                context = PolicyContext(company_prompt,
                                        baseline if any(p.mode == "default" for p in policies) else None,
                                        tuple(p.instructions for p in policies if p.mode == "custom"))
                # Compare by Agent+stage, so renaming checks/profiles cannot bypass review.
                for agent, profile_name in data.agents.items():
                    if profile_name == name:
                        contexts[("agent", agent, stage)] = context
            profiles[name] = MappingProxyType(enforce_company(stages, definitions, company_prompt, baseline, name))
        return Snapshot(version, MappingProxyType(data.agents), MappingProxyType(profiles),
                        MappingProxyType(definitions), MappingProxyType(contexts))

    def _review(self, candidate, previous):
        reviewed = set()
        for scope, context in candidate.policy_contexts.items():
            old = previous.get(scope) if previous else None
            if old == context:
                continue
            pair = (context, old)
            if pair not in reviewed:
                self.policy_reviewer.review(context, old)
                reviewed.add(pair)

    def reload(self):
        # Serialize validation+publish; /v1/check continues reading the old snapshot.
        with self._reload_lock:
            candidate = self._candidate(self.snapshot.version + 1)
            self._review(candidate, previous=self.snapshot.policy_contexts)
            self._state.save(candidate.policy_contexts)
            self.snapshot = candidate
            return candidate
