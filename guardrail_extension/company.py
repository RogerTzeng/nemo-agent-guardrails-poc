"""Private company policy compiled with per-Agent rules before dispatch."""
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import CheckSpec, Policy

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "service"


def service_root():
    return Path(os.environ.get("GUARDRAIL_SERVICE_CONFIG", str(SERVICE_ROOT)))


class CompanyContentSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str = Field(min_length=1, max_length=8000)


class CompanyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_safety: CompanyContentSafety


def load_company(root):
    # Never include private validation details in the public admin response.
    try:
        company = CompanyFile.model_validate(yaml.safe_load((root / "company.yaml").read_text()))
        baseline = (root / "default_content_safety.txt").read_text().strip()
        if not company.content_safety.prompt.strip() or not baseline:
            raise ValueError()
        return company.content_safety.prompt, baseline
    except Exception:
        raise ValueError("invalid private company policy") from None


def compose_policy(company_prompt, baseline, policies):
    sections = [
        "Evaluate compliance with the following combined policy. Violating any restriction is unsafe."
        "\n\n### Policy\nName: Company and Agent Policy\nDisallowed Behaviors:"
    ]
    if any(policy.mode == "default" for policy in policies):
        sections.append(baseline)
    sections.append(company_prompt)
    for policy in policies:
        if policy.mode == "custom":
            sections.append(policy.instructions)
    # Registry reviews source policies before publish; preserve every approved section.
    return Policy(mode="custom", instructions="\n".join(sections))


def enforce_company(stages, definitions, company_prompt, baseline, profile_name):
    """One safety check per stage, including when the public list contains none."""
    result = {}
    for stage, specs in stages.items():
        safety = [spec for spec in specs if spec.rail == "content_safety"]
        policy = compose_policy(company_prompt, baseline, [definitions[s.policy_id] for s in safety])
        # This key exists only in the compiled snapshot, not in public YAML.
        key = f"__company__/{profile_name}/{stage}"
        definitions[key] = policy
        merged = CheckSpec(rail="content_safety", config_id="content_safety_shared", policy_id=key)
        position = next((i for i, spec in enumerate(specs) if spec.rail == "content_safety"), 0)
        compiled = [spec for spec in specs if spec.rail != "content_safety"]
        compiled.insert(position, merged)
        result[stage] = tuple(compiled)
    return result
