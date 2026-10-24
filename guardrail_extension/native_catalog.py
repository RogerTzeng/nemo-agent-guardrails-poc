"""Public NeMo manifest catalog and service-owned native rail deployments."""
import json
import os
from typing import Any

import yaml
from nemoguardrails import RailsConfig
from nemoguardrails.manifests import default_rail_catalog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .company import service_root

# This manifest configures other fact-check rails and has no executable surface.
RESTRICTED = {"factchecking": "configuration-only manifest; select an executable facts rail"}


def manifests():
    return default_rail_catalog().manifests


class NativeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str
    preset: str = Field(default="default", min_length=1, max_length=100)

    @field_validator("type")
    @classmethod
    def known_rail(cls, value):
        if value not in manifests() or value in {"content_safety", "regex"}:
            raise ValueError("unknown native rail type")
        return value


class NativePreset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    surfaces: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class NativeDeployments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rails: dict[str, dict[str, NativePreset]]


class NativeRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rail: str
    surfaces: tuple[tuple[str, str], ...]
    parameters_json: str
    config_json: str


def load_deployments():
    try:
        return NativeDeployments.model_validate(yaml.safe_load((service_root() / "native_rails.yaml").read_text()))
    except Exception:
        raise ValueError("invalid private native rail deployments") from None


def compile_native(check, deployments):
    manifest = manifests()[check.type]
    preset = deployments.rails.get(check.type, {}).get(check.preset)
    if preset is None or not preset.enabled:
        raise ValueError("native rail preset is not enabled by service")
    if check.type in RESTRICTED:
        raise ValueError("native rail has no executable surface")
    declared = {(s.direction.value, s.name): s for s in manifest.surfaces}
    if not preset.surfaces or any((d, n) not in declared or d not in {"input", "output"}
                                  for d, n in preset.surfaces.items()):
        raise ValueError("unsupported native rail surface or stage")
    for d, n in preset.surfaces.items():
        surface = declared[d, n]
        if "mask " in n:
            raise ValueError("masking/retrieval requires a transformation HTTP contract")
        for binding in surface.bindings:
            if binding.kind == "surface_param" and binding.required and binding.key not in preset.parameters:
                raise ValueError("missing native surface parameter")
    for requirement in manifest.requirements.env_vars:
        if requirement.required and not os.environ.get(requirement.name):
            raise ValueError("native provider environment is not configured")
    # All provider endpoints, prompts, models and filesystem paths are service-owned.
    config = RailsConfig.from_content(config={"models": [], **preset.config})
    if config.models:
        raise ValueError("configure model resources in native_shared/config.yml, not presets")
    key = manifest.config_schema.key if manifest.config_schema else None
    if key and not preset.config.get("rails", {}).get("config", {}).get(key):
        raise ValueError("explicit native rail configuration required")
    if check.type == "hf_classifier" and not preset.parameters.get("expected_labels"):
        raise ValueError("HF classification requires expected_labels to reject unknown labels")
    if check.type in {"ai_defense", "f5", "trend_micro"}:
        settings = getattr(config.rails.config, key)
        if getattr(settings, "fail_open", False):
            raise ValueError("fail_open is prohibited")
    if check.type in {"sensitive_data_detection", "privateai", "gliner"}:
        settings = getattr(config.rails.config, key)
        for direction in preset.surfaces:
            options = getattr(settings, direction, None)
            if options is None or not getattr(options, "entities", None):
                raise ValueError("PII detection requires explicit nonempty entities")
    if check.type == "context_bloat_detection" and config.rails.config.context_bloat_detection.action != "reject":
        raise ValueError("only reject is supported for context bloat")
    if check.type == "injection_detection":
        settings = config.rails.config.injection_detection
        if settings.action != "reject" or not settings.injections or settings.yara_path or settings.yara_rules:
            raise ValueError("only bundled YARA rules in reject mode are supported")
    if check.type == "jailbreak_detection":
        settings = config.rails.config.jailbreak_detection
        if not settings.server_endpoint and not settings.nim_base_url:
            raise ValueError("jailbreak requires a configured classification endpoint")
    return NativeRule(rail=check.type, surfaces=tuple(preset.surfaces.items()),
                      parameters_json=json.dumps(preset.parameters), config_json=json.dumps(preset.config))


def public_catalog():
    deployments = load_deployments()
    result = []
    for name, manifest in manifests().items():
        enabled = []
        for preset, definition in deployments.rails.get(name, {}).items():
            if definition.enabled and name not in {"content_safety", "regex"}:
                try:
                    compile_native(NativeCheck(type=name, preset=preset), deployments)
                    enabled.append(preset)
                except ValueError:
                    pass
        result.append({"type": name, "directions": sorted({s.direction.value for s in manifest.surfaces}),
                       "enabled_presets": enabled,
                       "status": "integrated" if name in {"content_safety", "regex"} else
                                 "restricted" if name in RESTRICTED else
                                 "enabled_unverified" if enabled else "requires_service_configuration"})
    return result
