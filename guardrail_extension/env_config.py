"""Read simple .env assignments without executing shell code or printing secrets."""
from pathlib import Path
import shlex


def read_live_env(path: Path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw = line.removeprefix("export ").partition("=")
        if not separator:
            raise ValueError("expected .env assignment")
        tokens = shlex.split(raw, comments=True)
        if len(tokens) > 1:
            raise ValueError("quote .env values containing whitespace")
        values[key.strip()] = tokens[0] if tokens else ""
    for target, alias in [("SAFETY_API_KEY", "NVIDIA_API_KEY"),
                          ("SAFETY_BASE_URL", "base_url"), ("SAFETY_MODEL", "CONTENT_SAFETY")]:
        if not values.get(target) and values.get(alias):
            values[target] = values[alias]
    if not values.get("SAFETY_BASE_URL") and values.get("invoke_url"):
        values["SAFETY_BASE_URL"] = values["invoke_url"].removesuffix("/chat/completions")
    values.setdefault("SAFETY_EVALUATOR", "nim_content_safety")
    values.setdefault("GUARDRAIL_CHECK_TIMEOUT", "90")
    return values
