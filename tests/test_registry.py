"""Management YAML validation and its translation to the unchanged native calls."""
import asyncio

import pytest
import yaml
from pydantic import ValidationError

from conftest import check_body
from guardrail_extension.dispatcher import dispatch
from guardrail_extension.schemas import CheckRequest


@pytest.mark.parametrize("section,name,value", [
    ("agents", "yield_agent", "missing"),
    ("profiles", "yield_profile", {"input": ["missing"], "output": ["semiconductor_safety"]}),
    ("profiles", "yield_profile", {"input": [], "output": ["semiconductor_safety"]}),
    ("profiles", "yield_profile", {"input": ["semiconductor_safety"]}),
    ("profiles", "yield_profile", {"before_model": ["semiconductor_safety"], "output": ["semiconductor_safety"]}),
    ("checks", "semiconductor_safety", {"type": "content_safety", "mode": "custom"}),
    ("checks", "semiconductor_safety", {"type": "content_safety", "mode": "custom", "prompt": "  "}),
    ("checks", "semiconductor_safety", {"type": "content_safety", "mode": "default", "prompt": "override"}),
    ("checks", "semiconductor_safety", {"type": "content_safety", "mode": "default", "prompt": None}),
    ("checks", "semiconductor_safety", {"type": "content_safety", "mode": "custom", "prompt": "policy", "deny_terms": ["term"]}),
    ("checks", "semiconductor_safety", {"type": "unknown"}),
    ("checks", "no_email", {"type": "regex", "deny_patterns": ["["]}),
    ("checks", "no_email", {"type": "regex", "deny_patterns": ["email"], "prompt": "policy"}),
    ("checks", " ", {"type": "regex", "deny_patterns": ["email"]}),
])
def test_invalid_reload_preserves_snapshot(registry, section, name, value):
    path = registry.root / "guardrails.yaml"
    data = yaml.safe_load(path.read_text())
    data[section][name] = value
    path.write_text(yaml.safe_dump(data))
    original = registry.snapshot
    with pytest.raises(ValueError):
        registry.reload()
    assert registry.snapshot is original


def test_input_output_selection_and_request_context(registry):
    path = registry.root / "guardrails.yaml"
    data = yaml.safe_load(path.read_text())
    data["profiles"]["yield_profile"] = {
        "input": ["no_email", "semiconductor_safety"], "output": ["model_default"]}
    path.write_text(yaml.safe_dump(data))
    registry.reload()
    calls = []

    class Native:
        async def check(self, body):
            calls.append(body)
            return {"status": "passed", "content": "hello"}

    for stage in ["before_model", "after_model"]:
        result = asyncio.run(dispatch(CheckRequest(**check_body(text="hello", stage=stage)),
                                      registry.snapshot, Native()))
        assert result["decision"] == "ALLOW"
    assert [b["guardrails"]["config_ids"] for b in calls] == [
        ["regex_shared"], ["content_safety_shared"], ["content_safety_shared"]]
    assert [b["guardrails"]["rail_types"] for b in calls] == [["input"], ["input"], ["output"]]
    custom = calls[1]["guardrails"]["context"]["effective_policy"]
    assert data["checks"]["semiconductor_safety"]["prompt"] in custom["instructions"]
    assert "direct SQL execution" in custom["instructions"]
    assert calls[2]["guardrails"]["context"]["effective_policy"]["mode"] == "custom"
    assert "S24: Economic Harm." in calls[2]["guardrails"]["context"]["effective_policy"]["instructions"]
    with pytest.raises(TypeError):
        registry.snapshot.agents["yield_agent"] = "equipment_profile"
    with pytest.raises(TypeError):
        registry.snapshot.profiles["yield_profile"]["before_model"] = ()
    with pytest.raises(ValidationError):
        registry.snapshot.checks["semiconductor_safety"].instructions = "changed"
