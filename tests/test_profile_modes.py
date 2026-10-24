"""Config composition uses real regex evaluation and explicit stub custom policies."""
import asyncio
import json

import httpx
import pytest
import yaml
from pydantic import ValidationError

from conftest import KEY, check_body
from guardrail_extension.nim_safety import check_nim
from guardrail_extension.regex_action import evaluate_regex
from guardrail_extension.schemas import Policy, RegexRule


@pytest.mark.parametrize("stage", ["before_model", "after_model"])
@pytest.mark.parametrize("text,expected", [
    ("Public lithography concepts", ["ALLOW", "ALLOW"]),
    ("Contact help@example.com", ["DENY", "ALLOW"]),
    ("機密配方", ["ALLOW", "DENY"]),
    ("機密配方 help@example.com", ["DENY", "DENY"]),
])
def test_different_rail_combinations(server, stage, text, expected):
    client, _, log = server
    combined = client.post("/v1/check", json=check_body("protected_agent", text, stage)).json()
    single = client.post("/v1/check", json=check_body("yield_agent", text, stage)).json()
    assert [c["rail"] for c in combined["checks"]] == ["regex", "content_safety"]
    assert [c["decision"] for c in combined["checks"]] == expected
    assert combined["decision"] == ("ALLOW" if expected == ["ALLOW", "ALLOW"] else "DENY")
    assert [c["rail"] for c in single["checks"]] == ["content_safety"]
    assert single["decision"] == expected[1]
    assert log.read_text().count("POC_RUNTIME_CREATED") == 1
    assert log.read_text().count("POC_REGEX_RUNTIME_CREATED") == 1


def test_regex_rules_and_profile_mapping_reload(server):
    client, root, log = server
    path = root / "guardrails.yaml"
    original = path.read_text()
    headers = {"X-Admin-Key": KEY}
    try:
        data = yaml.safe_load(original)
        data["checks"]["no_email"]["deny_patterns"] = [r"\bCOBALT\b"]
        path.write_text(yaml.safe_dump(data, allow_unicode=True))
        assert client.post("/v1/admin/reload", headers=headers).status_code == 200
        for text, expected in [("help@example.com", "ALLOW"), ("COBALT", "DENY")]:
            result = client.post("/v1/check", json=check_body("protected_agent", text)).json()
            assert result["decision"] == expected
        data["checks"]["no_email"]["deny_patterns"] = ["["]
        path.write_text(yaml.safe_dump(data))
        assert client.post("/v1/admin/reload", headers=headers).status_code == 422
        assert client.post("/v1/check", json=check_body("protected_agent", "COBALT")).json()["decision"] == "DENY"
        data = yaml.safe_load(original)
        data["agents"]["protected_agent"] = "yield_profile"
        path.write_text(yaml.safe_dump(data))
        assert client.post("/v1/admin/reload", headers=headers).status_code == 200
        result = client.post("/v1/check", json=check_body("protected_agent", "help@example.com")).json()
        assert result["decision"] == "ALLOW" and len(result["checks"]) == 1
        # Runtime IDs are internal; management YAML cannot override them.
        data["checks"]["no_email"]["config_id"] = "content_safety_shared"
        path.write_text(yaml.safe_dump(data))
        assert client.post("/v1/admin/reload", headers=headers).status_code == 422
        assert log.read_text().count("POC_RUNTIME_CREATED") == 1
        assert log.read_text().count("POC_REGEX_RUNTIME_CREATED") == 1
    finally:
        path.write_text(original)
        assert client.post("/v1/admin/reload", headers=headers).status_code == 200


@pytest.mark.parametrize("mode", ["default", "custom"])
@pytest.mark.parametrize("stage", ["input", "output"])
def test_default_and_custom_model_request(mode, stage):
    policy = Policy(mode="default") if mode == "default" else Policy(instructions="Block ORCHID")

    async def run():
        async def handler(request):
            body = json.loads(request.content)
            template = body["chat_template_kwargs"]
            if mode == "default":
                assert "custom_policy" not in template
                assert all("Block ORCHID" not in m["content"] for m in body["messages"])
            else:
                assert template["custom_policy"] == "Block ORCHID"
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"content": "User Safety: safe\nResponse Safety: safe"}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await check_nim(client, "https://nim.invalid/v1/chat/completions", "test", "model", "text", policy, stage)
    asyncio.run(run())


def test_regex_action_fail_closed():
    for context in [{}, {"effective_policy": {}}, {"effective_policy": {"deny_patterns": ["["]}}]:
        assert not asyncio.run(evaluate_regex("hello", context))
    with pytest.raises(ValidationError):
        RegexRule(deny_patterns=[])
