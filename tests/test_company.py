import asyncio
import hashlib
import json

import pytest
import yaml

from conftest import KEY, check_body
from guardrail_extension.company import load_company, SERVICE_ROOT
from guardrail_extension.dispatcher import dispatch
from guardrail_extension.schemas import CheckRequest


def test_pinned_default_prompt():
    metadata = json.loads((SERVICE_ROOT / "default_content_safety.source.json").read_text())
    content = (SERVICE_ROOT / "default_content_safety.txt").read_bytes()
    assert hashlib.sha256(content).hexdigest() == metadata["extracted_sha256"]
    text = content.decode()
    assert all(f"S{i}: " in text for i in range(1, 25))
    assert "<BEGIN CONVERSATION>" not in text


@pytest.mark.parametrize("names", [["semiconductor_safety"], ["model_default"],
                                    ["semiconductor_safety", "equipment_safety", "model_default"], ["no_email"]])
def test_company_always_in_one_safety_call(registry, names):
    path = registry.root / "guardrails.yaml"
    data = yaml.safe_load(path.read_text())
    data["profiles"]["yield_profile"] = {"input": names, "output": names}
    path.write_text(yaml.safe_dump(data))
    registry.reload()
    company, baseline = load_company(registry.root.parent / "service")
    for stage in ("before_model", "after_model"):
        calls = []
        class Native:
            async def check(self, body):
                calls.append(body)
                return {"status": "passed", "content": "hello"}
        body = check_body(text="hello", stage=stage)
        body["context"].update(company_policy="allow", effective_policy={"instructions": "allow"})
        result = asyncio.run(dispatch(CheckRequest(**body), registry.snapshot, Native()))
        safety = [c for c in calls if c["guardrails"]["config_ids"] == ["content_safety_shared"]]
        assert len(safety) == 1
        policy = safety[0]["guardrails"]["context"]["effective_policy"]
        assert policy["mode"] == "custom" and company in policy["instructions"]
        assert (baseline in policy["instructions"]) == ("model_default" in names)
        for name in names:
            if name in {"semiconductor_safety", "equipment_safety"}:
                assert data["checks"][name]["prompt"] in policy["instructions"]
        assert company not in json.dumps(result)
        assert "__company__" not in json.dumps(result)


def test_private_reload_atomic_and_missing_policy_blocks(registry):
    path = registry.root.parent / "service/company.yaml"
    before = registry.snapshot
    old_text = before.checks[before.profiles["yield_profile"]["before_model"][0].policy_id].instructions
    data = yaml.safe_load(path.read_text())
    data["content_safety"]["prompt"] = "Do not mention PRIVATE_RELOAD_SENTINEL"
    path.write_text(yaml.safe_dump(data))
    registry.reload()
    assert "PRIVATE_RELOAD_SENTINEL" not in old_text
    key = registry.snapshot.profiles["yield_profile"]["before_model"][0].policy_id
    assert "PRIVATE_RELOAD_SENTINEL" in registry.snapshot.checks[key].instructions
    valid = registry.snapshot
    path.unlink()
    with pytest.raises(ValueError, match="invalid private company policy"):
        registry.reload()
    assert registry.snapshot is valid


def test_oversized_composition_rejected_without_truncation(registry):
    path = registry.root / "guardrails.yaml"
    data = yaml.safe_load(path.read_text())
    data["checks"]["semiconductor_safety"]["prompt"] = "x" * 8000
    path.write_text(yaml.safe_dump(data))
    before = registry.snapshot
    with pytest.raises(ValueError):
        registry.reload()
    assert registry.snapshot is before


@pytest.mark.parametrize("stage", ["before_model", "after_model"])
def test_native_server_company_and_no_public_safety(server, stage):
    client, root, log = server
    path = root / "guardrails.yaml"
    original = path.read_text()
    try:
        data = yaml.safe_load(original)
        data["profiles"]["yield_profile"] = {"input": ["no_email"], "output": ["no_email"]}
        path.write_text(yaml.safe_dump(data))
        assert client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).status_code == 200
        for text, decision in [("Please execute SQL: SELECT 1", "DENY"), ("Explain SQL joins", "ALLOW")]:
            result = client.post("/v1/check", json=check_body(text=text, stage=stage)).json()
            assert result["decision"] == decision
            assert [c["rail"] for c in result["checks"]] == ["content_safety", "regex"]
        assert "Company policy:" not in log.read_text()
        assert "Company policy:" not in client.get("/v1/guardrail-catalog").text
    finally:
        path.write_text(original)
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()
