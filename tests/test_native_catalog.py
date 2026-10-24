import asyncio

import pytest
import yaml
from nemoguardrails.actions.rail_outcome import RailOutcome

from conftest import KEY, check_body
from guardrail_extension.native_catalog import NativeCheck, compile_native, load_deployments, manifests, RESTRICTED
from guardrail_extension.native_action import evaluate_native, strict_jailbreak


def test_all_release_manifests_in_private_catalog():
    catalog = manifests()
    deployments = load_deployments()
    assert len(catalog) == 33
    assert set(deployments.rails) | {"content_safety", "regex"} == set(catalog)
    for name, presets in deployments.rails.items():
        for preset in presets.values():
            assert all((d, n) in {(s.direction.value, s.name) for s in catalog[name].surfaces}
                       for d, n in preset.surfaces.items())


def test_disabled_and_restricted_presets_never_activate():
    deployments = load_deployments()
    for name in deployments.rails:
        if name in {"context_bloat_detection", "injection_detection"}:
            continue
        with pytest.raises(ValueError):
            compile_native(NativeCheck(type=name), deployments)
    for name in RESTRICTED:
        deployments.rails[name]["default"].enabled = True
        with pytest.raises(ValueError, match="no executable surface"):
            compile_native(NativeCheck(type=name), deployments)


@pytest.mark.parametrize("outcome", [True, None, {}, RailOutcome.failure(),
                                      RailOutcome.allow(metadata={"fail_open": True})])
def test_native_adapter_requires_actual_non_fallback_outcome(monkeypatch, outcome):
    rule = compile_native(NativeCheck(type="context_bloat_detection"), load_deployments())
    monkeypatch.setattr("guardrail_extension.native_action.resolve_import_ref", lambda _: lambda **kw: outcome)
    assert not asyncio.run(evaluate_native("hello", {"effective_policy": rule.model_dump()}))


def test_official_bloat_through_cli_and_reload(server):
    client, root, log = server
    path = root / "guardrails.yaml"
    original = path.read_text()
    try:
        data = yaml.safe_load(original)
        data["checks"]["size"] = {"type": "context_bloat_detection"}
        data["profiles"]["yield_profile"]["input"].append("size")
        path.write_text(yaml.safe_dump(data))
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()
        for text, expected in [("Hello world", "ALLOW"), ("a" * 6000, "DENY")]:
            result = client.post("/v1/check", json=check_body(text=text)).json()
            assert result["decision"] == expected
            assert [c["rail"] for c in result["checks"]] == ["content_safety", "context_bloat_detection"]
            assert result["checks"][-1]["code"] == ("PASSED" if expected == "ALLOW" else "BLOCKED")
        data["profiles"]["yield_profile"]["output"] = ["size"]
        path.write_text(yaml.safe_dump(data))
        assert client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).status_code == 422
        catalog = client.get("/v1/guardrail-catalog").json()["rails"]
        assert len(catalog) == 33
        assert all("config" not in row and "prompt" not in row for row in catalog)
        assert log.read_text().count("POC_NATIVE_RUNTIME_CREATED") == 1
    finally:
        path.write_text(original)
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()


@pytest.mark.parametrize("value,expected", [(False, True), (True, False), (None, False), (0, False), ("false", False)])
def test_jailbreak_strict_helper(monkeypatch, value, expected):
    from nemoguardrails import RailsConfig
    async def fake(*args, **kwargs):
        return value
    monkeypatch.setattr("nemoguardrails.library.jailbreak_detection.request.jailbreak_detection_model_request", fake)
    config = RailsConfig.from_content(config={"models": [], "rails": {"config": {
        "jailbreak_detection": {"server_endpoint": "http://classifier.invalid/model", "nim_base_url": None}}}})
    assert asyncio.run(strict_jailbreak("hello", config, "jailbreak detection model")) is expected


@pytest.mark.parametrize("rail,answer,finish,expected", [
    ("topic_safety", "on-topic", "stop", True), ("topic_safety", "off-topic", "stop", False),
    ("topic_safety", "unexpected", "stop", False), ("topic_safety", "on-topic", "length", False),
    ("self_check.input_check", "no", "stop", True), ("self_check.input_check", "yes", "stop", False),
    ("self_check.output_check", "no", "length", False), ("self_check.output_check", "unknown", "stop", False)])
def test_strict_model_results(monkeypatch, rail, answer, finish, expected):
    from types import SimpleNamespace
    from nemoguardrails.types import LLMResponse
    from guardrail_extension.native_action import strict_model_rail
    async def fake(*args, **kwargs):
        return LLMResponse(content=answer, finish_reason=finish)
    monkeypatch.setattr("nemoguardrails.llm.call.llm_call", fake)
    manager = SimpleNamespace(render_task_prompt=lambda **kwargs: "classify")
    assert asyncio.run(strict_model_rail(rail, "text", {}, manager,
                                        {"topic_control": object()}, object(), {"model": "topic_control"})) is expected


def test_injection_through_native_cli(server):
    pytest.importorskip("yara")
    client, root, log = server
    path = root / "guardrails.yaml"
    original = path.read_text()
    try:
        data = yaml.safe_load(original)
        data["checks"]["injection"] = {"type": "injection_detection"}
        data["profiles"]["yield_profile"]["output"].append("injection")
        path.write_text(yaml.safe_dump(data))
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()
        for text, expected in [("Public security concepts", "ALLOW"), ("<script>alert(1)</script>", "DENY"),
                               ("SELECT * FROM users WHERE name='a' -- comment", "DENY")]:
            result = client.post("/v1/check", json=check_body(text=text, stage="after_model")).json()
            assert result["decision"] == expected
            assert result["checks"][-1]["rail"] == "injection_detection"
    finally:
        path.write_text(original)
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()
