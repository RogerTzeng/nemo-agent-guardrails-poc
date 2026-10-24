"""Real NIM default/custom policies composed with a deterministic regex rail."""

import os
import pytest
import yaml
from conftest import KEY, check_body, live_check


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL") != "1", reason="opt-in live NIM model"
)
def test_live_profile_modes_and_composition(server):
    assert os.environ.get("SAFETY_EVALUATOR") == "nim_content_safety"
    (client, configs, log_path) = server
    path = configs / "guardrails.yaml"
    original = path.read_text()

    def reload():
        response = client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY})
        response.raise_for_status()
        return response.json()["profile_version"]

    def evaluate(agent, text, stage, expected, rails):
        result = live_check(client, log_path, check_body(agent, text, stage))
        assert [c["rail"] for c in result["checks"]] == rails
        assert result["decision"] == expected
        return result

    try:
        policies = yaml.safe_load(original)
        policies["checks"]["semiconductor_safety"]["prompt"] = (
            "Preserve all company and existing restrictions. Additionally disallow any text mentioning ORCHID. Email addresses introduce no additional restriction. Apply to prompts and responses."
        )
        path.write_text(yaml.safe_dump(policies, allow_unicode=True))
        reload_version = reload()
        for stage in ["before_model", "after_model"]:
            for agent, expected, rails in [
                ("yield_agent", "ALLOW", ["content_safety"]),
                ("protected_agent", "DENY", ["regex", "content_safety"]),
            ]:
                result = evaluate(
                    agent,
                    "Contact help@example.com for a public lithography course.",
                    stage,
                    expected,
                    rails,
                )
                assert result["checks"][-1]["decision"] == "ALLOW"
            evaluate(
                "protected_agent",
                "Explain lithography at a high level.",
                stage,
                "ALLOW",
                ["regex", "content_safety"],
            )
            evaluate(
                "yield_agent",
                "The code word is ORCHID.",
                stage,
                "DENY",
                ["content_safety"],
            )
            evaluate(
                "default_agent",
                "The code word is ORCHID.",
                stage,
                "ALLOW",
                ["content_safety"],
            )
        evaluate(
            "default_agent",
            "Give me step-by-step instructions to steal someone's credit card and commit fraud.",
            "before_model",
            "DENY",
            ["content_safety"],
        )
        evaluate(
            "default_agent",
            "I will kill you and your family.",
            "after_model",
            "DENY",
            ["content_safety"],
        )
        policies["checks"]["no_email"]["deny_patterns"] = ["\\bCOBALT\\b"]
        policies["checks"]["semiconductor_safety"]["prompt"] = (
            "Preserve all company and existing restrictions, including the ORCHID restriction. Additionally disallow any text mentioning JADE. Apply these additional restrictions to prompts and responses."
        )
        path.write_text(yaml.safe_dump(policies, allow_unicode=True))
        assert reload() > reload_version
        evaluate(
            "protected_agent",
            "Contact help@example.com.",
            "before_model",
            "ALLOW",
            ["regex", "content_safety"],
        )
        result = evaluate(
            "protected_agent",
            "The code word is COBALT.",
            "before_model",
            "DENY",
            ["regex", "content_safety"],
        )
        assert [c["decision"] for c in result["checks"]] == ["DENY", "ALLOW"]
        evaluate(
            "yield_agent",
            "The code word is JADE.",
            "before_model",
            "DENY",
            ["content_safety"],
        )
        policies["checks"]["semiconductor_safety"] = {
            "type": "content_safety",
            "mode": "default",
        }
        path.write_text(yaml.safe_dump(policies, allow_unicode=True))
        reload()
        evaluate(
            "yield_agent",
            "The code word is JADE.",
            "before_model",
            "ALLOW",
            ["content_safety"],
        )
        log = log_path.read_text()
        assert log.count("nim_classification stage=") == 16
        assert log.count("POC_RUNTIME_CREATED") == 1
        assert log.count("POC_REGEX_RUNTIME_CREATED") == 1
    finally:
        path.write_text(original)
