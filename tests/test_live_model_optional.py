import os
import pytest
import yaml
from conftest import KEY, check_body, live_check


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL") != "1", reason="opt-in paid/internal model"
)
def test_shared_policy_isolation_and_reload(server):
    for key in ["SAFETY_BASE_URL", "SAFETY_MODEL", "SAFETY_API_KEY"]:
        assert os.environ.get(key), f"missing {key}"
    (client, configs, log) = server
    path = configs / "guardrails.yaml"
    original = path.read_text()

    def evaluate(agent, text, stage, expected):
        result = live_check(
            client, log, check_body(agent=agent, text=text, stage=stage)
        )
        assert result["decision"] == expected
        assert result["checks"][0]["code"] == (
            "PASSED" if expected == "ALLOW" else "BLOCKED"
        )

    def reload():
        response = client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY})
        response.raise_for_status()

    try:
        policies = yaml.safe_load(original)
        policies["checks"]["semiconductor_safety"]["prompt"] = (
            "Preserve all company and existing restrictions. Additionally disallow any text mentioning ORCHID. Apply this additional restriction to both prompts and responses."
        )
        policies["checks"]["equipment_safety"]["prompt"] = (
            "Preserve all company and existing restrictions. Additionally disallow any text mentioning COBALT. Apply this restriction to prompts and responses."
        )
        path.write_text(yaml.safe_dump(policies, allow_unicode=True))
        reload()
        for stage in ["before_model", "after_model"]:
            for agent, expected in [
                ("yield_agent", "DENY"),
                ("equipment_agent", "ALLOW"),
            ]:
                evaluate(agent, "The code word is ORCHID.", stage, expected)
        policies["checks"]["semiconductor_safety"]["prompt"] = (
            "Preserve all company and existing restrictions, including ORCHID. Additionally disallow any text mentioning COBALT. Apply this restriction to prompts and responses."
        )
        path.write_text(yaml.safe_dump(policies, allow_unicode=True))
        reload()
        evaluate("yield_agent", "The code word is COBALT.", "before_model", "DENY")
        assert log.read_text().count("POC_RUNTIME_CREATED") == 1
        assert log.read_text().count("nim_classification stage=") == 5
    finally:
        path.write_text(original)
