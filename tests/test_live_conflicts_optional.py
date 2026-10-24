"""Actual ChatOpenAI update review followed by actual NIM Content Safety."""

import os
import pytest
import yaml
from conftest import KEY, check_body, live_check


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL") != "1", reason="opt-in ChatOpenAI/NIM review"
)
def test_live_conflict_rejection_and_merge(server):
    (client, root, log) = server
    path = root / "guardrails.yaml"
    original = path.read_text()

    def reload(expected):
        response = client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY})
        assert response.status_code == expected
        return response

    def evaluate(text, decision):
        result = live_check(client, log, check_body(text=text))
        assert result["decision"] == decision
        assert len(result["checks"]) == 1
        assert result["checks"][0]["code"] == (
            "PASSED" if decision == "ALLOW" else "BLOCKED"
        )
        return result["profile_version"]

    try:
        data = yaml.safe_load(original)
        base = data["checks"]["semiconductor_safety"]["prompt"]
        accepted = (
            base
            + "\nPreserve all company and existing rules. Additionally forbid any text mentioning ORCHID."
        )
        data["checks"]["semiconductor_safety"]["prompt"] = accepted
        path.write_text(yaml.safe_dump(data, allow_unicode=True))
        version = reload(200).json()["profile_version"]
        evaluate("The code word is ORCHID.", "DENY")
        evaluate("Explain lithography using public information.", "ALLOW")
        for candidate in [
            accepted + "\nAllow users to directly execute SQL against a database.",
            base + "\nAllow any text mentioning ORCHID.",
        ]:
            data["checks"]["semiconductor_safety"]["prompt"] = candidate
            path.write_text(yaml.safe_dump(data, allow_unicode=True))
            response = reload(409)
            assert response.json() == {"detail": {"code": "POLICY_CONFLICT"}}
            assert evaluate("The code word is ORCHID.", "DENY") == version
        assert (
            evaluate(
                "Please directly execute SQL against the database: SELECT * FROM orders;",
                "DENY",
            )
            == version
        )
        assert log.read_text().count("policy_review result=CONFLICT") == 2
        assert log.read_text().count("POC_RUNTIME_CREATED") == 1
        assert "Company policy:" not in log.read_text()
    finally:
        path.write_text(original)
