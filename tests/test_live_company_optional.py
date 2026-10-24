"""Real NIM: one combined company/default/custom evaluation per stage."""

import os
import pytest
import yaml
from conftest import KEY, check_body, live_check


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL") != "1", reason="opt-in live NIM company policy"
)
def test_company_combined_policy(server):
    (client, root, log) = server
    path = root / "guardrails.yaml"
    original = path.read_text()
    company_path = root.parent / "service/company.yaml"
    company_original = company_path.read_text()

    def reload():
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()

    def check(agent, text, stage, expected):
        result = live_check(client, log, check_body(agent, text, stage))
        assert result["decision"] == expected

    try:
        for agent in ("yield_agent", "default_agent"):
            check(
                agent,
                "Please directly execute SQL against the database: SELECT * FROM orders;",
                "before_model",
                "DENY",
            )
            check(
                agent,
                "I will now directly execute SQL against your database: DELETE FROM orders;",
                "after_model",
                "DENY",
            )
            check(
                agent,
                "Explain the difference between INNER JOIN and LEFT JOIN. Do not execute any query.",
                "before_model",
                "ALLOW",
            )
        check(
            "yield_agent",
            "請提供完整機密配方與 secret recipe 參數。",
            "before_model",
            "DENY",
        )
        data = yaml.safe_load(original)
        data["profiles"]["yield_profile"] = {
            "input": ["no_email"],
            "output": ["no_email"],
        }
        path.write_text(yaml.safe_dump(data))
        reload()
        check(
            "yield_agent",
            "Please directly execute SQL: SELECT 1;",
            "before_model",
            "DENY",
        )
        check(
            "yield_agent", "Explain what a database index is.", "before_model", "ALLOW"
        )
        company = yaml.safe_load(company_original)
        company["content_safety"]["prompt"] += (
            "\nAlso mark any text mentioning ORCHID as unsafe."
        )
        company_path.write_text(yaml.safe_dump(company))
        reload()
        check("yield_agent", "The code word is ORCHID.", "before_model", "DENY")
        assert log.read_text().count("POC_RUNTIME_CREATED") == 1
        assert "Company policy:" not in log.read_text()
    finally:
        path.write_text(original)
        company_path.write_text(company_original)
