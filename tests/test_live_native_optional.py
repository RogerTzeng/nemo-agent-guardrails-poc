"""Official CLI/runtime with strict self-check adapters and the real NIM LLM."""

import os
import pytest
import yaml
from conftest import KEY, check_body, live_check


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL") != "1", reason="opt-in live NIM native rails"
)
def test_native_self_checks_with_real_model(server):
    (client, root, log) = server
    path = root / "guardrails.yaml"
    private_path = root.parent / "service/native_rails.yaml"
    runtime_path = root / "native_shared/config.yml"
    originals = {p: p.read_text() for p in (path, private_path, runtime_path)}
    try:
        runtime = yaml.safe_load(originals[runtime_path])
        runtime["models"] = [
            {
                "type": "main",
                "engine": "openai",
                "model": os.environ["LLM"],
                "api_key_env_var": "SAFETY_API_KEY",
                "parameters": {
                    "base_url": os.environ["SAFETY_BASE_URL"],
                    "timeout": 90,
                    "max_retries": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
        ]
        runtime_path.write_text(yaml.safe_dump(runtime))
        deployment = yaml.safe_load(originals[private_path])
        data = yaml.safe_load(originals[path])
        for kind, stage, task, placeholder in [
            ("self_check.input_check", "input", "self_check_input", "{{ user_input }}"),
            (
                "self_check.output_check",
                "output",
                "self_check_output",
                "{{ bot_response }}",
            ),
        ]:
            preset = deployment["rails"][kind]["default"]
            preset["enabled"] = True
            preset["config"] = {
                "prompts": [
                    {
                        "task": task,
                        "content": "Decide whether the following text contains the word COPPER. Answer exactly yes if it does, otherwise exactly no. Do not explain.\nText: "
                        + placeholder,
                    }
                ]
            }
            data["checks"][kind] = {"type": kind}
            data["profiles"]["yield_profile"][stage] = [kind]
        private_path.write_text(yaml.safe_dump(deployment))
        path.write_text(yaml.safe_dump(data))
        client.post("/v1/admin/reload", headers={"X-Admin-Key": KEY}).raise_for_status()
        for stage in ["before_model", "after_model"]:
            for text, expected in [
                ("The word is COPPER.", "DENY"),
                ("The word is SILVER.", "ALLOW"),
            ]:
                result = live_check(client, log, check_body(text=text, stage=stage))
                assert result["checks"][0]["decision"] == "ALLOW"
                assert result["decision"] == expected
                assert result["checks"][-1]["code"] == (
                    "PASSED" if expected == "ALLOW" else "BLOCKED"
                )
        text = log.read_text()
        assert text.count("native_evaluation rail=self_check.") == 4
        assert text.count("nim_classification stage=") == 4
        assert text.count("POC_NATIVE_RUNTIME_CREATED") == 1
    finally:
        for p, original in originals.items():
            p.write_text(original)
