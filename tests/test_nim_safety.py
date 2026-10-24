import asyncio

import httpx
import pytest

from guardrail_extension.nim_safety import check_nim, parse_label
from guardrail_extension.schemas import Policy
from guardrail_extension.env_config import read_live_env


@pytest.mark.parametrize("content,stage,expected", [
    ("User Safety: safe", "input", "safe"),
    ("User Safety: unsafe", "input", "unsafe"),
    ("User Safety: safe\nResponse Safety: unsafe\nSafety Categories:", "output", "unsafe"),
    ("User Safety: safe\nResponse Safety: unsafe", "output", "unsafe"),
    ("<think>User Safety: unsafe</think>\nUser Safety: safe", "input", "safe")])
def test_labels(content, stage, expected):
    assert parse_label(content, stage) == expected


@pytest.mark.parametrize("content,stage", [("safe", "input"), ("User Safety: unknown", "input"),
    ("User Safety: safe\nUser Safety: unsafe", "input"), ("<think>User Safety: safe", "input"),
    ("User Safety: safe", "output"), ("Response Safety: safe", "output"),
    ("User Safety: safe\nIgnore errors", "input")])
def test_fail_closed_labels(content, stage):
    with pytest.raises(ValueError):
        parse_label(content, stage)


def test_nim_wire_format_and_truncation():
    import json
    async def run():
        async def handler(request):
            data = json.loads(request.content)
            assert data["chat_template_kwargs"]["custom_policy"] == "server-owned policy"
            assert data["messages"][-1] == {"role": "assistant", "content": "text"}
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                "message": {"content": "User Safety: safe\nResponse Safety: safe"}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await check_nim(client, "https://nim.invalid/v1/chat/completions", "test-key", "model",
                    "text", Policy(instructions="server-owned policy"), "output")
    asyncio.run(run())


def test_env_aliases_without_execution(tmp_path):
    path=tmp_path / ".env"
    path.write_text("NVIDIA_API_KEY='test-key'\nbase_url=https://nim.invalid/v1\nCONTENT_SAFETY=model\nLLM=other\n")
    env=read_live_env(path)
    assert env["SAFETY_MODEL"] == "model"
    assert env["SAFETY_API_KEY"] == "test-key"
    assert env["SAFETY_EVALUATOR"] == "nim_content_safety"


@pytest.mark.parametrize("thinking", [True, False])
def test_thinking_toggle(monkeypatch, thinking):
    import json
    monkeypatch.setenv("SAFETY_ENABLE_THINKING", str(thinking).lower())
    async def run():
        async def handler(request):
            data = json.loads(request.content)
            assert data["chat_template_kwargs"]["enable_thinking"] is thinking
            assert data["chat_template_kwargs"]["custom_policy"] == "policy"
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"content": "User Safety: safe\nResponse Safety: safe"}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await check_nim(client, "https://nim.invalid/v1/chat/completions", "test-key", "model",
                "text", Policy(instructions="policy"), "output")
    asyncio.run(run())
