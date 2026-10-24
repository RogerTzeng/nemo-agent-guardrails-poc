import asyncio
import json

import httpx
import pytest

from guardrail_extension.nemo_checks_client import NemoChecksClient
from guardrail_extension.safety_action import evaluate_policy

POLICY = {"mode": "custom", "instructions": "禁止機密配方"}


@pytest.mark.parametrize("status,finish,content,allowed", [
    (200, "stop", "User Safety: safe", True),
    (200, "stop", "User Safety: unsafe", False),
    (200, "stop", "unknown", False),
    (200, "length", "User Safety: safe", False),
    (429, "stop", "User Safety: safe", False),
])
def test_nim_action_default_and_failure_boundary(monkeypatch, status, finish, content, allowed):
    monkeypatch.delenv("SAFETY_EVALUATOR", raising=False)
    monkeypatch.setenv("SAFETY_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("SAFETY_API_KEY", "test-key")
    monkeypatch.setenv("SAFETY_MODEL", "test-model")
    original = httpx.AsyncClient
    async def handler(request):
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["custom_policy"] == POLICY["instructions"]
        return httpx.Response(status, json={"choices": [{"finish_reason": finish,
            "message": {"content": content}}]})
    monkeypatch.setattr("guardrail_extension.safety_action.httpx.AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler)))
    assert asyncio.run(evaluate_policy("text", {"effective_policy": POLICY})) is allowed


@pytest.mark.parametrize("terms,text,allowed", [
    (None, "hello", False), ([], "hello", False), ([""], "hello", False),
    ("secret", "hello", False), (["secret"], "hello", True), (["secret"], "SECRET", False),
])
def test_stub_requires_explicit_rules(monkeypatch, terms, text, allowed):
    monkeypatch.setenv("SAFETY_EVALUATOR", "stub")
    monkeypatch.setenv("SAFETY_STUB_POLICIES", json.dumps({POLICY["instructions"]: terms}))
    assert asyncio.run(evaluate_policy(text, {"effective_policy": POLICY})) is allowed


def test_missing_policy_and_bad_backend(monkeypatch):
    assert not asyncio.run(evaluate_policy("text", {}))
    monkeypatch.setenv("SAFETY_EVALUATOR", "unsupported")
    assert not asyncio.run(evaluate_policy("text", {"effective_policy": POLICY}))


def test_client_overlap_and_cancellation_cleanup():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"status": "passed"})
        native = NemoChecksClient("http://127.0.0.1:8000", transport=httpx.MockTransport(handler))
        tasks = [asyncio.create_task(native.check({})) for _ in range(2)]
        await entered.wait()
        shared = native.client
        assert native.users == 2
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert native.users == 1 and not shared.is_closed
        release.set()
        await tasks[1]
        assert native.users == 0 and native.client is None and shared.is_closed
    asyncio.run(run())
