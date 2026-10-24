import asyncio
import json

import httpx
import pytest

pytest.importorskip("deepagents", reason="install the agent extra to run Agent tests")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from http_guardrail_demo import build_agent
from http_guardrails import HTTPGuardrailsMiddleware, HTTPGuardrailViolation


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def middleware(handler):
    return HTTPGuardrailsMiddleware("yield_agent", transport=httpx.MockTransport(handler))


def response(body, decision="ALLOW"):
    return httpx.Response(
        200,
        json={
            "decision": decision,
            "agent_id": body["agent_id"],
            "stage": body["stage"],
            "checks": [
                {"decision": decision, "code": "PASSED" if decision == "ALLOW" else "BLOCKED"}
            ],
        },
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("blocked", [None, "before_model", "after_model"])
def test_real_deepagent_hooks(async_mode, blocked):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        assert set(body) == {"agent_id", "stage", "payload", "context"}
        calls.append(body)
        return response(body, "DENY" if body["stage"] == blocked else "ALLOW")

    model = Model(responses=[AIMessage(content="public educational answer")])
    agent = build_agent(model, middleware(handler))

    def invoke():
        state = {"messages": [HumanMessage(content="Explain lithography.")]}
        return asyncio.run(agent.ainvoke(state)) if async_mode else agent.invoke(state)

    if blocked:
        with pytest.raises(HTTPGuardrailViolation) as caught:
            invoke()
        assert caught.value.stage == blocked
    else:
        assert invoke()["messages"][-1].text == "public educational answer"
    assert [c["stage"] for c in calls] == (
        ["before_model"] if blocked == "before_model" else ["before_model", "after_model"]
    )
    if len(calls) == 2:
        assert all(len(c["context"]["request_id"]) == 32 for c in calls)


@pytest.mark.parametrize(
    "kind", [500, 429, "network", "timeout", "json", "partial", "unknown", "empty_checks"]
)
def test_service_failure_prevents_model(kind):
    def handler(request):
        if kind == "network":
            raise httpx.ConnectError("unreachable")
        if kind == "timeout":
            raise httpx.ReadTimeout("timeout")
        if isinstance(kind, int):
            return httpx.Response(kind)
        if kind == "json":
            return httpx.Response(200, text="invalid")
        return httpx.Response(
            200,
            json={
                "partial": {"decision": "ALLOW"},
                "unknown": {"decision": "UNKNOWN"},
                "empty_checks": {
                    "decision": "ALLOW",
                    "agent_id": "yield_agent",
                    "stage": "before_model",
                    "checks": [],
                },
            }[kind],
        )

    model = Model(responses=[AIMessage(content="must never be released")])
    agent = build_agent(model, middleware(handler))
    with pytest.raises(HTTPGuardrailViolation):
        agent.invoke({"messages": [HumanMessage(content="hello")]})


def test_tool_call_rejected_before_execution():
    def handler(request):
        return response(json.loads(request.content))

    model = Model(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "write_todos", "args": {"todos": []}, "id": "t1"}]
            )
        ]
    )
    with pytest.raises(HTTPGuardrailViolation, match="UNSUPPORTED_TOOL_CALL"):
        build_agent(model, middleware(handler)).invoke(
            {"messages": [HumanMessage(content="hello")]}
        )


def test_text_blocks_and_incoming_history_checked_at_entry():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response(calls[-1])

    model = Model(responses=[AIMessage(content=[{"type": "text", "text": "answer"}])])
    build_agent(model, middleware(handler)).invoke(
        {
            "messages": [
                HumanMessage(content="old input"),
                AIMessage(content="old output"),
                HumanMessage(content=[{"type": "text", "text": "new input"}]),
            ]
        }
    )
    assert [c["payload"]["text"] for c in calls] == [
        "old input",
        "old output",
        "new input",
        "answer",
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_input_once_and_each_output_checked_across_model_loop(async_mode):
    from deepagents import create_deep_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import hook_config

    class RunTwice(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime):
            if sum(isinstance(m, AIMessage) for m in state["messages"]) == 1:
                return {"jump_to": "model"}

        @hook_config(can_jump_to=["model"])
        async def aafter_model(self, state, runtime):
            return self.after_model(state, runtime)

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response(calls[-1])

    agent = create_deep_agent(
        model=Model(responses=[AIMessage(content="first"), AIMessage(content="second")]),
        middleware=[middleware(handler), RunTwice()],
    )
    state = {"messages": [HumanMessage(content="input")]}
    result = asyncio.run(agent.ainvoke(state)) if async_mode else agent.invoke(state)
    assert result["messages"][-1].text == "second"
    assert [(c["stage"], c["payload"]["text"]) for c in calls] == [
        ("before_model", "input"),
        ("after_model", "first"),
        ("after_model", "second"),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_empty_input_blocked_at_agent_entry(async_mode):
    def handler(request):
        pytest.fail("empty input must fail before HTTP")

    agent = build_agent(Model(responses=[AIMessage(content="unused")]), middleware(handler))
    with pytest.raises(HTTPGuardrailViolation, match="MISSING_MESSAGES"):
        if async_mode:
            asyncio.run(agent.ainvoke({"messages": []}))
        else:
            agent.invoke({"messages": []})
