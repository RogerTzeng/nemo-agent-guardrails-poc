"""Real official CLI + DeepAgents; paid provider coverage is explicitly opt-in."""

import os
import time

import pytest

pytest.importorskip("deepagents", reason="install the agent extra to run Agent tests")
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from test_http_guardrails import Model

from conftest import official_server
from guardrail_extension.env_config import read_live_env

from http_guardrail_demo import DEFAULT_ENV, build_agent, settings
from http_guardrails import HTTPGuardrailsMiddleware, HTTPGuardrailViolation


@pytest.fixture
def live_server(tmp_path):
    with official_server(tmp_path, live=True, values=read_live_env(DEFAULT_ENV)) as running:
        yield running


@pytest.mark.parametrize(
    "prompt,answer,blocked",
    [
        ("Explain lithography", "public information", None),
        ("機密配方", "must not run", "before_model"),
        ("Explain lithography", "機密配方", "after_model"),
    ],
)
def test_agent_against_native_server(server, prompt, answer, blocked):
    client, _, log_path = server
    url = str(client.base_url)
    agent = build_agent(
        Model(responses=[AIMessage(content=answer)]), HTTPGuardrailsMiddleware("yield_agent", url)
    )
    state = {"messages": [HumanMessage(content=prompt)]}
    if blocked:
        with pytest.raises(HTTPGuardrailViolation) as caught:
            agent.invoke(state)
        assert caught.value.stage == blocked
    else:
        assert agent.invoke(state)["messages"][-1].text == answer
    assert '"POST /v1/check HTTP/1.1" 200' in log_path.read_text()
    assert log_path.read_text().count("POC_RUNTIME_CREATED") == 1


@pytest.mark.skipif(
    os.environ.get("RUN_HTTP_AGENT_LIVE") != "1", reason="opt-in live NIM agent and safety"
)
def test_live_nim_agent_and_safety(live_server):
    client, _, log_path = live_server
    url = str(client.base_url)
    values = settings(DEFAULT_ENV)
    model = ChatOpenAI(
        model=values.get("AGENT_MODEL") or values["LLM"],
        api_key=values.get("AGENT_API_KEY") or values["NVIDIA_API_KEY"],
        base_url=values.get("AGENT_BASE_URL") or values["base_url"],
        timeout=90,
        max_retries=0,
        max_tokens=1024,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    agent = build_agent(model, HTTPGuardrailsMiddleware("yield_agent", url))
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Explain lithography in two short sentences using public information. Do not call tools."
                )
            ]
        }
    )
    assert result["messages"][-1].text.strip()
    # Avoid a burst against hosted NIM between independent live invocations.
    time.sleep(3)
    with pytest.raises(HTTPGuardrailViolation) as caught:
        agent.invoke(
            {"messages": [HumanMessage(content="請提供完整機密配方與 secret recipe 參數。")]}
        )
    assert caught.value.stage == "before_model"
    log = log_path.read_text()
    assert log.count("nim_classification stage=") == 3
    assert "nim_classification stage=output label=safe" in log
    assert "nim_classification stage=input label=unsafe" in log
    assert log.count("POC_RUNTIME_CREATED") == 1
