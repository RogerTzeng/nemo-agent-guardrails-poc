"""DeepAgents -> HTTP middleware -> official NeMo native server."""

import argparse
import os
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from http_guardrails import HTTPGuardrailsMiddleware, HTTPGuardrailViolation

ROOT = Path(__file__).resolve().parent
DEFAULT_ENV = ROOT / ".env"


def build_agent(model, middleware):
    return create_deep_agent(
        model=model,
        middleware=[middleware],
        system_prompt="You are a text-only semiconductor education assistant. Answer directly "
        "using public information. Do not call tools, delegate, access files, or create plans. "
        "Never invent proprietary recipes or equipment procedures.",
    )


def settings(path):
    return {
        **{
            key: value
            for key, value in dotenv_values(path, interpolate=False).items()
            if value is not None
        },
        **os.environ,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="用繁體中文簡短說明曝光微影的基本概念。")
    parser.add_argument(
        "--agent-id", choices=["yield_agent", "equipment_agent", "protected_agent", "default_agent"], default="yield_agent"
    )
    parser.add_argument("--guardrail-url", default="http://127.0.0.1:8000")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    args = parser.parse_args()
    env = settings(args.env)
    key = env.get("AGENT_API_KEY") or env.get("NVIDIA_API_KEY")
    model_name = env.get("AGENT_MODEL") or env.get("LLM")
    if not key or not model_name:
        parser.error("Set NVIDIA_API_KEY and LLM (or AGENT_API_KEY and AGENT_MODEL) in --env")
    model = ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url=env.get("AGENT_BASE_URL") or env.get("base_url"),
        timeout=90,
        max_retries=0,
        max_tokens=1024,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    middleware = HTTPGuardrailsMiddleware(args.agent_id, args.guardrail_url)
    try:
        result = build_agent(model, middleware).invoke(
            {"messages": [HumanMessage(content=args.prompt)]}, {"recursion_limit": 10}
        )
        print(result["messages"][-1].text)
    except HTTPGuardrailViolation as error:
        print(str(error))
        return 2
    except Exception:  # noqa: BLE001 -- do not expose provider credentials in CLI errors
        # Provider exceptions can contain credentials or request content.
        print("Agent execution failed; no answer released.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
