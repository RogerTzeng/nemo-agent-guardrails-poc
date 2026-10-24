"""Text-only DeepAgents middleware calling the native server's /v1/check."""

import uuid

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage


class HTTPGuardrailViolation(RuntimeError):
    def __init__(self, stage, code):
        self.stage, self.code = stage, code
        super().__init__(f"Guardrail stopped execution: stage={stage}, code={code}")


def message_text(message, stage):
    # Do not serialize tool arguments, internal metadata, or multimodal content.
    if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None):
        raise HTTPGuardrailViolation(stage, "UNSUPPORTED_TOOL_CALL")
    content = message.content
    if isinstance(content, list):
        if any(
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
            for block in content
        ):
            raise HTTPGuardrailViolation(stage, "UNSUPPORTED_CONTENT")
        content = "\n".join(block["text"] for block in content)
    if not isinstance(content, str) or not content.strip() or len(content) > 16000:
        raise HTTPGuardrailViolation(stage, "INVALID_TEXT")
    return content


class HTTPGuardrailsMiddleware(AgentMiddleware):
    """Check invocation input once, then every model response before release.

    Text-only, invoke/ainvoke: tool calls are rejected before execution. No token
    streaming protection is claimed; do not expose stream()/astream() output.
    """

    def __init__(self, agent_id, base_url="http://127.0.0.1:8000", timeout=95, transport=None):
        self.agent_id = agent_id
        self.url = base_url.rstrip("/") + "/v1/check"
        self.timeout, self.transport = timeout, transport

    def _body(self, message, stage, request_id):
        return {
            "agent_id": self.agent_id,
            "stage": stage,
            "payload": {"text": message_text(message, stage)},
            "context": {"request_id": request_id},
        }

    def _validate(self, response, stage):
        try:
            response.raise_for_status()
            result = response.json()
            checks = result["checks"]
            allowed = (
                result["decision"] == "ALLOW"
                and result["agent_id"] == self.agent_id
                and result["stage"] == stage
                and isinstance(checks, list)
                and checks
                and all(
                    isinstance(check, dict)
                    and check.get("decision") == "ALLOW"
                    and check.get("code") == "PASSED"
                    for check in checks
                )
            )
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            raise HTTPGuardrailViolation(stage, "SERVICE_ERROR") from None
        if not allowed:
            raise HTTPGuardrailViolation(stage, "DENIED")

    def _check(self, message, stage, request_id):
        body = self._body(message, stage, request_id)
        try:
            with httpx.Client(
                timeout=self.timeout, transport=self.transport, trust_env=False
            ) as client:
                response = client.post(self.url, json=body)
        except httpx.HTTPError:
            raise HTTPGuardrailViolation(stage, "SERVICE_ERROR") from None
        self._validate(response, stage)

    async def _acheck(self, message, stage, request_id):
        body = self._body(message, stage, request_id)
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport, trust_env=False
            ) as client:
                response = await client.post(self.url, json=body)
        except httpx.HTTPError:
            raise HTTPGuardrailViolation(stage, "SERVICE_ERROR") from None
        self._validate(response, stage)

    def _output(self, response):
        if isinstance(response, ExtendedModelResponse):
            # Commands can add uninspected messages; this example does not support them.
            if response.command is not None:
                raise HTTPGuardrailViolation("after_model", "UNSUPPORTED_COMMAND")
            response = response.model_response
        if isinstance(response, AIMessage):
            return [response]
        if (
            isinstance(response, ModelResponse)
            and response.result
            and response.structured_response is None
        ):
            return response.result
        raise HTTPGuardrailViolation("after_model", "UNSUPPORTED_RESPONSE")

    def _input(self, state):
        messages = state.get("messages", [])
        if not messages:
            raise HTTPGuardrailViolation("before_model", "MISSING_MESSAGES")
        return messages

    def before_agent(self, state, runtime):
        # Keep the HTTP stage name for compatibility with existing server profiles.
        request_id = uuid.uuid4().hex
        for message in self._input(state):
            self._check(message, "before_model", request_id)

    async def abefore_agent(self, state, runtime):
        request_id = uuid.uuid4().hex
        for message in self._input(state):
            await self._acheck(message, "before_model", request_id)

    def wrap_model_call(self, request, handler):
        response = handler(request)
        request_id = uuid.uuid4().hex
        for message in self._output(response):
            self._check(message, "after_model", request_id)
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        request_id = uuid.uuid4().hex
        for message in self._output(response):
            await self._acheck(message, "after_model", request_id)
        return response
