import logging
import time

from .schemas import CheckRequest

log = logging.getLogger(__name__)


async def dispatch(request: CheckRequest, snapshot, native):
    profile_id = snapshot.agents.get(request.agent_id)
    if profile_id is None:
        return {"decision": "DENY", "agent_id": request.agent_id, "stage": request.stage,
                "profile_version": snapshot.version, "checks": [], "code": "UNKNOWN_AGENT"}
    results = []
    role, rail_type = (("user", "input") if request.stage == "before_model"
                       else ("assistant", "output"))
    messages = [{"role": role, "content": request.payload["text"]}]
    if rail_type == "output":
        messages.insert(0, {"role": "user", "content": ""})
    for check in snapshot.profiles[profile_id][request.stage]:
        start = time.perf_counter()
        policy = snapshot.checks[check.policy_id]
        body = {"model": "", "messages": messages,
                "guardrails": {"config_ids": [check.config_id], "rail_types": [rail_type],
                               "context": {"effective_policy": policy.model_dump(mode="json"),
                                           "policy_id": check.policy_id}}}
        if check.config_id == "native_shared":
            evidence = {key: request.context[key] for key in (
                "user_message", "relevant_chunks", "relevant_chunks_sep", "_last_bot_prompt", "bot_thinking")
                if key in request.context}
            body["guardrails"]["context"]["native_evidence"] = evidence
        decision, code = "DENY", "ERROR"
        try:
            result = await native.check(body)
            # No-op/missing/modified/unknown responses never pass. Explicit rail type
            # rejects unconfigured rails upstream; the shipped action always evaluates.
            if (isinstance(result, dict) and result.get("status") == "passed"
                    and result.get("content") == request.payload["text"]
                    and result.get("rail") is None):
                decision, code = "ALLOW", "PASSED"
            elif isinstance(result, dict) and result.get("status") == "blocked":
                code = "BLOCKED"
        except Exception:
            # Exceptions can embed request bodies or provider secrets: don't log them.
            pass
        item = {"rail": check.rail, "config_id": check.config_id,
                "policy_id": ("effective_content_safety" if check.policy_id.startswith("__company__/") else check.policy_id), "decision": decision, "code": code}
        results.append(item)
        request_id = request.context.get("request_id")
        log.info("check agent=%s stage=%s rail=%s policy=%s version=%s decision=%s ms=%.2f request_id=%s",
                 request.agent_id, request.stage, check.rail, check.policy_id, snapshot.version, decision,
                 (time.perf_counter() - start) * 1000,
                 request_id if isinstance(request_id, str) and request_id.isalnum() and len(request_id) < 80 else "-")
    return {"decision": "ALLOW" if all(r["decision"] == "ALLOW" for r in results) else "DENY",
            "agent_id": request.agent_id, "stage": request.stage, "profile_id": profile_id,
            "profile_version": snapshot.version, "checks": results}
