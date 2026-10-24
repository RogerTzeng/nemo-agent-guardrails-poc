"""Request-local content safety action; no shared config/prompt mutation."""
import json
import logging
import os

import httpx
from nemoguardrails.actions import action

from .schemas import Policy
from .nim_safety import check_nim

log = logging.getLogger(__name__)


@action(name="evaluate_policy")
async def evaluate_policy(text: str, context: dict, stage: str = "input"):
    try:
        policy = Policy.model_validate(context["effective_policy"])
        if not isinstance(text, str) or not text.strip():
            return False
        mode = os.environ.get("SAFETY_EVALUATOR", "nim_content_safety")
        if mode == "stub":
            terms = json.loads(os.environ.get("SAFETY_STUB_POLICIES", "{}")).get(policy.instructions, [])
            if not isinstance(terms, list) or not terms or not all(isinstance(t, str) and t.strip() for t in terms):
                return False
            return policy.mode == "custom" and not any(term.casefold() in text.casefold() for term in terms)
        if mode != "nim_content_safety":
            return False
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            return await check_nim(client,
                os.environ["SAFETY_BASE_URL"].rstrip("/") + "/chat/completions",
                os.environ["SAFETY_API_KEY"], os.environ["SAFETY_MODEL"], text, policy, stage)
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        log.info("safety_evaluation_error stage=%s type=%s http_status=%s", stage,
                 type(error).__name__, status if isinstance(status, int) else "-")
        return False
