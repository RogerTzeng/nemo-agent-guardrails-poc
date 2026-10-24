"""Deterministic regex deny rail executed by the official NeMo runtime."""
import re

from nemoguardrails.actions import action

from .schemas import RegexRule


@action(name="evaluate_regex")
async def evaluate_regex(text: str, context: dict):
    try:
        rule = RegexRule.model_validate(context["effective_policy"])
        if not isinstance(text, str) or not text.strip():
            return False
        return not any(re.search(pattern, text, re.IGNORECASE) for pattern in rule.deny_patterns)
    except (ValueError, KeyError, TypeError):
        return False
