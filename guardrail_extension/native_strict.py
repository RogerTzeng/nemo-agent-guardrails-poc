"""Strict boundaries for upstream helpers with permissive failure paths.

Uses public provider interfaces and request helpers; never patches upstream code.
"""
import json
import math

from nemoguardrails.http import create_http_client
from nemoguardrails.llm.call import llm_call


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


class CheckedHTTPClient:
    """Remember errors even if a provider action catches them and returns ALLOW."""
    def __init__(self, rail, client=None):
        self.rail = rail
        self.client = client or create_http_client()
        self.failed = False
        self.calls = 0

    async def request(self, method, url, **kwargs):
        self.calls += 1
        try:
            response = await self.client.request(method, url, **kwargs)
            if response.status_code != 200:
                raise ValueError("provider HTTP failure")
            data = response.json()
            if self.rail == "polygraf":
                entities = data if isinstance(data, list) else data.get("entities")
                if not isinstance(entities, list):
                    raise ValueError("missing provider entities")
            elif self.rail == "fiddler":
                keys = (["fdl_faithful_score"] if url.endswith("ftl-response-faithfulness") else
                        ["fdl_" + name for name in ("harmful", "violent", "unethical", "illegal", "sexual", "racist",
                                                   "jailbreaking", "harassing", "hateful", "sexist", "roleplaying")])
                if not all(probability(data.get(key)) for key in keys):
                    raise ValueError("missing provider scores")
            elif self.rail == "patronusai":
                results = data.get("results")
                if not isinstance(results, list) or not results or not all(
                    isinstance(r, dict) and isinstance(r.get("evaluation_result"), dict)
                    and type(r["evaluation_result"].get("pass")) is bool for r in results):
                    raise ValueError("missing provider evaluations")
            return response
        except Exception:
            self.failed = True
            raise

    async def close(self):
        await self.client.close()


async def strict_hf(text, config, params):
    from nemoguardrails.library.hf_classifier.backends import get_backend
    name = params["classifier"]
    settings = config.rails.config.hf_classifier[name]
    expected = set(params["expected_labels"])
    if not settings.blocked_labels or not set(settings.blocked_labels) <= expected:
        return False
    results = await get_backend(settings, name=name).classify(text)
    if not results or any(r.get("label") not in expected or not probability(r.get("score")) for r in results):
        return False
    return not any(r["label"] in settings.blocked_labels and r["score"] >= settings.threshold for r in results)


async def strict_alignscore(config, evidence):
    from nemoguardrails.library.factchecking.align_score.request import alignscore_request
    chunks = evidence.get("relevant_chunks")
    if not chunks:
        return False
    endpoint = config.rails.config.fact_checking.parameters.get("endpoint")
    if not endpoint:
        return False
    score = await alignscore_request(endpoint, chunks, evidence["bot_message"])
    return probability(score) and score >= 0.5


async def strict_grounding(rail, manager, evidence, llm, llms):
    if rail == "hallucination":
        original_prompt = evidence.get("_last_bot_prompt")
        if not original_prompt or llm is None:
            return False
        alternatives = []
        for _ in range(2):
            result = await llm_call(llm, original_prompt, llm_params={"temperature": 1.0, "max_tokens": 1024})
            if result.finish_reason != "stop" or not result.content.strip():
                return False
            alternatives.append(result.content)
        task = "self_check_hallucination"
        prompt = manager.render_task_prompt(task=task, context={
            "statement": evidence["bot_message"], "paragraph": ". ".join(alternatives)})
    elif rail == "self_check.facts":
        if not evidence.get("relevant_chunks") or llm is None:
            return False
        task = "self_check_facts"
        prompt = manager.render_task_prompt(task=task, context={
            "evidence": evidence["relevant_chunks"], "response": evidence["bot_message"]})
    else:
        llm = llms.get("patronus_lynx")
        if not evidence.get("relevant_chunks") or llm is None:
            return False
        task = "patronus_lynx_check_output_hallucination"
        prompt = manager.render_task_prompt(task=task, context={
            "user_input": evidence["user_message"], "bot_response": evidence["bot_message"],
            "provided_context": evidence["relevant_chunks"]})
    result = await llm_call(llm, prompt, llm_params={"temperature": 0, "max_tokens": 1024})
    if result.finish_reason != "stop":
        return False
    if rail == "patronusai":
        parsed = json.loads(result.content)
        return isinstance(parsed, dict) and parsed.get("SCORE") == "PASS"
    return result.content.strip().lower() == "yes"
