"""Invoke declared official actions inside the native Colang checks runtime.

No user-selected Python imports, no second LLMRails instance, no shared config
mutation. Provider/model configuration comes from the service-compiled snapshot.
"""
import inspect
import json
import logging

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailDecision, RailOutcome
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.manifests import resolve_import_ref

from .native_catalog import NativeRule, RESTRICTED, manifests
from .native_strict import CheckedHTTPClient, strict_alignscore, strict_grounding, strict_hf

log = logging.getLogger(__name__)


async def strict_jailbreak(text, config, surface, http_client=None):
    from nemoguardrails.library.jailbreak_detection.request import (
        jailbreak_detection_heuristics_request, jailbreak_detection_model_request, jailbreak_nim_request)
    settings = config.rails.config.jailbreak_detection
    if surface == "jailbreak detection heuristics" and settings.server_endpoint:
        result = await jailbreak_detection_heuristics_request(
            text, settings.server_endpoint, settings.length_per_perplexity_threshold,
            settings.prefix_suffix_perplexity_threshold, http_client=http_client)
    elif surface == "jailbreak detection model" and settings.nim_base_url:
        result = await jailbreak_nim_request(text, settings.nim_base_url, settings.get_api_key(),
                                             settings.nim_server_endpoint, http_client=http_client)
    elif surface == "jailbreak detection model" and settings.server_endpoint:
        result = await jailbreak_detection_model_request(text, settings.server_endpoint, http_client=http_client)
    else:
        return False
    # Official helpers return None on some failures; only actual False means safe.
    return result is False


async def strict_model_rail(rail, text, context, manager, llms, llm, params):
    from nemoguardrails.llm.call import llm_call
    if rail == "topic_safety":
        from nemoguardrails.library.topic_safety.actions import TOPIC_SAFETY_OUTPUT_RESTRICTION
        model_name = params["model"]
        model = llms.get(model_name)
        task = f"topic_safety_check_input $model={model_name}"
        prompt = [{"type": "system", "content": manager.render_task_prompt(task=task) + "\n" + TOPIC_SAFETY_OUTPUT_RESTRICTION},
                  {"type": "user", "content": text}]
        allowed, denied = "on-topic", "off-topic"
    else:
        task = "self_check_input" if rail == "self_check.input_check" else "self_check_output"
        task = params.get("variant") or task
        model = llms.get(task) or llm
        prompt = manager.render_task_prompt(task=task, context={
            "user_input": context.get("user_message", ""), "bot_response": context.get("bot_message", ""),
            "bot_thinking": context.get("bot_thinking", "")})
        allowed, denied = "no", "yes"
    if model is None:
        return False
    response = await llm_call(model, prompt, llm_params={"temperature": 0, "max_tokens": 256})
    if response.finish_reason != "stop":
        return False
    answer = response.content.strip().lower()
    if answer not in {allowed, denied}:
        return False
    return answer == allowed


@action(name="evaluate_native")
async def evaluate_native(text: str, context: dict, stage: str = "input", llms=None, llm=None):
    try:
        rule = NativeRule.model_validate(context["effective_policy"])
        if rule.rail in RESTRICTED or not text.strip():
            return False
        manifest = manifests()[rule.rail]
        surface_name = dict(rule.surfaces)[stage]
        surface = next(s for s in manifest.surfaces if s.name == surface_name and s.direction.value == stage)
        config = RailsConfig.from_content(config={"models": [], **json.loads(rule.config_json)})
        manager = LLMTaskManager(config)
        params = json.loads(rule.parameters_json)
        evidence = dict(context.get("native_evidence", {}))
        evidence["user_message" if stage == "input" else "bot_message"] = text
        # Never pass effective company policy or arbitrary client context to providers.
        kwargs = {"context": evidence, "config": config, "llm_task_manager": manager,
                  "llms": llms or {}, "llm": llm, "events": []}
        for binding in surface.bindings:
            if binding.kind == "literal":
                value = binding.value
            elif binding.kind == "surface_param":
                value = params.get(binding.key)
            else:
                value = evidence.get(binding.key)
            if (value is None or value == "" or value == []) and binding.required:
                raise ValueError("missing rail evidence or parameter")
            if value is not None:
                kwargs[binding.action_param] = value
        if rule.rail == "jailbreak_detection":
            allowed = await strict_jailbreak(text, config, surface.name)
        elif rule.rail in {"topic_safety", "self_check.input_check", "self_check.output_check"}:
            allowed = await strict_model_rail(rule.rail, text, evidence, manager, llms or {}, llm, params)
        elif rule.rail == "hf_classifier":
            allowed = await strict_hf(text, config, params)
        elif rule.rail == "factchecking.align_score":
            allowed = await strict_alignscore(config, evidence)
        elif rule.rail in {"hallucination", "self_check.facts"} or (
                rule.rail == "patronusai" and "lynx" in surface.name):
            allowed = await strict_grounding(rule.rail, manager, evidence, llm, llms or {})
        else:
            function = resolve_import_ref(surface.action)
            arguments = inspect.signature(function).parameters
            transport = CheckedHTTPClient(rule.rail) if "http_client" in arguments else None
            try:
                if transport:
                    kwargs["http_client"] = transport
                result = function(**{k: v for k, v in kwargs.items() if k in arguments})
                if inspect.isawaitable(result):
                    result = await result
                allowed = (isinstance(result, RailOutcome) and result.decision == RailDecision.ALLOW
                           and not result.failed and not result.metadata.get("fail_open")
                           and (transport is None or (transport.calls > 0 and not transport.failed)))
            finally:
                if transport:
                    await transport.close()
        log.info("native_evaluation rail=%s stage=%s decision=%s", rule.rail, stage, "ALLOW" if allowed else "DENY")
        return allowed
    except Exception as error:
        log.info("native_evaluation stage=%s decision=DENY code=ERROR type=%s", stage, type(error).__name__)
        return False
