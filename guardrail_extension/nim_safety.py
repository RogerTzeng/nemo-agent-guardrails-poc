"""Nemotron 3.5 Content Safety's documented custom-policy wire protocol."""
import logging
import os
import re

log = logging.getLogger(__name__)


def parse_label(content, stage):
    if not isinstance(content, str):
        raise ValueError("missing safety labels")
    content = content.strip()
    if content.startswith("<think>"):
        if "</think>" not in content:
            raise ValueError("incomplete reasoning")
        content = content.split("</think>", 1)[1].strip()
    labels = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"(User Safety|Response Safety): (safe|unsafe)", line)
        if match:
            if match[1] in labels:
                raise ValueError("duplicate safety label")
            labels[match[1]] = match[2]
        elif not re.fullmatch(r"Safety Categories:(?: .*)?", line):
            raise ValueError("unexpected safety output")
    required = "Response Safety" if stage == "output" else "User Safety"
    if "User Safety" not in labels or required not in labels:
        raise ValueError("required safety label missing")
    return labels[required]


async def check_nim(client, url, key, model, text, policy, stage):
    setting = os.environ.get("SAFETY_ENABLE_THINKING", "false").lower()
    if setting not in {"true", "false"}:
        raise ValueError("SAFETY_ENABLE_THINKING must be true or false")
    messages = [{"role": "user", "content": text}]
    if stage == "output":
        # Synthetic control prompt, not an invented prior conversation.
        messages = [{"role": "user", "content": "Evaluate the provided response under the custom policy."},
                    {"role": "assistant", "content": text}]
    template = {"enable_thinking": setting == "true", "request_categories": "/categories"}
    if policy.mode == "custom":
        template["custom_policy"] = policy.instructions
    response = await client.post(url, headers={"Authorization": "Bearer " + key}, json={
        "model": model, "messages": messages, "temperature": 0, "max_tokens": 1024,
        "chat_template_kwargs": template})
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        log.warning("nim_invalid_classification stage=%s code=incomplete", stage)
        raise ValueError("incomplete safety evaluation")
    try:
        label = parse_label(choice["message"]["content"], stage)
    except ValueError as error:
        log.warning("nim_invalid_classification stage=%s code=%s", stage, str(error))
        raise
    # Receipt contains no messages, policy, provider reasoning or credentials.
    log.info("nim_classification stage=%s label=%s", stage, label)
    return label == "safe"
