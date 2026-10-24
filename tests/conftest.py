from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import httpx
import pytest

from guardrail_extension.company import compose_policy, load_company, SERVICE_ROOT
from guardrail_extension.schemas import Policy
from guardrail_extension.registry import ProfileRegistry
from policy_review_fixture import OfflinePolicyReviewer, install_offline_reviewer

ROOT = Path(__file__).resolve().parents[1]
KEY = "test-administrator-secret-key"

BASE_STUB_POLICIES = {
    "禁止洩漏機密配方；允許一般設備說明。": ["機密配方", "secret recipe"],
    "禁止繞過安全互鎖；允許一般製程概念。": ["繞過安全互鎖", "bypass interlock"],
    "禁止機密配方及繞過安全互鎖": ["機密配方", "繞過安全互鎖"],
}
# Explicit offline rules; no natural-language interpretation or default taxonomy simulation.
_company, _baseline = load_company(SERVICE_ROOT)
COMPANY_TERMS = ["execute SQL", "run SQL", "直接執行 SQL"]
STUB_POLICIES = json.dumps({
    "禁止機密配方": ["機密配方"],
    **BASE_STUB_POLICIES,
    compose_policy(_company, _baseline, []).instructions: COMPANY_TERMS,
    **{compose_policy(_company, _baseline, [Policy(instructions=prompt)]).instructions:
       COMPANY_TERMS + terms for prompt, terms in BASE_STUB_POLICIES.items()},
})


@pytest.fixture(autouse=True)
def private_approval_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDRAIL_STATE_DIR", str(tmp_path / "approvals"))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "service", tmp_path / "service", ignore=shutil.ignore_patterns("state"))
    monkeypatch.setenv("GUARDRAIL_SERVICE_CONFIG", str(tmp_path / "service"))
    return ProfileRegistry(tmp_path / "configs", policy_reviewer=OfflinePolicyReviewer())


@contextmanager
def official_server(directory, *, live=False, values=None):
    configs = directory / "configs"
    shutil.copytree(ROOT / "configs", configs, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "service", directory / "service", ignore=shutil.ignore_patterns("state"))
    if not live:
        install_offline_reviewer(configs)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, **(values or {}), "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "tests")]), "GUARDRAIL_ADMIN_KEY": KEY,
           "GUARDRAIL_NATIVE_URL": url, "SAFETY_EVALUATOR": "nim_content_safety" if live else "stub", "NO_COLOR": "1",
           "SAFETY_STUB_POLICIES": STUB_POLICIES, "GUARDRAIL_SERVICE_CONFIG": str(directory / "service"),
           "GUARDRAIL_STATE_DIR": str(directory / "approvals")}
    log_path = directory / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([str(Path(sys.executable).parent / "nemoguardrails"),
            "server", "--config", str(configs), "--port", str(port), "--disable-chat-ui"],
            cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        last_check = [0.0]
        def pace_live_checks(request):
            if live and request.url.path == "/v1/check":
                interval = max(0.0, float(os.environ.get("LIVE_CHECK_INTERVAL_SECONDS", "3")))
                delay = interval - (time.monotonic() - last_check[0])
                if delay > 0:
                    time.sleep(delay)
                last_check[0] = time.monotonic()
        client = httpx.Client(base_url=url, timeout=120, trust_env=False,
                             event_hooks={"request": [pace_live_checks]})
        try:
            for _ in range(1800 if live else 150):
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    if client.get("/v1/rails/configs").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                pytest.fail("server startup timed out: " + log_path.read_text())
            yield client, configs, log_path
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture(scope="module")
def server(tmp_path_factory, request):
    with official_server(tmp_path_factory.mktemp("official-server"),
                         live=request.module.__name__.startswith("test_live_")) as running:
        yield running


def check_body(agent="yield_agent", text="機密配方", stage="before_model"):
    return {"agent_id": agent, "stage": stage, "payload": {"text": text},
            "context": {"request_id": "test001"}}


def live_check(client, log, body):
    before = log.read_text().count("nim_classification stage=")
    response = client.post("/v1/check", json=body)
    response.raise_for_status()
    result = response.json()
    assert log.read_text().count("nim_classification stage=") == before + 1, (
        "Expected one parsed NIM classification; provider failures cannot count as policy denials"
    )
    assert sum(check["rail"] == "content_safety" for check in result["checks"]) == 1
    assert all(check["code"] in {"PASSED", "BLOCKED"} for check in result["checks"])
    return result


def native_body(text="一般概念", rail_type="input"):
    messages = [{"role": "user" if rail_type == "input" else "assistant", "content": text}]
    if rail_type == "output":
        messages.insert(0, {"role": "user", "content": ""})
    return {"model": "", "messages": messages, "guardrails": {"config_ids": ["content_safety_shared"],
            "rail_types": [rail_type], "context": {"effective_policy": {
                "mode": "custom", "instructions": "禁止機密配方"}}}}
