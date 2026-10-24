import asyncio

import httpx
import pytest
import yaml

from conftest import check_body
from guardrail_extension.dispatcher import dispatch
from guardrail_extension.nemo_checks_client import NemoChecksClient
from guardrail_extension.schemas import CheckRequest


@pytest.mark.parametrize("kind", [500, 429, "network", "timeout", "json", "unknown", "modified", "missing", "partial"])
def test_native_failures(registry, kind):
    async def run():
        async def handler(request):
            if kind == "network":
                raise httpx.ConnectError("unreachable")
            if kind == "timeout":
                raise httpx.ReadTimeout("timeout")
            if isinstance(kind, int):
                return httpx.Response(kind)
            if kind == "json":
                return httpx.Response(200, text="not json")
            return httpx.Response(200, json={"unknown": {"status": "success"},
                "modified": {"status": "modified"}, "missing": {}, "partial": {"status": "passed"}}[kind])
        native = NemoChecksClient("http://127.0.0.1:8000", transport=httpx.MockTransport(handler))
        result = await dispatch(CheckRequest(**check_body()), registry.snapshot, native)
        assert result["decision"] == "DENY"
        assert native.client is None and native.users == 0
    asyncio.run(run())


def test_inflight_request_keeps_original_snapshot(registry):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        class Native:
            async def check(self, body):
                started.set()
                await release.wait()
                assert "禁止洩漏機密配方；允許一般設備說明。" in body["guardrails"]["context"]["effective_policy"]["instructions"]
                return {"status": "passed", "content": "hello"}
        task = asyncio.create_task(dispatch(CheckRequest(**check_body(text="hello")), registry.snapshot, Native()))
        await started.wait()
        path = registry.root / "guardrails.yaml"
        data = yaml.safe_load(path.read_text())
        data["checks"]["semiconductor_safety"]["prompt"] = "changed"
        path.write_text(yaml.safe_dump(data))
        registry.reload()
        release.set()
        assert (await task)["profile_version"] == 1
        assert registry.snapshot.version == 2
    asyncio.run(run())
