import asyncio
import os
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx


class NemoChecksClient:
    def __init__(self, url, transport=None):
        parsed = urlparse(url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path
                or not parsed.port or parsed.username or parsed.query or parsed.fragment):
            raise ValueError("native URL must be http://127.0.0.1:PORT")
        self.url, self.transport = url, transport
        self.timeout = float(os.environ.get("GUARDRAIL_CHECK_TIMEOUT", "12"))
        self.client = None
        self.users = 0
        self.semaphore = asyncio.Semaphore(16)

    @asynccontextmanager
    async def lease(self):
        # Shared by overlapping requests; close when the last evaluation releases
        # it. Root init runs inside an entered lifespan, with no teardown callback.
        # No idle client survives shutdown and no private lifecycle API is used.
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout, trust_env=False,
                transport=self.transport, limits=httpx.Limits(max_connections=16))
        client = self.client
        self.users += 1
        try:
            yield client
        finally:
            self.users -= 1
            if self.users == 0:
                self.client = None
                await client.aclose()

    async def check(self, body):
        async with asyncio.timeout(self.timeout):
            async with self.lease() as client:
                async with self.semaphore:
                    response = await client.post("/v1/checks", json=body)
                    response.raise_for_status()
                    return response.json()
