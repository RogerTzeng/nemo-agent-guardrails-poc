import asyncio
import hmac
import os
import logging

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .dispatcher import dispatch
from .nemo_checks_client import NemoChecksClient
from .registry import ProfileRegistry, RegistryError
from .native_catalog import public_catalog
from .schemas import CheckRequest
from .policy_conflicts import PolicyConflict, PolicyReviewUnavailable


def install(app, root, policy_reviewer=None):
    # Upstream server constructs LLMRails(verbose=True). Silence its payload-rich
    # logs using standard Python logging configuration, without changing NeMo.
    for name in ["nemoguardrails", *[n for n in logging.Logger.manager.loggerDict
                                    if n.startswith("nemoguardrails.")]]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    audit = logging.getLogger("guardrail_extension")
    audit.setLevel(logging.INFO)
    audit.propagate = False
    if not audit.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        audit.addHandler(handler)
    key = os.environ.get("GUARDRAIL_ADMIN_KEY", "")
    if len(key) < 24:
        raise ValueError("GUARDRAIL_ADMIN_KEY must contain at least 24 characters")
    registry = ProfileRegistry(root, policy_reviewer=policy_reviewer)
    native = NemoChecksClient(os.environ.get("GUARDRAIL_NATIVE_URL", "http://127.0.0.1:8000"))
    router = APIRouter()

    @router.get("/v1/guardrail-catalog")
    async def catalog():
        return {"rails": public_catalog()}

    @router.post("/v1/check")
    async def check(request: CheckRequest):
        snapshot = registry.snapshot
        return await dispatch(request, snapshot, native)

    @router.post("/v1/admin/reload")
    async def reload(x_admin_key: str = Header(default="")):
        if not hmac.compare_digest(x_admin_key, key):
            raise HTTPException(403, "unauthorized")
        try:
            snapshot = await asyncio.to_thread(registry.reload)
        except PolicyConflict:
            return JSONResponse(status_code=409, content={"detail": {"code": "POLICY_CONFLICT"}})
        except PolicyReviewUnavailable:
            return JSONResponse(status_code=503, content={"detail": {"code": "POLICY_REVIEW_UNAVAILABLE"}})
        except RegistryError as error:
            raise HTTPException(422, str(error)) from None
        except ValidationError as error:
            raise HTTPException(422, error.errors(include_input=False, include_context=False,
                                                  include_url=False)) from None
        except Exception:
            # Don't echo validation inputs/policy text or filesystem contents.
            raise HTTPException(422, "invalid registry; check schema and agent/profile/check references") from None
        return {"profile_version": snapshot.version}

    app.include_router(router)
