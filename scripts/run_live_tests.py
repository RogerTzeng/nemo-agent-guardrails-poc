"""Load local credentials without shell execution, then run official-server tests."""
import os
from pathlib import Path
import secrets
import subprocess
import sys

from guardrail_extension.env_config import read_live_env

root = Path(__file__).resolve().parents[1]
env = {**os.environ, **read_live_env(root / ".env"), "RUN_LIVE_MODEL": "1"}
for key in ["SAFETY_BASE_URL", "SAFETY_MODEL", "SAFETY_API_KEY"]:
    if not env.get(key):
        raise SystemExit(f"Missing {key} in .env")
env.setdefault("GUARDRAIL_ADMIN_KEY", secrets.token_hex(32))
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q",
    "tests/test_live_model_optional.py", "tests/test_live_profile_modes_optional.py", "tests/test_live_company_optional.py", "tests/test_live_native_optional.py", "tests/test_live_conflicts_optional.py", "--tb=short"], cwd=root, env=env))
