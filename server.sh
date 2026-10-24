#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python - <<'PY'
import os
from pathlib import Path
import secrets
import subprocess
import sys

from guardrail_extension.env_config import read_live_env

env = {**os.environ, **read_live_env(Path('.env'))}
env.setdefault('GUARDRAIL_ADMIN_KEY', secrets.token_hex(32))
env.setdefault('GUARDRAIL_NATIVE_URL', 'http://127.0.0.1:8000')
subprocess.run(
    [str(Path(sys.executable).with_name('nemoguardrails')), 'server',
     '--config', './configs', '--port', '8000', '--disable-chat-ui'],
    env=env, check=True,
)
PY
