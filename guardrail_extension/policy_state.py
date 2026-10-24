"""Private, atomically persisted approval baseline for one registry path."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from pydantic import TypeAdapter

from .company import service_root
from .policy_conflicts import PolicyContext


class PolicyState:
    def __init__(self, registry_root):
        directory = Path(os.environ.get("GUARDRAIL_STATE_DIR", str(service_root() / "state")))
        key = hashlib.sha256(str(registry_root.resolve()).encode()).hexdigest()
        self.path = directory / f"{key}.json"

    def load(self):
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            if set(data) != {"format", "contexts"} or data["format"] != 1:
                raise ValueError()
            contexts = {}
            for entry in data["contexts"]:
                scope = tuple(entry["scope"])
                if not scope or any(not isinstance(p, str) for p in scope) or scope in contexts:
                    raise ValueError()
                contexts[scope] = TypeAdapter(PolicyContext).validate_python(entry["context"])
            return contexts
        except Exception:
            raise ValueError("invalid private policy approval state") from None

    def save(self, contexts):
        temporary = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            data = {"format": 1, "contexts": [{"scope": list(scope), "context": asdict(context)}
                                             for scope, context in contexts.items()]}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            raise ValueError("cannot persist private policy approval state") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
