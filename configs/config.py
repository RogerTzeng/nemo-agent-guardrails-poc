from pathlib import Path

from guardrail_extension.api import install


def init(app):
    install(app, Path(__file__).resolve().parent)
