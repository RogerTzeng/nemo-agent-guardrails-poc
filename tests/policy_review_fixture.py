"""Offline orchestration double; NOT a semantic model or acceptance evidence.

Only explicit fixture markers model conflict/unavailability. Real conflict quality
is verified separately against ChatOpenAI/NIM in the opt-in live suite.
"""
from guardrail_extension.policy_conflicts import PolicyConflict, PolicyReviewUnavailable


class OfflinePolicyReviewer:
    def review(self, candidate, previous):
        if any("REVIEW_UNAVAILABLE_FIXTURE" in p for p in candidate.custom):
            raise PolicyReviewUnavailable()
        if any("ALLOW_DIRECT_SQL_FIXTURE" in p for p in candidate.custom):
            raise PolicyConflict()


def install_offline_reviewer(configs):
    (configs / "config.py").write_text('''from pathlib import Path
from guardrail_extension.api import install
from policy_review_fixture import OfflinePolicyReviewer

def init(app):
    install(app, Path(__file__).resolve().parent, policy_reviewer=OfflinePolicyReviewer())
''')
