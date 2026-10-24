"""Real ChatOpenAI wire parsing with mock HTTP; atomic registry orchestration."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
from threading import Event

import httpx
from langchain_openai import ChatOpenAI
import pytest
import yaml

from conftest import ROOT, KEY, check_body
from guardrail_extension.dispatcher import dispatch
from guardrail_extension.policy_conflicts import PolicyContext, PolicyConflict, PolicyReviewer, PolicyReviewUnavailable
from guardrail_extension.registry import ProfileRegistry
from guardrail_extension.schemas import CheckRequest


@pytest.mark.parametrize("content,finish,status,error", [
    ('{"conflict":false}', 'stop', 200, None),
    ('{"conflict":true}', 'stop', 200, PolicyConflict),
    ('{"conflict":"false"}', 'stop', 200, PolicyReviewUnavailable),
    ('{"conflict":0}', 'stop', 200, PolicyReviewUnavailable),
    ('{"conflict":false,"reason":"private"}', 'stop', 200, PolicyReviewUnavailable),
    ('{"conflict":true,"conflict":false}', 'stop', 200, PolicyReviewUnavailable),
    ('```json\n{"conflict":false}\n```', 'stop', 200, PolicyReviewUnavailable),
    ('{}', 'stop', 200, PolicyReviewUnavailable), ('not json', 'stop', 200, PolicyReviewUnavailable),
    ('{"conflict":false}', 'length', 200, PolicyReviewUnavailable),
    ('{"conflict":false}', 'tool_calls', 200, PolicyReviewUnavailable),
    ('', 'stop', 429, PolicyReviewUnavailable), ('', 'stop', 500, PolicyReviewUnavailable),
])
def test_chatopenai_strict_verdict(content, finish, status, error):
    def provider(request):
        body = json.loads(request.content)
        assert body['chat_template_kwargs']['enable_thinking'] is False
        data = json.loads(body['messages'][1]['content'])
        assert data['mandatory_company_policy'] == 'PRIVATE: forbid SQL execution'
        assert data['active_agent_custom_policies'] == ['Forbid secrets']
        assert data['candidate_agent_custom_policies'] == ['Also forbid keys']
        if status != 200:
            return httpx.Response(status, json={'error': {'message': 'PRIVATE_PROVIDER_ERROR'}})
        return httpx.Response(200, json={'id': 'review-test', 'object': 'chat.completion',
            'model': 'review-model', 'choices': [{'index': 0, 'finish_reason': finish,
                                                'message': {'role': 'assistant', 'content': content}}]})
    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        model = ChatOpenAI(model='review-model', api_key='test', base_url='https://nim.invalid/v1',
                           http_client=client, max_retries=0,
                           extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        reviewer = PolicyReviewer(model)
        args = (PolicyContext('PRIVATE: forbid SQL execution', None, ('Also forbid keys',)),
                PolicyContext('PRIVATE: forbid SQL execution', None, ('Forbid secrets',)))
        if error:
            with pytest.raises(error) as caught:
                reviewer.review(*args)
            assert 'PRIVATE' not in str(caught.value)
        else:
            reviewer.review(*args)


def test_missing_model_configuration_cannot_bypass(monkeypatch):
    for key in ('POLICY_CONFLICT_MODEL', 'AGENT_MODEL', 'LLM'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(PolicyReviewUnavailable):
        PolicyReviewer().review(PolicyContext('company', None, ('candidate',)), None)


class RecordingReviewer:
    def __init__(self):
        self.calls = []
        self.reject = False
        self.unavailable = False

    def review(self, candidate, previous):
        self.calls.append((candidate, previous))
        if self.reject:
            raise PolicyConflict()
        if self.unavailable:
            raise PolicyReviewUnavailable()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'configs', tmp_path / 'configs')
    shutil.copytree(ROOT / 'service', tmp_path / 'service')
    monkeypatch.setenv('GUARDRAIL_SERVICE_CONFIG', str(tmp_path / 'service'))
    reviewer = RecordingReviewer()
    registry = ProfileRegistry(tmp_path / 'configs', policy_reviewer=reviewer)
    reviewer.calls.clear()
    return registry, reviewer


def update(registry, mutate):
    path = registry.root / 'guardrails.yaml'
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True))


@pytest.mark.parametrize('failure', ['conflict', 'unavailable'])
def test_rejected_policy_never_publishes(setup, failure):
    registry, reviewer = setup
    old = registry.snapshot
    update(registry, lambda d: d['checks']['semiconductor_safety'].update(prompt='candidate'))
    reviewer.reject = failure == 'conflict'
    reviewer.unavailable = failure == 'unavailable'
    with pytest.raises(PolicyConflict if reviewer.reject else PolicyReviewUnavailable):
        registry.reload()
    assert registry.snapshot is old


def test_identical_and_regex_only_reload_no_model_call(setup):
    registry, reviewer = setup
    registry.reload()
    update(registry, lambda d: d['checks']['no_email'].update(deny_patterns=['word']))
    registry.reload()
    assert reviewer.calls == []


def test_mapping_rename_and_default_inclusion_are_reviewed(setup):
    registry, reviewer = setup
    def mutate(data):
        data['checks']['new_rule'] = {'type': 'content_safety', 'mode': 'custom', 'prompt': 'candidate'}
        data['profiles']['renamed'] = {'input': ['new_rule', 'model_default'], 'output': ['new_rule']}
        data['agents']['yield_agent'] = 'renamed'
    update(registry, mutate)
    registry.reload()
    assert any(c.custom == ('candidate',) and p and '機密配方' in p.custom[0] for c, p in reviewer.calls)
    assert any(c.default and 'S24: Economic Harm.' in c.default for c, p in reviewer.calls)
    for stage in ('before_model', 'after_model'):
        safety = [s for s in registry.snapshot.profiles['renamed'][stage] if s.rail == 'content_safety']
        assert len(safety) == 1
        assert 'candidate' in registry.snapshot.checks[safety[0].policy_id].instructions


def test_concurrent_reload_and_requests_keep_valid_snapshot(setup):
    registry, reviewer = setup
    entered, release = Event(), Event()
    class BlockingReviewer(RecordingReviewer):
        def review(self, candidate, previous):
            super().review(candidate, previous)
            if candidate.custom == ('HOLD',):
                entered.set()
                assert release.wait(10)
    registry.policy_reviewer = reviewer = BlockingReviewer()
    update(registry, lambda d: d['checks']['semiconductor_safety'].update(prompt='HOLD'))
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(registry.reload)
        assert entered.wait(10)
        assert registry.snapshot.version == 1
        class Native:
            async def check(self, body):
                assert 'HOLD' not in body['guardrails']['context']['effective_policy']['instructions']
                return {'status': 'passed', 'content': 'hello'}
        assert asyncio.run(dispatch(CheckRequest(**check_body(text='hello')), registry.snapshot, Native()))['profile_version'] == 1
        update(registry, lambda d: d['checks']['semiconductor_safety'].update(prompt='NEXT'))
        second = executor.submit(registry.reload)
        release.set()
        assert first.result(10).version == 2
        assert second.result(10).version == 3
    assert any(c.custom == ('NEXT',) and p.custom == ('HOLD',) for c, p in reviewer.calls)


def test_server_conflict_codes_and_previous_policy(server):
    client, root, log = server
    path = root / 'guardrails.yaml'
    original = path.read_text()
    try:
        before = client.post('/v1/check', json=check_body()).json()
        for marker, status, code in [('ALLOW_DIRECT_SQL_FIXTURE', 409, 'POLICY_CONFLICT'),
                                     ('REVIEW_UNAVAILABLE_FIXTURE', 503, 'POLICY_REVIEW_UNAVAILABLE')]:
            data = yaml.safe_load(original)
            data['checks']['semiconductor_safety']['prompt'] = marker
            path.write_text(yaml.safe_dump(data))
            response = client.post('/v1/admin/reload', headers={'X-Admin-Key': KEY})
            assert response.status_code == status
            assert response.json() == {'detail': {'code': code}}
            after = client.post('/v1/check', json=check_body()).json()
            assert after['profile_version'] == before['profile_version']
            assert after['decision'] == before['decision'] == 'DENY'
            assert marker not in response.text and marker not in log.read_text()
    finally:
        path.write_text(original)
        client.post('/v1/admin/reload', headers={'X-Admin-Key': KEY}).raise_for_status()


def test_restart_cannot_bypass_previously_rejected_update(setup):
    registry, reviewer = setup
    state_before = registry._state.path.read_bytes()
    update(registry, lambda d: d['checks']['semiconductor_safety'].update(prompt='candidate'))
    reviewer.reject = True
    with pytest.raises(PolicyConflict):
        registry.reload()
    assert registry._state.path.read_bytes() == state_before
    # Fresh registry must compare against the last approved custom policy.
    restarted_reviewer = RecordingReviewer()
    restarted_reviewer.reject = True
    with pytest.raises(PolicyConflict):
        ProfileRegistry(registry.root, policy_reviewer=restarted_reviewer)
    assert any(p and '機密配方' in p.custom[0] for c, p in restarted_reviewer.calls)


def test_unchanged_restart_reuses_private_approval(setup):
    registry, reviewer = setup
    restarted = ProfileRegistry(registry.root, policy_reviewer=reviewer)
    assert reviewer.calls == []
    assert restarted._state.path.stat().st_mode & 0o777 == 0o600


def test_state_write_failure_never_publishes(setup, monkeypatch):
    registry, reviewer = setup
    before = registry.snapshot
    state = registry._state.path.read_bytes()
    update(registry, lambda d: d['checks']['semiconductor_safety'].update(prompt='candidate'))
    def fail(*args):
        raise OSError('private path')
    monkeypatch.setattr('guardrail_extension.policy_state.os.replace', fail)
    with pytest.raises(ValueError, match='cannot persist private policy approval state'):
        registry.reload()
    assert registry.snapshot is before
    assert registry._state.path.read_bytes() == state
    assert len(list(registry._state.path.parent.iterdir())) == 1


def test_corrupt_approval_state_rejects_startup(setup):
    registry, reviewer = setup
    registry._state.path.write_text('{broken')
    with pytest.raises(ValueError, match='invalid private policy approval state'):
        ProfileRegistry(registry.root, policy_reviewer=reviewer)
