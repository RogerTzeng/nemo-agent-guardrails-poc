"""Strict provider boundary contracts; no live provider claims."""
import asyncio
from types import SimpleNamespace

import pytest
from nemoguardrails.types import LLMResponse

from guardrail_extension.native_strict import CheckedHTTPClient, probability, strict_alignscore, strict_grounding, strict_hf


@pytest.mark.parametrize('value,valid', [(True, False), (None, False), (float('nan'), False),
                                       (float('inf'), False), (-0.1, False), (1.1, False), (0, True), (1, True), (.5, True)])
def test_probability(value, valid):
    assert probability(value) is valid


@pytest.mark.parametrize('rail,data,status,valid', [
    ('polygraf', {'entities': []}, 200, True), ('polygraf', {'entities': None}, 200, False),
    ('polygraf', {}, 200, False), ('fiddler', {}, 200, False),
    ('fiddler', {'fdl_faithful_score': .9}, 200, True),
    ('fiddler', {'fdl_faithful_score': True}, 200, False),
    ('patronusai', {'results': []}, 200, False),
    ('patronusai', {'results': [{'evaluation_result': {'pass': True}}]}, 200, True),
    ('patronusai', {'results': [{'evaluation_result': {'pass': 'true'}}]}, 200, False),
    ('f5', {}, 500, False),
])
def test_http_wrapper_remembers_swallowed_errors(rail, data, status, valid):
    class Transport:
        closed = False
        async def request(self, *a, **kw):
            return SimpleNamespace(status_code=status, json=lambda: data)
        async def close(self):
            self.closed = True
    transport = Transport()
    async def run():
        client = CheckedHTTPClient(rail, transport)
        try:
            await client.request('POST', 'https://provider.invalid/ftl-response-faithfulness')
        except ValueError:
            pass  # Simulates an upstream action swallowing the failure.
        assert client.calls == 1
        assert client.failed is not valid
        await client.close()
    asyncio.run(run())
    assert transport.closed


@pytest.mark.parametrize('score,allowed', [(None, False), (True, False), (float('nan'), False), (.2, False), (.8, True)])
def test_alignscore_no_fallback(monkeypatch, score, allowed):
    async def request(*args, **kwargs):
        return score
    monkeypatch.setattr('nemoguardrails.library.factchecking.align_score.request.alignscore_request', request)
    config = SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(fact_checking=SimpleNamespace(parameters={'endpoint': 'http://score.invalid'}))))
    assert asyncio.run(strict_alignscore(config, {'bot_message': 'claim', 'relevant_chunks': ['evidence']})) is allowed
    assert not asyncio.run(strict_alignscore(config, {'bot_message': 'claim'}))


@pytest.mark.parametrize('results,allowed', [([], False), ([{'label': 'unknown', 'score': .1}], False),
    ([{'label': 'safe', 'score': float('nan')}], False), ([{'label': 'safe', 'score': .9}], True),
    ([{'label': 'unsafe', 'score': .9}], False)])
def test_classifier_missing_and_unknown_labels_block(monkeypatch, results, allowed):
    class Backend:
        async def classify(self, text):
            return results
    monkeypatch.setattr('nemoguardrails.library.hf_classifier.backends.get_backend', lambda *a, **k: Backend())
    settings = SimpleNamespace(blocked_labels=['unsafe'], threshold=.5)
    config = SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(hf_classifier={'demo': settings})))
    assert asyncio.run(strict_hf('text', config, {'classifier': 'demo', 'expected_labels': ['safe', 'unsafe']})) is allowed


@pytest.mark.parametrize('rail', ['hallucination', 'self_check.facts', 'patronusai'])
def test_missing_grounding_never_allows(rail):
    assert not asyncio.run(strict_grounding(rail, None, {'bot_message': 'claim'}, None, {}))


@pytest.mark.parametrize('rail,content,finish,allowed', [
    ('self_check.facts', 'yes', 'stop', True), ('self_check.facts', 'no', 'stop', False),
    ('self_check.facts', 'maybe', 'stop', False), ('self_check.facts', 'yes', 'length', False),
    ('patronusai', '{"SCORE":"PASS"}', 'stop', True), ('patronusai', '{"SCORE":"FAIL"}', 'stop', False),
    ('hallucination', 'yes', 'stop', True), ('hallucination', 'no', 'stop', False),
])
def test_grounding_strict_model_result(monkeypatch, rail, content, finish, allowed):
    async def generate(*a, **kw):
        return LLMResponse(content=content, finish_reason=finish)
    monkeypatch.setattr('guardrail_extension.native_strict.llm_call', generate)
    manager = SimpleNamespace(render_task_prompt=lambda **kw: 'classify')
    context = {'bot_message': 'claim', 'user_message': 'question', 'relevant_chunks': ['evidence'], '_last_bot_prompt': 'question'}
    assert asyncio.run(strict_grounding(rail, manager, context, object(), {'patronus_lynx': object()})) is allowed
