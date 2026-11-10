"""Offline inference tests using a synthetic fitted model; no competition credentials."""
from pathlib import Path
import json

import httpx
import numpy as np
import pytest

import predict as strategy


@pytest.fixture(autouse=True)
def fresh_model_cache():
    clear_cache = strategy.load_model.cache_clear
    clear_cache()
    yield
    clear_cache()


def event(kind="EARNINGS_RELEASE"):
    return {"event_type": kind, "event_id": "offline-example",
            "focal_assets": [{"identifier_value":"AAPL"}, {"identifier_value":"MSFT"}],
            "information_url":"https://example.test/signed-materials"}


def bundle():
    return {"items": [
        {"id":"earnings-preview", "content":"This must not enter the trained text features."},
        {"id":"option-implied-stats", "content":{"value":999}},
        {"id":"earnings-call-facts", "content":[" Revenue grew 12%. ","",None,"Guidance increased."]},
    ], "event_returns":{"car1":999}, "baseline_predictions":{"do_not_use":1}}


def test_same_text_as_training_and_no_other_fields():
    assert strategy.facts_text(bundle()) == "Revenue grew 12%.\nGuidance increased."
    assert strategy.facts_text({"items":"wrong"}) == ""
    assert strategy.facts_text(None) == ""


def test_actual_model_prediction_matches_pipeline_without_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = strategy.predict_from_materials(event(),bundle())
    expected = float(np.clip(strategy.load_model().predict([strategy.facts_text(bundle())])[0],0,1))
    assert result == [
        {"identifier_value":"AAPL", "predicted_percentile":expected},
        {"identifier_value":"MSFT", "predicted_percentile":expected},
    ]
    assert expected != 0.5


def test_http_to_inference(monkeypatch):
    def get(url, **kwargs):
        assert kwargs['timeout'] == 15.0
        return httpx.Response(200,json=bundle(),request=httpx.Request('GET',url))
    monkeypatch.setattr(strategy.httpx,'get',get)
    assert strategy.predict(event()) == strategy.predict_from_materials(event(),bundle())


def test_http_failure_has_explicit_neutral_fallback(monkeypatch,capsys):
    def fail(*args,**kwargs):
        raise httpx.ReadTimeout('https://example.test/private-signed-token')
    monkeypatch.setattr(strategy.httpx,'get',fail)
    assert all(p['predicted_percentile']==0.5 for p in strategy.predict(event()))
    captured=capsys.readouterr().out
    assert '[WARN]' in captured
    assert 'private-signed-token' not in captured


def test_missing_text_fallback():
    for data in ({}, {'items':[]}, {'items':[{'id':'earnings-call-facts','content':['123','!']}]}):
        assert all(p['predicted_percentile']==0.5 for p in strategy.predict_from_materials(event(),data))


def test_test_events_do_not_fetch_or_require_model(monkeypatch):
    monkeypatch.setenv('RIDGE_MODEL_PATH','/nonexistent/model.joblib')
    def forbidden(*args,**kwargs):
        raise AssertionError('Network call during TEST')
    monkeypatch.setattr(strategy.httpx,'get',forbidden)
    assert all(p['predicted_percentile']==0.5 for p in strategy.predict(event('TEST')))


def test_corrupt_model_stops_before_unpickling(tmp_path,monkeypatch):
    original = strategy._model_path()
    (tmp_path/'metadata.json').write_text((original.parent/'metadata.json').read_text())
    (tmp_path/'model.joblib').write_bytes(b'corrupt model')
    monkeypatch.setenv('RIDGE_MODEL_PATH',str(tmp_path/'model.joblib'))
    with pytest.raises(RuntimeError,match='checksum'):
        strategy.load_model()


def test_version_mismatch_is_explicit(monkeypatch):
    monkeypatch.setattr(strategy.sklearn,'__version__','0.0.0')
    with pytest.raises(RuntimeError,match='version mismatch'):
        strategy.load_model()


def test_synthetic_model_self_check():
    report = strategy.self_check()
    assert report['ok'] is True
    assert report['alpha'] == 10
    assert report['feature_count'] > 0
    assert report['submitted_to_competition'] is False


def test_nonfinite_prediction_is_not_submitted(monkeypatch):
    class Broken:
        def predict(self,*args): return np.array([np.nan])
    monkeypatch.setattr(strategy,'load_model',lambda:Broken())
    with pytest.raises(RuntimeError,match='non-finite'):
        strategy.score_text('Revenue increased.')


def test_output_clipping(monkeypatch):
    class Extreme:
        def predict(self,*args): return np.array([1.5])
    monkeypatch.setattr(strategy,'load_model',lambda:Extreme())
    assert strategy.score_text('Revenue increased.') == 1.0
