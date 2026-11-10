"""Inference for the user's fitted TF-IDF + Ridge strategy.

No OpenAI API calls. Model files live in models/. Keep text preprocessing
identical to the training notebook: only earnings-call-facts, stripped and
joined by newlines. HTTP failure/missing facts get an explicit neutral fallback;
a missing, corrupt, or incompatible model raises instead of hiding deployment errors.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import warnings

import httpx
import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion

FACTS_ID = "earnings-call-facts"
SUMMARY_TIMEOUT_SECONDS = 15.0
STRATEGY = "tfidf_word_char_ridge_v2_alpha10"


def facts_text(bundle: object) -> str:
    """Accept the live materials bundle, never serialize a whole event as input."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("items"), list):
        return ""
    for item in bundle["items"]:
        if isinstance(item, dict) and item.get("id") == FACTS_ID:
            facts = item.get("content")
            if not isinstance(facts, list):
                return ""
            return "\n".join(f.strip() for f in facts if isinstance(f, str) and f.strip())
    return ""


def _model_path() -> Path:
    override = os.environ.get("RIDGE_MODEL_PATH")
    return Path(override) if override else Path(__file__).resolve().parent / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    path = _model_path()
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise RuntimeError(
            f"scikit-learn version mismatch: need {metadata['sklearn_version']}, "
            f"got {sklearn.__version__}. Install requirements-ridge.txt."
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Model checksum mismatch: copy model.joblib and metadata.json together.")
    # Load only this verified model artifact that you trained, never arbitrary remote pickles.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if metadata.get("strategy") != STRATEGY:
        raise RuntimeError("Unexpected strategy metadata.")
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["features", "ridge"]:
        raise RuntimeError("Expected a features/ridge Pipeline.")
    features = model["features"]
    if not isinstance(features, FeatureUnion) or not isinstance(model["ridge"], Ridge):
        raise RuntimeError("Unexpected estimator types.")
    if model["ridge"].alpha != 10.0 or metadata["alpha"] != 10.0:
        raise RuntimeError("Expected Ridge alpha=10.")
    if features.transformer_weights != {"word": 1.0, "char": 0.5}:
        raise RuntimeError("Unexpected word/character weights.")
    if [name for name, _ in features.transformer_list] != ["word", "char"]:
        raise RuntimeError("Expected word and char transformers.")
    transformers = dict(features.transformer_list)
    for name, analyzer, ngrams, min_df in [("word", "word", (1,2), 2), ("char", "char_wb", (3,5), 3)]:
        tf = transformers[name]
        if not isinstance(tf, TfidfVectorizer) or tf.analyzer != analyzer or tf.ngram_range != ngrams:
            raise RuntimeError("Unexpected vectorizer configuration.")
        if tf.min_df != min_df or tf.max_features != 30000 or not tf.sublinear_tf:
            raise RuntimeError("Unexpected vectorizer parameters.")
        if len(tf.vocabulary_) != metadata["feature_counts"][name]:
            raise RuntimeError("Unexpected vocabulary size.")
    if sum(metadata["feature_counts"].values()) != model["ridge"].n_features_in_:
        raise RuntimeError("Feature count mismatch.")
    print(f"[MODEL] loaded strategy={STRATEGY} sklearn={sklearn.__version__}")
    return model


def score_text(text: str) -> float:
    # Check the model even for missing-text events so a broken deployment is visible.
    model = load_model()
    if not re.search(r"[A-Za-z]{2}", text):
        print("[WARN] Missing usable earnings-call facts; using neutral 0.5.")
        return 0.5
    score = float(model.predict([text])[0])
    if not math.isfinite(score):
        raise RuntimeError("The fitted model returned a non-finite prediction.")
    return float(np.clip(score, 0.0, 1.0))


def _tickers(event: dict) -> list[str]:
    assets = event.get("focal_assets")
    if not isinstance(assets, list):
        raise ValueError("Event has no focal_assets list.")
    tickers = []
    for asset in assets:
        ticker = asset.get("identifier_value") if isinstance(asset, dict) else None
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Invalid focal asset identifier.")
        if ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        raise ValueError("Event has no focal assets.")
    return tickers


def _rows(tickers: list[str], value: float) -> list[dict]:
    return [{"identifier_value": ticker, "predicted_percentile": value} for ticker in tickers]


def predict_from_materials(event: dict, bundle: object) -> list[dict]:
    """Pure inference entrypoint for checks; never submits a prediction."""
    tickers = _tickers(event)
    if event.get("event_type") == "TEST":
        return _rows(tickers, 0.5)
    value = score_text(facts_text(bundle))
    # Training gives each focal asset the same event-level facts; preserve that behavior.
    return _rows(tickers, value)


def predict(event: dict) -> list[dict]:
    tickers = _tickers(event)
    if event.get("event_type") == "TEST":
        return _rows(tickers, 0.5)
    load_model()
    url = event.get("information_url")
    if not isinstance(url, str) or not url:
        print("[WARN] No information_url; using neutral 0.5.")
        return _rows(tickers, 0.5)
    try:
        response = httpx.get(url, timeout=SUMMARY_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
        bundle = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Do not include signed material URLs or response bodies in logs.
        print(f"[WARN] Material fetch failed ({type(exc).__name__}); using neutral 0.5.")
        return _rows(tickers, 0.5)
    result = predict_from_materials(event, bundle)
    print(f"[PREDICT] strategy={STRATEGY} assets={tickers} score={result[0]['predicted_percentile']:.6f}")
    return result


def self_check() -> dict:
    """Load the actual artifact and exercise live-format inference, without API calls."""
    cases = [
        "Revenue increased and management raised full year guidance.",
        "Revenue declined and management cut full year guidance.",
    ]
    event = {"event_type": "EARNINGS_RELEASE", "focal_assets": [{"identifier_value": "CHECK"}]}
    output = [predict_from_materials(event, {"items": [{"id": FACTS_ID, "content": [text]}]})[0]["predicted_percentile"]
              for text in cases]
    expected = np.clip(load_model().predict(cases), 0, 1)
    if not np.allclose(output, expected, rtol=0, atol=1e-12):
        raise RuntimeError("Live-format predictions differ from the fitted model.")
    meta = json.loads((_model_path().parent / "metadata.json").read_text())
    if meta.get("smoke_texts") != cases or not np.allclose(output, meta["smoke_predictions"], rtol=0, atol=1e-8):
        raise RuntimeError("Predictions differ from training export; check dependencies and files.")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in output):
        raise RuntimeError("Invalid smoke-test output.")
    if abs(output[0] - output[1]) < 1e-8:
        raise RuntimeError("Smoke test did not distinguish the two fixed examples.")
    if predict_from_materials(event, {"items": []})[0]["predicted_percentile"] != 0.5:
        raise RuntimeError("Missing-text fallback failed.")
    return {"ok": True, "strategy": STRATEGY, "sklearn_version": sklearn.__version__,
            "alpha": float(load_model()["ridge"].alpha), "feature_count": int(load_model()["ridge"].n_features_in_),
            "smoke_predictions": output, "submitted_to_competition": False}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
