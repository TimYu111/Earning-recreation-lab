"""Build a small synthetic artifact for tests. No archive or credential required."""
import hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pytest
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

@pytest.fixture(scope='session')
def synthetic_model(tmp_path_factory):
    root=tmp_path_factory.mktemp('synthetic-model')
    cases=['Revenue increased and management raised full year guidance.',
           'Revenue declined and management cut full year guidance.']
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True,dtype=np.float64)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=30000,sublinear_tf=True,dtype=np.float64)
    features=FeatureUnion([('word',word),('char',char)],transformer_weights={'word':1.,'char':.5},n_jobs=1)
    model=Pipeline([('features',features),('ridge',Ridge(alpha=10.,solver='lsqr'))])
    model.fit(cases*20,[.75,.25]*20)
    path=root/'model.joblib';joblib.dump(model,path)
    meta=dict(strategy='tfidf_word_char_ridge_v2_alpha10',alpha=10.,sklearn_version=sklearn.__version__,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              feature_counts={name:len(tf.vocabulary_) for name,tf in features.transformer_list},
              smoke_texts=cases,smoke_predictions=np.clip(model.predict(cases),0,1).tolist())
    (root/'metadata.json').write_text(json.dumps(meta))
    return path

@pytest.fixture(autouse=True)
def model_environment(synthetic_model,monkeypatch):
    monkeypatch.setenv('RIDGE_MODEL_PATH',str(synthetic_model))
