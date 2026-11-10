"""Train the frozen v2 candidate from the local archive. Never calls the API."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, hashlib, json, math, re, sys, tempfile, shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from examples.archive import load_archive
from examples.summary import facts_from_disclosure
from examples.scoring import percentile_ranks

QUARTERS = ['2025Q4', '2026Q1', '2026Q2', '2026Q3']
STRATEGY = 'tfidf_word_char_ridge_v2_alpha10'
SMOKE_TEXTS = ['Revenue increased and management raised full year guidance.',
               'Revenue declined and management cut full year guidance.']

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def facts_text(record):
    disclosure = record.get('disclosure')
    if not isinstance(disclosure, dict):
        return ''
    facts = facts_from_disclosure(record)
    return '\n'.join(f.strip() for f in facts if isinstance(f, str) and f.strip())

def make_outcomes(frame):
    rows, skipped = [], Counter()
    for rec in frame.to_dict('records'):
        if rec.get('event_type') != 'EARNINGS_RELEASE':
            skipped['not_earnings'] += 1
            continue
        if not isinstance(rec.get('event_id'), str):
            skipped['missing_event_id'] += 1
            continue
        quarter = rec.get('quarter')
        if not isinstance(quarter, str) or not re.fullmatch(r'\d{4}Q[1-4]', quarter):
            skipped['invalid_quarter'] += 1
            continue
        assets, returns = rec.get('focal_assets'), rec.get('event_returns')
        if not isinstance(assets, list) or not isinstance(returns, dict):
            skipped['missing_assets_or_returns'] += 1
            continue
        text = facts_text(rec)
        metrics = rec.get('metrics')
        es = metrics.get('earnings_surprise') if isinstance(metrics, dict) else None
        sv = es.get('surprise') if isinstance(es, dict) and es.get('surprise_status') == 'ok' else None
        surprise = float(sv) if isinstance(sv, (int, float)) and not isinstance(sv, bool) and math.isfinite(sv) else np.nan
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            ticker = asset.get('identifier_value')
            if not isinstance(ticker, str) or not ticker:
                continue
            leg = returns.get(ticker)
            if not isinstance(leg, dict) or leg.get('return_status') != 'ok':
                skipped['invalid_return_status'] += 1
                continue
            value = leg.get('car1')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                skipped['invalid_car1'] += 1
                continue
            rows.append(dict(event_id=rec['event_id'], ticker=ticker, quarter=quarter,
                             event_datetime=rec.get('event_datetime'), text=text, car1=float(value), surprise=surprise))
    if not rows:
        raise RuntimeError('没有找到有效财报收益，请检查是否使用了完整档案。')
    out = pd.DataFrame(rows)
    keys = ['event_id', 'ticker']
    duplicates = out[out.duplicated(keys, keep=False)]
    if not duplicates.empty:
        conflict = duplicates.groupby(keys)[['quarter', 'car1', 'text', 'event_datetime', 'surprise']].nunique(dropna=False)
        if (conflict > 1).any().any():
            raise RuntimeError('重复事件记录存在冲突，请检查档案版本。')
        out = out.drop_duplicates(keys).copy()
    out['event_datetime'] = pd.to_datetime(out['event_datetime'], utc=True, errors='coerce')
    if out['event_datetime'].isna().any():
        raise RuntimeError('存在无效事件时间，请先检查。')
    # 官方每季度百分位变换；不是 pandas rank(pct=True)。
    out['y'] = out.groupby('quarter')['car1'].transform(lambda s: percentile_ranks(s.tolist()))
    out['has_text'] = out['text'].str.contains(r'[A-Za-z]{2}', regex=True)
    return out.sort_values(['event_datetime', 'event_id', 'ticker']).reset_index(drop=True), skipped

def make_features(family):
    word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000,
                           sublinear_tf=True, dtype=np.float64)
    if family == 'word':
        return word
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3,
                           max_features=30000, sublinear_tf=True, dtype=np.float64)
    return FeatureUnion([('word', word), ('char', char)],
                         transformer_weights={'word':1.0, 'char':0.5}, n_jobs=1)


def train(research_root):
    if sklearn.__version__ != '1.9.1':
        raise RuntimeError('Use scikit-learn==1.9.1; install requirements-ridge.txt first.')
    now=datetime.now(timezone.utc)
    if QUARTERS[-1] >= f'{now.year}Q{(now.month-1)//3+1}':
        raise RuntimeError('Training quarters must be completed.')
    archive=research_root/'data'/'archive'
    files=sorted(archive.rglob('*.jsonl.gz'))
    if not files: raise RuntimeError('No cached archive in data/archive.')
    output=research_root/'artifacts'/'tfidf_ridge_v2_alpha10'
    if output.exists():
        raise RuntimeError(f'Output already exists: {output}. Preserve it; this script will not overwrite it.')
    frame=load_archive(archive)
    frame=frame[frame['quarter'].isin(QUARTERS)].copy()
    data,skipped=make_outcomes(frame)
    if set(data.quarter)!=set(QUARTERS): raise RuntimeError('One or more training quarters are missing.')
    fit=data[data.has_text].copy()
    if fit.groupby('quarter').size().reindex(QUARTERS,fill_value=0).min()<30:
        raise RuntimeError('Too few training texts in one or more quarters.')
    print('Training quarters:', QUARTERS, flush=True)
    print('Training rows:', len(fit), 'Skipped:',dict(skipped),flush=True)
    model=Pipeline([('features',make_features('word_char')),('ridge',Ridge(alpha=10.0,solver='lsqr'))])
    model.fit(fit.text,fit.y)
    smoke=np.clip(model.predict(SMOKE_TEXTS),0,1)
    if not np.isfinite(smoke).all() or abs(smoke[0]-smoke[1])<1e-8:
        raise RuntimeError('Smoke examples failed; inspect training before export.')
    counts={name:len(tf.vocabulary_) for name,tf in model['features'].transformer_list}
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.v2-export-',dir=output.parent))
    try:
        joblib.dump(model,stage/'model.joblib',compress=3)
        metadata=dict(strategy=STRATEGY,alpha=10.0,sklearn_version=sklearn.__version__,
          sha256=sha256(stage/'model.joblib'),feature_counts=counts,training_quarters=QUARTERS,
          training_rows=len(fit),training_rows_by_quarter=fit.groupby('quarter').size().to_dict(),
          smoke_texts=SMOKE_TEXTS,smoke_predictions=smoke.tolist(),
          exported_at=datetime.now(timezone.utc).isoformat(),
          candidate_frozen='word_char_a10; word weight 1.0, char weight 0.5',
          preprocessing='earnings-call-facts strings stripped and joined by newline; ASCII >=2 letters',
          evaluation='selected using historical development folds; no independent final-refit test',
          archives={str(p.relative_to(research_root)):sha256(p) for p in files})
        (stage/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        restored=joblib.load(stage/'model.joblib')
        if not np.allclose(restored.predict(SMOKE_TEXTS),model.predict(SMOKE_TEXTS),rtol=0,atol=1e-12):
            raise RuntimeError('Export/reload prediction mismatch.')
        stage.rename(output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print('Model exported:',output)
    print('Feature counts:',counts,'Smoke predictions:',smoke.tolist())
    print('Not installed or deployed. No predictions submitted.')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--research-root',type=Path,default=Path.cwd())
    args=parser.parse_args()
    train(args.research_root.expanduser().resolve())
