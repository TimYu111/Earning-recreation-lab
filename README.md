# Attention Is All You Need

**Earnings-event return prediction with TF-IDF, Ridge regression, and chronological validation.**

The project title is a nod to the Transformer paper; the models here use TF-IDF and Ridge regression, not an attention architecture.

This personal research project studies whether earnings-call facts contain predictive information beyond earnings surprise. It compares word and character TF-IDF features with Ridge regression, evaluates configurations using chronological quarter splits, and provides an inference path for a Modal webhook service.

The current candidate is **word + character TF-IDF with Ridge alpha=10**. Research and inference require no paid language-model API. A prior word-only model was deployed and passed a platform round-trip test; the v2 code is prepared for deployment, but a v2 live deployment is not asserted here.

## Research question

Can a compact text model explain the cross-sectional ranking of one-day abnormal returns beyond the earnings-surprise benchmark?

- Input: only the platform-provided `earnings-call-facts` strings, stripped and joined with newlines.
- Target: the within-quarter percentile rank of `car1`.
- Features: word unigrams/bigrams; optionally character 3–5 grams within word boundaries.
- Model: Ridge regression; predictions clipped to [0,1].
- Missing text: explicit neutral prediction of 0.5.

The target is a return percentile, not the probability that a stock price will rise. This repository does not report a trading backtest, realized profits, or Sharpe ratio.

The [initial baseline notebook](notebooks/02_my_tfidf_ridge_strategy.ipynb) records the first MSE-based experiment. The subsequent walk-forward notebook uses the official score for model selection. Notebook outputs are cleared in the public repository.

## Chronological evaluation

| Training quarters | Development validation quarter |
|---|---|
| 2025Q4 | 2026Q1 |
| 2025Q4–2026Q1 | 2026Q2 |
| 2025Q4–2026Q2 | 2026Q3 |

Vocabulary and IDF weights are fit only on each fold's training text. Model selection compares two feature families and alpha values 1, 10, 100, 300, 1000 and 3000.

The primary metric is the official `delta_r_squared_imputed`: the increase in retrospective OLS R² when adding the model prediction to a regression already containing percentile-ranked earnings surprise. MSE and Spearman correlation are diagnostic measures. The arithmetic mean across quarters below is a model-selection summary, not an official competition-period score.

## Historical development results

| Configuration | Q1 ΔR² | Q2 ΔR² | Q3 ΔR² | Mean ΔR² |
|---|---:|---:|---:|---:|
| Word, alpha=100 (v1 configuration) | 0.01660 | 0.01276 | 0.02340 | 0.01759 |
| Word, alpha=10 | 0.01644 | 0.01164 | 0.03782 | 0.02197 |
| Word + character, alpha=10 | 0.01681 | 0.01251 | 0.04235 | 0.02389 |
| Word + character, alpha=100 | 0.01699 | 0.01403 | 0.02647 | 0.01916 |

Source: [complete comparison](results/summary.csv) and [fold diagnostics](results/fold_results.csv), from the local archive experiment conducted on November 10, 2026. Each v1 comparison is refit using only that fold's training window; it is not the final deployed model evaluated on its own training data.

At alpha=10 and alpha=100, adding character features improves ΔR² in all three folds. The selected alpha=10 candidate has the highest mean, but almost all of its net improvement over the v1 configuration comes from Q3. The alpha=100 character model improves on v1 in every quarter with a smaller average gain.

**These are development results, not an untouched test-set claim.** Q3 was inspected during model development. Archived disclosures and labels may also have been generated or revised after events. These results therefore do not establish point-in-time tradability, statistical significance, or future performance.

## Run the offline tests

Python 3.12 is the reference deployment/test environment. Model artifacts use scikit-learn 1.9.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests build a small synthetic model in a temporary directory. They exercise text extraction, inference equivalence, finite/bounded outputs, checksum validation, version mismatch, TEST bypass, and HTTP failure handling without platform credentials. Synthetic test scores are not included in the research results. The GitHub Actions workflow runs these tests; no hosted CI pass is claimed before the repository is pushed.

## Reproduce the research

Historical data access is managed by the [Explaining Markets platform](https://explainingmarkets.ai/). Raw archives, per-event predictions, API credentials and trained model artifacts are not distributed in this repository.

1. Set up the official [examples repository](https://github.com/explaining-markets/examples) and download archives using its documented workflow and your own authorized account.
2. Install its Python package in the environment used for this project. For the existing local layout:

   ```bash
   python -m pip install -e ../explaining-markets-research
   python -m pip install ipykernel
   mkdir -p data
   ln -s "$HOME/explaining-markets-research/data/archive" data/archive
   ```

3. Open [the walk-forward notebook](notebooks/01_walkforward_research.ipynb), select this environment and run the cells. The notebook contains Chinese working notes and fixes the experiment to the completed 2025Q4–2026Q3 quarters.
4. Run the frozen candidate training script:

   ```bash
   python train.py --research-root "$HOME/explaining-markets-research"
   ```

The script exports into that research project's `artifacts/tfidf_ridge_v2_alpha10/`. It refuses to overwrite an existing export. If that exact candidate has already been trained, reuse the verified export rather than rerunning training. Training on all four quarters produces a deployment artifact, not another independent evaluation.

## Deployment

The webhook and client integration build on the official [Modal starter](https://github.com/explaining-markets/starter-modal). The training package (`examples`) and deployment SDK (`explaining_markets`) are separate upstream dependencies.

```bash
python -m pip install -r requirements-deploy.txt
python -m pip install -e ../starter-modal
mkdir -p models
cp ../explaining-markets-research/artifacts/tfidf_ridge_v2_alpha10/model.joblib models/
cp ../explaining-markets-research/artifacts/tfidf_ridge_v2_alpha10/metadata.json models/
cp .env.example .env
# Fill .env locally, then authenticate to Modal using its setup workflow.
python predict.py
modal run modal_app.py::check_model
# Only after both checks pass:
modal deploy modal_app.py
```

The deployment retains the starter's app/endpoint names. In an existing Modal workspace, deploying can replace that named app: review the destination before running the final command. TEST events check the receive/submit path with neutral predictions; `check_model` separately exercises the trained model. The cloud smoke check compares predictions against values recorded at training export.

## Project contributions and provenance

The project-specific work covers the chronological experimental design, comparison against the official conditional score, feature/regularization experiments, artifact validation, and the word/character inference integration. Webhook verification, client submission, and competition scoring originate from Explaining Markets. See [upstream provenance](THIRD_PARTY.md).

Development used AI-assisted coding and review. The reported tables come from executed local archive experiments; they are not generated estimates. This is an independent personal project, not an official Explaining Markets or Optiver product.

## Next evaluation

Freeze the candidate and its preprocessing, record the actual deployment timestamp, and evaluate on subsequently arriving events not used for selection. Keep the prior model artifact for rollback. Confirm that forecast materials and labels respect the relevant availability times before making stronger backtesting claims.
