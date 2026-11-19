# Machine Learning Models, explained on real data — project report

A CS Funda tutorial: fifteen machine learning models, what each one does, how it works, and which kind of data it suits.
Thirteen models are trained live in one notebook on public datasets; the Transformer and LLM results come from the
[ScamGuard](https://github.com/reazulzishan/ScamGuard) project.

## Contents

| Path | What it is |
|---|---|
| `notebooks/make_notebook.py` | Generates the tutorial notebook |
| `notebooks/ml_models.ipynb` | The notebook (clean) |
| `notebooks/ml_models_live.ipynb` | The same notebook, executed: every number in the video comes from these outputs |
| `reports/viz.json` | Results and plot data saved by the notebook |
| `video/realviz_live.py` → `reports/live.json` | Step-by-step values for the animated scenes (one real example per model). It retrains with the same splits and seeds and **stops if any score differs** from the notebook |
| `recording/` | Scripts that run the notebook in JupyterLab and capture the cells shown in the video |
| `video/script.md`, `video/make_vo.py`, `video/film/build_ml.py` | Narration script, voice-over and the video build (HyperFrames) |

## Results (live run, 2026-11-19)

| # | Model | Data | Result |
|---|---|---|---|
| 1 | Linear regression | California Housing (20,640 districts) | R² 0.576, MAE $53,320; gradient descent: error 5.63 → 0.70 in 60 steps |
| 2 | Logistic regression | Breast Cancer Wisconsin (569 tumours, 30 features) | accuracy 98.6%, AUC 0.998 |
| 3 | k-nearest neighbours | Wine (alcohol, flavanoids) | k = 1: 85.2%; best k = 2: 90.7% |
| 4 | Decision tree (depth 3) | Breast Cancer | 94.4%; root split "worst radius ≤ 16.80" |
| 5 | Random forest (300 trees) | Breast Cancer | one deep tree 92.3% → forest 95.8% |
| 6 | Gradient boosting (LightGBM) | California Housing | R² 0.854 (random forest 0.806); error 1.107 → 0.437 |
| 7 | Support vector machine | Wine | 28 support vectors (2-D); RBF kernel, 13 features: 98.1% |
| 8 | Naive Bayes | 20 Newsgroups, 4 topics | 88.1% |
| 9 | K-means | Wine (no labels) | converged in 6 steps, ARI 0.915 |
| 10 | PCA | Digits (64 pixels) | 2 components keep 28.5%; 21 keep 90% |
| 11 | Neural network (MLP) | Digits | 98.2%, 4,810 weights |
| 12 | CNN | MNIST | 97.54% on 10,000 test images, 54,314 weights, 2 epochs |
| 13 | LSTM | Mauna Loa CO₂ (526 months) | MAE 2.95 ppm vs seasonal-naive baseline 0.99 ppm: **the baseline won** |
| 14 | Transformer (XLM-RoBERTa) | ScamGuard, Bangla scam SMS | 26 tokens, 99.9% scam |
| 15 | LLM (Qwen2.5-3B, zero-shot) | ScamGuard | P("Yes") = 0.9998; 0.96 on the locked real-world test |

## Which model for which data

- **Tables:** linear / logistic regression as a baseline, then gradient boosting.
- **Text:** naive Bayes as a baseline, then a Transformer.
- **Images:** a CNN.
- **Sequences:** a simple baseline first, then an LSTM or a Transformer.
- **No labels:** k-means and PCA.

## Honest limits

- Each model runs with reasonable defaults on one dataset; this is a comparison for learning, not a benchmark.
- The LSTM is small and the CO₂ series is short (526 points), which is exactly why the baseline beat it.
- k-NN, SVM (2-D view) and k-means use two wine features so the decision regions can be drawn.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-learn lightgbm torch torchvision statsmodels pandas matplotlib jupyter nbclient
python notebooks/make_notebook.py
python recording/run_nb.py notebooks/ml_models.ipynb notebooks/ml_models_live.ipynb
python video/realviz_live.py
```

On macOS, the notebook imports LightGBM before PyTorch and sets `torch.set_num_threads(1)` to avoid an OpenMP conflict.

## Data and credits

California Housing (StatLib, via scikit-learn) · Breast Cancer Wisconsin (Diagnostic), Wine and Optical Recognition of
Handwritten Digits (UCI Machine Learning Repository, via scikit-learn) · 20 Newsgroups · MNIST (LeCun, Cortes, Burges) ·
Mauna Loa CO₂ (NOAA, via statsmodels). Datasets are downloaded by the libraries and are not stored in this repository.
