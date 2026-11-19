"""Builds ml_models.ipynb: 13 models trained live on real datasets (+ Transformer/LLM results from ScamGuard).
Every number and every visual in the video comes from running this notebook (viz data -> ../reports/viz.json)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Machine Learning Models: how they work and which one to use
Real data, real training, real results. Datasets: California housing, breast cancer (Wisconsin), wine,
handwritten digits, 20 Newsgroups, MNIST, Mauna Loa CO₂ — all loaded from scikit-learn, torchvision or statsmodels.
""")
code("""
import json, time, numpy as np, pandas as pd
import lightgbm as lgb      # load LightGBM before PyTorch: on macOS the reverse order crashes (two OpenMP runtimes)
import torch
torch.set_num_threads(1)    # one CPU thread: avoids a thread deadlock with LightGBM's OpenMP on macOS (the GPU does the heavy work)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, adjusted_rand_score
np.random.seed(42); torch.manual_seed(42)
VIZ = {}                                   # everything the video draws, exported at the end
r = lambda x, n=3: float(np.round(x, n))
print("ready")
""")

# ---------------------------------------------------------------- 1 linear regression
md("## 1 · Linear Regression — predict a number (tabular)")
code("""
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
H = fetch_california_housing(as_frame=True); X, y = H.data, H.target          # target: median house value (×100,000 $)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
lin = LinearRegression().fit(Xtr, ytr); p = lin.predict(Xte)
print(f"{len(X):,} districts, {X.shape[1]} features")
print(f"Linear Regression   R² {r2_score(yte, p):.3f}   MAE ${mean_absolute_error(yte, p) * 100000:,.0f}")
""")
code("""
# How it learns: gradient descent on one feature (median income), step by step
x = ((Xtr.MedInc - Xtr.MedInc.mean()) / Xtr.MedInc.std()).values; t = ytr.values
w = b = 0.0; losses = []
for step in range(60):
    pred = w * x + b; err = pred - t
    losses.append(float((err ** 2).mean()))
    w -= 0.1 * 2 * (err * x).mean(); b -= 0.1 * 2 * err.mean()
print(f"after 60 steps: y = {w:.3f}·x + {b:.3f}   loss {losses[0]:.2f} → {losses[-1]:.3f}")
idx = np.random.RandomState(0).choice(len(x), 400, replace=False)
VIZ["linreg"] = {"x": x[idx].round(3).tolist(), "y": t[idx].round(3).tolist(), "w": r(w), "b": r(b), "loss": [r(l) for l in losses],
                 "r2": r(r2_score(yte, p)), "mae": int(mean_absolute_error(yte, p) * 100000), "n": len(X)}
""")

# ---------------------------------------------------------------- 2 logistic regression
md("## 2 · Logistic Regression — yes or no (tabular)")
code("""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
B = load_breast_cancer(); Xb, yb = B.data, B.target                            # 1 = benign, 0 = malignant
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.25, random_state=42, stratify=yb)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xbtr, ybtr)
pb = logit.predict_proba(Xbte)[:, 1]; z = logit.decision_function(Xbte)
print(f"{len(Xb)} tumours, {Xb.shape[1]} features")
print(f"Logistic Regression accuracy {accuracy_score(ybte, pb > .5):.3f}   ROC-AUC {roc_auc_score(ybte, pb):.3f}")
VIZ["logreg"] = {"z": z.round(2).tolist(), "p": pb.round(3).tolist(), "y": ybte.tolist(), "acc": r(accuracy_score(ybte, pb > .5)), "auc": r(roc_auc_score(ybte, pb)), "n": len(Xb)}
""")

# ---------------------------------------------------------------- 3 kNN
md("## 3 · k-Nearest Neighbours — ask the neighbours (tabular, small data)")
code("""
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
Wd = load_wine(); f2 = [Wd.feature_names.index("alcohol"), Wd.feature_names.index("flavanoids")]
Xw = StandardScaler().fit_transform(Wd.data[:, f2]); yw = Wd.target
Xwtr, Xwte, ywtr, ywte = train_test_split(Xw, yw, test_size=0.3, random_state=42, stratify=yw)
accs = {k: accuracy_score(ywte, KNeighborsClassifier(k).fit(Xwtr, ywtr).predict(Xwte)) for k in range(1, 26)}
best_k = max(accs, key=accs.get); knn = KNeighborsClassifier(best_k).fit(Xwtr, ywtr)
print(f"{len(yw)} wines, 3 grape cultivars, 2 features: alcohol, flavanoids")
print(f"k=1 accuracy {accs[1]:.3f}   best k={best_k} accuracy {accs[best_k]:.3f}")
g = np.linspace(-3, 3, 90); GX, GY = np.meshgrid(g, g)
grid = knn.predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
VIZ["knn"] = {"grid": grid.tolist(), "lim": 3, "x": Xw[:, 0].round(3).tolist(), "y": Xw[:, 1].round(3).tolist(), "c": yw.tolist(),
              "best_k": int(best_k), "acc_k1": r(accs[1]), "acc_best": r(accs[best_k]), "accs": [r(accs[k]) for k in range(1, 26)]}
""")

# ---------------------------------------------------------------- 4 decision tree
md("## 4 · Decision Tree — a chain of yes/no questions (tabular)")
code("""
from sklearn.tree import DecisionTreeClassifier, export_text
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xbtr, ybtr)
print(export_text(tree, feature_names=list(B.feature_names), decimals=2))
print(f"Decision tree (depth 3) accuracy {accuracy_score(ybte, tree.predict(Xbte)):.3f}")
T = tree.tree_
VIZ["tree"] = {"feature": [B.feature_names[i] if i >= 0 else None for i in T.feature], "threshold": T.threshold.round(2).tolist(),
               "left": T.children_left.tolist(), "right": T.children_right.tolist(), "samples": T.n_node_samples.tolist(),
               "value": [[int(round(v)) for v in row] for row in (T.value[:, 0, :] * T.n_node_samples[:, None])],
               "acc": r(accuracy_score(ybte, tree.predict(Xbte)))}
""")

# ---------------------------------------------------------------- 5 random forest
md("## 5 · Random Forest — many trees vote (tabular)")
code("""
from sklearn.ensemble import RandomForestClassifier
deep = DecisionTreeClassifier(random_state=42).fit(Xbtr, ybtr)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xbtr, ybtr)
imp = pd.Series(rf.feature_importances_, index=B.feature_names).nlargest(5)
print(f"one deep tree accuracy {accuracy_score(ybte, deep.predict(Xbte)):.3f}   random forest (300 trees) {accuracy_score(ybte, rf.predict(Xbte)):.3f}")
print(imp.round(3))
VIZ["rf"] = {"tree_acc": r(accuracy_score(ybte, deep.predict(Xbte))), "rf_acc": r(accuracy_score(ybte, rf.predict(Xbte))),
             "imp": [[k, r(v)] for k, v in imp.items()], "n_trees": 300}
""")

# ---------------------------------------------------------------- 6 gradient boosting
md("## 6 · Gradient Boosting (LightGBM) — each tree fixes the last one's mistakes (tabular)")
code("""
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
gbm.fit(Xtr, ytr, eval_X=Xte, eval_y=yte, eval_metric="l2")
curve = np.sqrt(gbm.evals_result_["valid_0"]["l2"])
rfr = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
scores = {"Linear Regression": r2_score(yte, lin.predict(Xte)), "Random Forest": r2_score(yte, rfr.predict(Xte)), "LightGBM": r2_score(yte, gbm.predict(Xte))}
for k, v in scores.items(): print(f"{k:18s} R² {v:.3f}")
print(f"LightGBM error after 1 tree {curve[0]:.3f} → after 600 trees {curve[-1]:.3f}")
VIZ["gbm"] = {"r2": {k: r(v) for k, v in scores.items()}, "curve": [r(c) for c in curve[::5]], "n": 600}
""")

# ---------------------------------------------------------------- 7 SVM
md("## 7 · Support Vector Machine — the widest street between classes")
code("""
from sklearn.svm import SVC
m01 = yw < 2; X2, y2 = Xw[m01], yw[m01]                                          # two cultivars, two features
svm = SVC(kernel="linear", C=1.0).fit(X2, y2)
w_, b_ = svm.coef_[0], svm.intercept_[0]
print(f"linear SVM: {len(svm.support_)} support vectors decide the boundary; margin width {2 / np.linalg.norm(w_):.3f}")
Wtr, Wte, wtr, wte = train_test_split(StandardScaler().fit_transform(Wd.data), yw, test_size=0.3, random_state=42, stratify=yw)
print(f"RBF SVM on all 13 wine features: accuracy {accuracy_score(wte, SVC(kernel='rbf').fit(Wtr, wtr).predict(Wte)):.3f}")
VIZ["svm"] = {"x": X2[:, 0].round(3).tolist(), "y": X2[:, 1].round(3).tolist(), "c": y2.tolist(), "w": w_.round(4).tolist(), "b": r(b_, 4),
              "sv": svm.support_.tolist(), "n_sv": int(len(svm.support_)), "rbf_acc": r(accuracy_score(wte, SVC(kernel='rbf').fit(Wtr, wtr).predict(Wte)))}
""")

# ---------------------------------------------------------------- 8 naive bayes
md("## 8 · Naive Bayes — word probabilities (text)")
code("""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
cats = ["sci.space", "rec.autos", "sci.med", "comp.graphics"]
kw = dict(categories=cats, remove=("headers", "footers", "quotes"), random_state=42)
tr, te = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
cv = CountVectorizer(stop_words="english", min_df=3); Ttr = cv.fit_transform(tr.data)
nb = MultinomialNB(alpha=0.1).fit(Ttr, tr.target)
acc_nb = accuracy_score(te.target, nb.predict(cv.transform(te.data)))
print(f"{len(tr.data):,} training posts, {len(te.data):,} test posts, {Ttr.shape[1]:,} words")
print(f"Naive Bayes accuracy {acc_nb:.3f}")
voc = np.array(cv.get_feature_names_out()); lp = nb.feature_log_prob_
top = {tr.target_names[i]: voc[np.argsort(lp[i] - lp.mean(0))[-6:][::-1]].tolist() for i in range(len(cats))}
for k, v in top.items(): print(f"  {k:14s} {', '.join(v)}")
VIZ["nb"] = {"acc": r(acc_nb), "top": top, "n_train": len(tr.data), "n_test": len(te.data), "n_words": int(Ttr.shape[1])}
""")

# ---------------------------------------------------------------- 9 k-means
md("## 9 · K-Means — find groups without labels")
code("""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
Xall = StandardScaler().fit_transform(Wd.data); P2 = PCA(2, random_state=42).fit(Xall); Z = P2.transform(Xall)
km0 = KMeans(3, init="random", n_init=1, max_iter=1, random_state=7).fit(Xall); C0 = km0.cluster_centers_
steps = []; Cc = C0
for it in range(8):
    km = KMeans(3, init=Cc, n_init=1, max_iter=1).fit(Xall)
    steps.append({"centers": P2.transform(km.cluster_centers_).round(3).tolist(), "labels": km.labels_.tolist(), "inertia": r(km.inertia_, 1)})
    if np.allclose(km.cluster_centers_, Cc): break
    Cc = km.cluster_centers_
ari = adjusted_rand_score(yw, steps[-1]["labels"])
print(f"converged after {len(steps)} steps; inertia {steps[0]['inertia']} → {steps[-1]['inertia']}")
print(f"agreement with the real cultivars it never saw (adjusted Rand index): {ari:.3f}")
VIZ["kmeans"] = {"x": Z[:, 0].round(3).tolist(), "y": Z[:, 1].round(3).tolist(), "steps": steps, "ari": r(ari), "true": yw.tolist()}
""")

# ---------------------------------------------------------------- 10 PCA
md("## 10 · PCA — squeeze many features into a few")
code("""
from sklearn.datasets import load_digits
D = load_digits(); Pd = PCA(random_state=42).fit(D.data)
ev = Pd.explained_variance_ratio_; cum = np.cumsum(ev)
n90 = int(np.argmax(cum >= 0.90) + 1)
print(f"{len(D.data):,} images of 8×8 pixels = 64 features")
print(f"first 2 components keep {cum[1]:.1%} of the variance; {n90} components keep 90%")
Z2 = Pd.transform(D.data)[:, :2]; ix = np.random.RandomState(1).choice(len(D.data), 1200, replace=False)
VIZ["pca"] = {"ev": ev[:20].round(4).tolist(), "cum2": r(cum[1]), "n90": n90, "x": Z2[ix, 0].round(2).tolist(), "y": Z2[ix, 1].round(2).tolist(),
              "c": D.target[ix].tolist()}
""")

# ---------------------------------------------------------------- 11 MLP
md("## 11 · Neural Network (MLP) — layers of weighted sums")
code("""
from sklearn.neural_network import MLPClassifier
Dtr, Dte, dtr, dte = train_test_split(D.data / 16.0, D.target, test_size=0.25, random_state=42, stratify=D.target)
import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():                      # training stops at 300 epochs on purpose, so silence that one notice
    warnings.simplefilter("ignore", ConvergenceWarning)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42).fit(Dtr, dtr)
print(f"64 inputs → 64 hidden neurons → 10 outputs   ({sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_):,} weights)")
print(f"MLP accuracy {accuracy_score(dte, mlp.predict(Dte)):.3f}   loss {mlp.loss_curve_[0]:.2f} → {mlp.loss_curve_[-1]:.3f} in {mlp.n_iter_} epochs")
VIZ["mlp"] = {"acc": r(accuracy_score(dte, mlp.predict(Dte))), "loss": [r(l) for l in mlp.loss_curve_], "weights": int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
              "digit": (Dte[0] * 16).astype(int).tolist(), "probs": mlp.predict_proba(Dte[:1])[0].round(3).tolist(), "true": int(dte[0])}
""")

# ---------------------------------------------------------------- 12 CNN
md("## 12 · Convolutional Neural Network — filters that scan images")
code("""
import torch.nn as nn, torchvision
from torchvision import transforms
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tf = transforms.ToTensor()
mtr = torchvision.datasets.MNIST("../data", train=True, download=True, transform=tf)
mte = torchvision.datasets.MNIST("../data", train=False, download=True, transform=tf)
cnn = nn.Sequential(nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)).to(dev)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
Xg = mtr.data.float().div(255).unsqueeze(1).to(dev); Yg = mtr.targets.to(dev); hist = []   # whole training set on the GPU
for ep in range(2):
    perm = torch.randperm(len(Yg), device=dev)
    for i in range(0, len(Yg), 128):
        b_ = perm[i:i + 128]; loss = lossf(cnn(Xg[b_]), Yg[b_]); opt.zero_grad(); loss.backward(); opt.step()
        if (i // 128) % 47 == 0: hist.append(r(loss.item()))
cnn.eval(); Xt = mte.data.float().div(255).unsqueeze(1)
with torch.no_grad(): pred = torch.cat([cnn(Xt[i:i + 2000].to(dev)).argmax(1).cpu() for i in range(0, 10000, 2000)])
acc_cnn = (pred == mte.targets).float().mean().item()
print(f"{len(mtr):,} training images (28×28), {sum(p.numel() for p in cnn.parameters()):,} weights, 2 epochs")
print(f"CNN test accuracy {acc_cnn:.4f} on {len(mte):,} images")
with torch.no_grad(): fmap = cnn[1](cnn[0](Xt[:1].to(dev)))[0].cpu().numpy()
VIZ["cnn"] = {"acc": r(acc_cnn, 4), "loss": hist, "filters": cnn[0].weight.detach().cpu().numpy()[:, 0].round(3).tolist(),
              "image": mte.data[0].tolist(), "label": int(mte.targets[0]), "fmap": fmap[:4, ::2, ::2].round(2).tolist(),
              "weights": int(sum(p.numel() for p in cnn.parameters()))}
""")

# ---------------------------------------------------------------- 13 LSTM
md("## 13 · LSTM — memory for sequences (time series)")
code("""
import statsmodels.api as sm
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
v = co2.values.astype("float32"); mu, sd = v[:-36].mean(), v[:-36].std(); s = (v - mu) / sd
W = 24; Xs = np.stack([s[i:i + W] for i in range(len(s) - W)]); Ys = s[W:]
ntr = len(Ys) - 36
class Net(nn.Module):
    def __init__(self): super().__init__(); self.l = nn.LSTM(1, 32, batch_first=True); self.o = nn.Linear(32, 1)
    def forward(self, x): h, _ = self.l(x); return self.o(h[:, -1])
net = Net(); op = torch.optim.Adam(net.parameters(), lr=0.01)
xt = torch.tensor(Xs[:ntr, :, None]); yt = torch.tensor(Ys[:ntr, None])
for ep in range(300):
    loss = ((net(xt) - yt) ** 2).mean(); op.zero_grad(); loss.backward(); op.step()
hist_ = s[:ntr + W].tolist(); preds = []                     # forecast 36 months, feeding its own predictions back
with torch.no_grad():
    for _ in range(36):
        nxt = net(torch.tensor(np.array(hist_[-W:], dtype="float32")[None, :, None])).item(); preds.append(nxt); hist_.append(nxt)
fc = np.array(preds) * sd + mu; actual = v[-36:]; naive = v[-48:-36] .tolist() * 3
naive = np.array(naive) + (v[-37] - v[-49])                 # seasonal naive: last year's pattern + last year's rise
print(f"{len(v)} months of CO₂ at Mauna Loa ({co2.index[0]:%Y}–{co2.index[-1]:%Y}); forecast the last 36 months")
print(f"LSTM MAE {np.abs(fc - actual).mean():.2f} ppm   seasonal-naive MAE {np.abs(naive - actual).mean():.2f} ppm")
VIZ["lstm"] = {"dates": [f"{d:%Y-%m}" for d in co2.index[-120:]], "actual": v[-120:].round(2).tolist(), "fc": fc.round(2).tolist(),
               "naive": naive.round(2).tolist(), "mae": r(np.abs(fc - actual).mean(), 2), "mae_naive": r(np.abs(naive - actual).mean(), 2), "n": len(v)}
""")

md("## 14–15 · Transformers and LLMs (text) — results from the ScamGuard project")
code("""
sg = json.load(open("../reports/scamguard_results.json"))       # real outputs from the ScamGuard notebook (Bangla scam SMS)
for row in sg["locked"]: print(f"{row['model']:28s} PR-AUC {row['PR-AUC']:.3f}")
VIZ["sg"] = sg
""")

md("## Save the visual data")
code("""
json.dump(VIZ, open("../reports/viz.json", "w"))
print("saved", ", ".join(VIZ))
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "ml_models.ipynb"); print(len(C), "cells")
