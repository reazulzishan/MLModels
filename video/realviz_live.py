"""Step-by-step data for the 'live' visuals: one real example travelling through each model.
Retrains the notebook's models with the same data, splits and seeds, and STOPS if any score differs from the recorded run."""
import json, numpy as np, pandas as pd, warnings
from pathlib import Path
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_wine, load_digits, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, accuracy_score

R = Path(__file__).resolve().parents[1]; V = json.loads((R / "reports/viz.json").read_text()); OUT = {}
def same(name, a, b, tol=6e-4):
    assert abs(a - b) <= tol, f"{name}: {a} != recorded {b}"; print(f"  ✓ {name} {a:.4f}")
np.random.seed(42)

# 1 gradient descent trajectory (exact notebook loop)
H = fetch_california_housing(as_frame=True); X, y = H.data, H.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
x = ((Xtr.MedInc - Xtr.MedInc.mean()) / Xtr.MedInc.std()).values; t = ytr.values
w = b = 0.0; traj = []
for step in range(60):
    pred = w * x + b; err = pred - t; traj.append([round(w, 4), round(b, 4), round(float((err ** 2).mean()), 4)])
    w -= 0.1 * 2 * (err * x).mean(); b -= 0.1 * 2 * err.mean()
same("GD final loss", traj[-1][2], V["linreg"]["loss"][-1], 2e-3)
OUT["gd"] = traj

# 2 logistic regression: one benign + one malignant test tumour
B = load_breast_cancer(); Xb, yb = B.data, B.target
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.25, random_state=42, stratify=yb)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xbtr, ybtr)
same("logistic accuracy", accuracy_score(ybte, logit.predict(Xbte)), V["logreg"]["acc"])
z = logit.decision_function(Xbte); p = logit.predict_proba(Xbte)[:, 1]
i_b = int(np.argmin(np.abs(z - 3))); i_m = int(np.argmin(np.abs(z + 4)))
OUT["logit"] = [{"z": round(float(z[i]), 2), "p": round(float(p[i]), 3), "true": int(ybte[i])} for i in (i_m, i_b)]

# 3 k-NN: one real test wine and its real neighbours (best k)
Wd = load_wine(); f2 = [Wd.feature_names.index("alcohol"), Wd.feature_names.index("flavanoids")]
Xw = StandardScaler().fit_transform(Wd.data[:, f2]); yw = Wd.target
Xwtr, Xwte, ywtr, ywte = train_test_split(Xw, yw, test_size=0.3, random_state=42, stratify=yw)
k = V["knn"]["best_k"]; knn = KNeighborsClassifier(k).fit(Xwtr, ywtr)
same("kNN accuracy", accuracy_score(ywte, knn.predict(Xwte)), V["knn"]["acc_best"])
k5 = KNeighborsClassifier(5).fit(Xwtr, ywtr)
_, all_idx = k5.kneighbors(Xwte)        # a test wine whose 5 neighbours disagree, so the vote is visible
q = next(i for i in range(len(Xwte)) if len(set(ywtr[all_idx[i]])) > 1 and knn.predict(Xwte[i:i + 1])[0] == ywte[i])
dist, idx = k5.kneighbors(Xwte[q:q + 1])
OUT["knn"] = {"q": Xwte[q].round(3).tolist(), "q_true": int(ywte[q]), "nb": Xwtr[idx[0]].round(3).tolist(), "nb_c": ywtr[idx[0]].tolist(),
              "dist": dist[0].round(3).tolist(), "pred_k": int(knn.predict(Xwte[q:q + 1])[0]), "k": k}

# 4 decision tree: one test tumour's path
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xbtr, ybtr)
same("tree accuracy", accuracy_score(ybte, tree.predict(Xbte)), V["tree"]["acc"])
T = tree.tree_; j = 7; node = 0; path = []
while T.children_left[node] != -1:
    f = T.feature[node]; val = float(Xbte[j, f]); go_left = val <= T.threshold[node]
    path.append({"node": int(node), "feature": B.feature_names[f], "value": round(val, 2), "thr": round(float(T.threshold[node]), 2), "left": bool(go_left)})
    node = T.children_left[node] if go_left else T.children_right[node]
OUT["tree_path"] = {"path": path, "leaf": int(node), "pred": int(tree.predict(Xbte[j:j + 1])[0]), "true": int(ybte[j])}

# 5 random forest: the 300 votes for one tumour (pick one the trees disagree on)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xbtr, ybtr)
same("forest accuracy", accuracy_score(ybte, rf.predict(Xbte)), V["rf"]["rf_acc"])
votes = np.array([e.predict(Xbte) for e in rf.estimators_])          # 300 × n_test
benign_share = votes.mean(0); jj = int(np.argmin(np.abs(benign_share - 0.75)))
OUT["rf_votes"] = {"benign": int(votes[:, jj].sum()), "malignant": int(300 - votes[:, jj].sum()), "true": int(ybte[jj]),
                   "order": votes[:, jj].astype(int).tolist()}

# 6 gradient boosting: one house, prediction after more and more trees
gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1).fit(Xtr, ytr)
same("LightGBM R2", r2_score(yte, gbm.predict(Xte)), V["gbm"]["r2"]["LightGBM"])
stages = [1, 5, 10, 25, 50, 100, 200, 400, 600]
p1, p600 = gbm.predict(Xte, num_iteration=1), gbm.predict(Xte)        # the house where adding trees helps most
h = int(np.argmax(np.abs(p1 - yte.values) - np.abs(p600 - yte.values)))
OUT["gbm_house"] = {"true": round(float(yte.iloc[h]), 3), "stages": stages,
                    "pred": [round(float(gbm.predict(Xte.iloc[h:h + 1], num_iteration=s_)[0]), 3) for s_ in stages],
                    "income": round(float(Xte.iloc[h].MedInc), 2), "rooms": round(float(Xte.iloc[h].AveRooms), 2)}

# 8 naive Bayes: one real test post, word by word
cats = ["sci.space", "rec.autos", "sci.med", "comp.graphics"]
kw = dict(categories=cats, remove=("headers", "footers", "quotes"), random_state=42)
tr, te = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
cv = CountVectorizer(stop_words="english", min_df=3); nb = MultinomialNB(alpha=0.1).fit(cv.fit_transform(tr.data), tr.target)
same("naive Bayes accuracy", accuracy_score(te.target, nb.predict(cv.transform(te.data))), V["nb"]["acc"])
names = tr.target_names; voc = np.array(cv.get_feature_names_out()); a_ = names.index("rec.autos")
Xc = cv.transform(te.data); cand = [i for i in range(len(te.data)) if te.target[i] == a_ and 6 <= Xc[i].nnz <= 25 and nb.predict(Xc[i])[0] == a_]
i0 = cand[0]; row = Xc[i0].toarray()[0]; words = np.nonzero(row)[0]
lp = nb.feature_log_prob_
contrib = sorted(((voc[w_], float(lp[a_, w_] - np.max(np.delete(lp[:, w_], a_)))) for w_ in words), key=lambda z_: -z_[1])[:6]
OUT["nb_post"] = {"words": [[w_, round(c_, 2)] for w_, c_ in contrib], "proba": dict(zip(names, nb.predict_proba(Xc[i0])[0].round(3).tolist())),
                  "true": names[te.target[i0]], "n_words": int(len(words))}

# 11 MLP: hidden neurons for the example digit
D = load_digits(); Dtr, Dte, dtr, dte = train_test_split(D.data / 16.0, D.target, test_size=0.25, random_state=42, stratify=D.target)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42).fit(Dtr, dtr)
same("MLP accuracy", accuracy_score(dte, mlp.predict(Dte)), V["mlp"]["acc"])
hid = np.maximum(0, Dte[:1] @ mlp.coefs_[0] + mlp.intercepts_[0])[0]
OUT["mlp_hidden"] = (hid / hid.max()).round(3).tolist()

(R / "reports/live.json").write_text(json.dumps(OUT))
print("saved reports/live.json:", ", ".join(OUT))
print("tree path:", [(p_["feature"], p_["value"], "≤" if p_["left"] else ">", p_["thr"]) for p_ in OUT["tree_path"]["path"]], "→ pred", OUT["tree_path"]["pred"], "true", OUT["tree_path"]["true"])
print("rf votes:", OUT["rf_votes"]["benign"], "benign /", OUT["rf_votes"]["malignant"], "malignant · true", OUT["rf_votes"]["true"])
print("house:", OUT["gbm_house"])
print("post words:", OUT["nb_post"]["words"], OUT["nb_post"]["proba"])
print("knn:", OUT["knn"]["nb_c"], "pred", OUT["knn"]["pred_k"], "true", OUT["knn"]["q_true"])
print("logit:", OUT["logit"])
