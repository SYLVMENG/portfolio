# %% [markdown]
# # Leakage-safe model tuning and PCA trade-offs

# %%
import warnings
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris

pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)

# %% [markdown]
# ## Question 1 – Leakage in `search.fit(X_dev, y_dev)`
# $y = 2\sin x - 0.5\cos x + 0.8x + \varepsilon$, $\varepsilon\sim N(0,1)$, so the noise floor is MSE = 1.

# %%
x_grid = np.linspace(-3, 3, 500)
y_true = 2*np.sin(x_grid) - 0.5*np.cos(x_grid) + 0.8*x_grid

rng = np.random.default_rng(42)
n = 1000
X = rng.uniform(-3, 3, size=n).reshape(-1, 1)
f_true = 2*np.sin(X[:, 0]) - 0.5*np.cos(X[:, 0]) + 0.8*X[:, 0]
y = f_true + rng.normal(0, 1, size=n)

# 60 / 20 / 20 split exactly as in the brief
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.40, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_dev, y_dev, test_size=0.50, random_state=42)
print("train", X_train.shape, "val", X_val.shape, "test", X_test.shape)

# %% [markdown]
# Pipeline, grid and folds as in Session 7 (also used in Q3).

# %%
degrees = list(range(1, 16))
alphas = [0.001, 0.01, 0.1, 1, 10]

def ridge_pipeline():
    return make_pipeline(
        PolynomialFeatures(include_bias=False),
        StandardScaler(),
        Ridge()
    )

param_grid = {"polynomialfeatures__degree": degrees, "ridge__alpha": alphas}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def run_search(Xd, yd, cv_obj=cv):
    s = GridSearchCV(estimator=ridge_pipeline(), param_grid=param_grid,
                     scoring="neg_mean_squared_error", cv=cv_obj, refit=True, n_jobs=-1)
    s.fit(Xd, yd)
    return s

# %% [markdown]
# ### (a) Original code (invalid)
# `X_dev` = validation + test rows, so test rows enter the CV folds and the refit.

# %%
search_bad = run_search(X_dev, y_dev)
bad_test_mse = mean_squared_error(y_test, search_bad.predict(X_test))
print("Invalid version")
print("  best params      :", search_bad.best_params_)
print("  CV MSE (on X_dev):", round(-search_bad.best_score_, 4))
print("  'Test' MSE       :", round(bad_test_mse, 4), " <- test rows were used in tuning and in the refit")
print("  overlap test/X_dev rows:", np.isin(X_test[:, 0], X_dev[:, 0]).sum(), "of", len(X_test))

# %% [markdown]
# ### (b) Corrected: development set = train + validation (800 rows); test used once

# %%
X_dev = np.vstack([X_train, X_val])          # 800 rows: development set used in Question 3
y_dev = np.concatenate([y_train, y_val])
assert not np.isin(X_test[:, 0], X_dev[:, 0]).any()   # no test row in the development set
print("new X_dev", X_dev.shape, "X_test", X_test.shape)

search = run_search(X_dev, y_dev)
test_mse = mean_squared_error(y_test, search.predict(X_test))
f_test = 2*np.sin(X_test[:, 0]) - 0.5*np.cos(X_test[:, 0]) + 0.8*X_test[:, 0]
print("Valid version")
print("  best params        :", search.best_params_)
print("  CV MSE (dev, 5-fold):", round(-search.best_score_, 4))
print("  Test MSE (untouched):", round(test_mse, 4))
print("  Test MSE of the true f (noise floor on these 200 rows):",
      round(mean_squared_error(y_test, f_test), 4))

# %%
q1_table = pd.DataFrame({
    "Version": ["Invalid: fit on X_dev = val+test (400)", "Valid: fit on train+val (800)"],
    "Degree": [search_bad.best_params_["polynomialfeatures__degree"], search.best_params_["polynomialfeatures__degree"]],
    "Alpha": [search_bad.best_params_["ridge__alpha"], search.best_params_["ridge__alpha"]],
    "CV MSE": [-search_bad.best_score_, -search.best_score_],
    "Test MSE": [bad_test_mse, test_mse],
    "Test rows used in fitting?": ["Yes (all 200)", "No"],
})
q1_table

# %% [markdown]
# ### (c) Size of the bias: 50 repetitions
# Each fitted model is also scored on a fresh 20,000-row sample (independent-sample MSE estimate).

# %%
def one_rep(seed, n_big=20_000):
    r = np.random.default_rng(seed)
    Xs = r.uniform(-3, 3, size=1000).reshape(-1, 1)
    ys = 2*np.sin(Xs[:, 0]) - 0.5*np.cos(Xs[:, 0]) + 0.8*Xs[:, 0] + r.normal(0, 1, 1000)
    Xtr, Xd, ytr, yd = train_test_split(Xs, ys, test_size=0.40, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xd, yd, test_size=0.50, random_state=seed)
    Xbig = r.uniform(-3, 3, size=n_big).reshape(-1, 1)
    ybig = 2*np.sin(Xbig[:, 0]) - 0.5*np.cos(Xbig[:, 0]) + 0.8*Xbig[:, 0] + r.normal(0, 1, n_big)
    cvs = KFold(5, shuffle=True, random_state=seed)
    out = {}
    for name, (Xa, ya) in {"invalid": (Xd, yd),
                           "valid": (np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))}.items():
        s = run_search(Xa, ya, cvs)
        out[f"{name}_test"] = mean_squared_error(yte, s.predict(Xte))
        out[f"{name}_true"] = mean_squared_error(ybig, s.predict(Xbig))
    return out

rep = pd.DataFrame([one_rep(s) for s in range(50)])
rep["invalid_gap"] = rep["invalid_test"] - rep["invalid_true"]
rep["valid_gap"] = rep["valid_test"] - rep["valid_true"]
summary = pd.DataFrame({
    "Reported test MSE (mean)": [rep.invalid_test.mean(), rep.valid_test.mean()],
    "Independent-sample MSE estimate (mean)": [rep.invalid_true.mean(), rep.valid_true.mean()],
    "Reported − independent (mean)": [rep.invalid_gap.mean(), rep.valid_gap.mean()],
    "SE of difference": [rep.invalid_gap.std(ddof=1)/np.sqrt(len(rep)),
                         rep.valid_gap.std(ddof=1)/np.sqrt(len(rep))],
    "Share of reps reported < independent": [(rep.invalid_gap < 0).mean(), (rep.valid_gap < 0).mean()],
}, index=["Invalid (fit on val+test)", "Valid (fit on train+val)"])
summary

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X_dev[:, 0], y_dev, s=6, alpha=0.3, label="development rows (800)")
ax[0].scatter(X_test[:, 0], y_test, s=8, alpha=0.6, color="tab:red", label="test rows (200)")
ax[0].plot(x_grid, y_true, color="black", lw=2, label="true f(x)")
ax[0].plot(x_grid, search.predict(x_grid.reshape(-1, 1)), color="tab:orange", lw=2,
           label=f"tuned Ridge (deg {search.best_params_['polynomialfeatures__degree']}, "
                 f"α={search.best_params_['ridge__alpha']})")
ax[0].set(xlabel="x", ylabel="y", title="Q1: corrected fit"); ax[0].legend(fontsize=8)
ax[1].boxplot([rep.invalid_gap, rep.valid_gap], tick_labels=["Invalid", "Valid"])
ax[1].axhline(0, color="grey", lw=1)
ax[1].set(ylabel="reported test MSE − independent-sample MSE", title="Q1: bias of reported test MSE (50 seeds)")
plt.tight_layout(); plt.savefig("fig_q1.png", dpi=150); plt.show()

# %% [markdown]
# ## Question 2 – Iris: logistic regression on 4 features vs 1–4 PCs
# Same 5 stratified folds for all models; PCA (on centred features) fitted inside each fold; metric = weighted F1.

# %%
iris = load_iris()
Xi, yi = iris.data, iris.target
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def logit():
    return LogisticRegression(random_state=0, max_iter=1000)   # as in Session 8

models = {"Original 4 features": Pipeline([("logisticregression", logit())])}
for k in range(1, 5):
    models[f"PCA, {k} component{'s' if k > 1 else ''}"] = Pipeline([("pca", PCA(n_components=k)),
                                                                     ("logisticregression", logit())])
rows = []
for name, m in models.items():
    res = cross_validate(m, Xi, yi, cv=skf, scoring="f1_weighted")
    s = res["test_score"]
    rows.append({"Model": name, "Mean weighted F1": s.mean(), "SD across folds": s.std(ddof=1),
                 **{f"Fold {i+1}": v for i, v in enumerate(s)}})
q2_table = pd.DataFrame(rows).set_index("Model")
q2_table

# %%
# Same folds, out-of-fold predictions: how many of the 150 flowers each model misclassifies
from sklearn.model_selection import cross_val_predict
errs = {name: int((cross_val_predict(m, Xi, yi, cv=skf) != yi).sum()) for name, m in models.items()}
pd.Series(errs, name="Misclassified (of 150)")

# %%
# PCA explained variance (fitted on all 150 rows, for description only; not used in the CV)
pca_all = PCA().fit(Xi)
pd.DataFrame({"Explained variance ratio": pca_all.explained_variance_ratio_,
              "Cumulative": pca_all.explained_variance_ratio_.cumsum()},
             index=[f"PC{i}" for i in range(1, 5)])

# %%
# Robustness check: standardise before PCA (scaler also inside each fold)
rows_s = []
for k in range(1, 5):
    m = Pipeline([("standardscaler", StandardScaler()), ("pca", PCA(n_components=k)), ("logisticregression", logit())])
    rows_s.append((k, cross_validate(m, Xi, yi, cv=skf, scoring="f1_weighted")["test_score"].mean()))
m = Pipeline([("standardscaler", StandardScaler()), ("logisticregression", logit())])
rows_s.insert(0, ("Original (standardised)",
                  cross_validate(m, Xi, yi, cv=skf, scoring="f1_weighted")["test_score"].mean()))
pd.DataFrame(rows_s, columns=["Components", "Mean weighted F1 (standardised)"])

# %% [markdown]
# ### SVD check on fold 1: $X_c = USV^\top$, directions = rows of $V^\top$, scores = $US$ (up to sign)

# %%
train_idx, test_idx = next(skf.split(Xi, yi))
X1 = Xi[train_idx]
Xc = X1 - X1.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

pca1 = PCA(n_components=2).fit(X1)
dirs_svd = Vt[:2]
scores_svd = Xc @ dirs_svd.T                       # = U[:, :2] * S[:2]
signs = np.sign(np.sum(dirs_svd * pca1.components_, axis=1))   # +1 or −1 per component
scores_sk = pca1.transform(X1)

print("Fold-1 training rows:", len(train_idx))
print("sklearn components_:\n", np.round(pca1.components_, 4))
print("NumPy SVD Vt[:2]:\n", np.round(dirs_svd, 4))
print("sign flips applied:", signs)
print("max |direction diff| after sign alignment:",
      np.abs(dirs_svd * signs[:, None] - pca1.components_).max())
print("max |score diff| after sign alignment     :",
      np.abs(scores_svd * signs - scores_sk).max())
print("scores via U*S equal Xc@V:", np.allclose(U[:, :2] * S[:2], scores_svd))
print("explained variance S^2/(n-1):", np.round(S[:2]**2/(len(X1)-1), 4),
      " sklearn:", np.round(pca1.explained_variance_, 4))

# %%
svd_table = pd.DataFrame(
    np.vstack([pca1.components_, dirs_svd * signs[:, None]]),
    columns=iris.feature_names,
    index=["PC1 (sklearn)", "PC2 (sklearn)", "PC1 (SVD, sign-aligned)", "PC2 (SVD, sign-aligned)"])
svd_table

# %%
pd.DataFrame(np.hstack([scores_sk[:5], (scores_svd * signs)[:5]]),
             columns=["PC1 sklearn", "PC2 sklearn", "PC1 SVD", "PC2 SVD"]).rename_axis("first 5 rows")

# %% [markdown]
# ## Question 3 – Polynomial Ridge vs Poly(15) → scale → PCA → Ridge
# Development set from Q1; both searches on the same `KFold(5, shuffle=True, random_state=42)`; scaling and PCA inside the pipelines.

# %%
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

base_pipe = make_pipeline(PolynomialFeatures(include_bias=False),
                          StandardScaler(),
                          Ridge())
base_search = GridSearchCV(base_pipe, {"polynomialfeatures__degree": degrees, "ridge__alpha": alphas},
                           scoring="neg_mean_squared_error", cv=cv5, refit=True, n_jobs=-1)
base_search.fit(X_dev, y_dev)

pca_pipe = make_pipeline(PolynomialFeatures(degree=15, include_bias=False),
                         StandardScaler(),
                         PCA(),
                         Ridge())
pca_search = GridSearchCV(pca_pipe, {"pca__n_components": list(range(1, 16)), "ridge__alpha": alphas},
                          scoring="neg_mean_squared_error", cv=cv5, refit=True, n_jobs=-1)
pca_search.fit(X_dev, y_dev)

def cv_frame(s):
    r = pd.DataFrame(s.cv_results_)
    r["cv_mse"] = -r["mean_test_score"]
    r["cv_mse_se"] = r["std_test_score"] / np.sqrt(5)
    return r

base_res, pca_res = cv_frame(base_search), cv_frame(pca_search)

def best_row(s, r, label):
    i = s.best_index_
    return {"Model": label,
            "Selected settings": ", ".join(f"{k.split('__')[1]}={v}" for k, v in s.best_params_.items()),
            "CV MSE": r.loc[i, "cv_mse"], "CV MSE SE": r.loc[i, "cv_mse_se"],
            "Test MSE (reference only)": mean_squared_error(y_test, s.predict(X_test))}

q3_table = pd.DataFrame([best_row(base_search, base_res, "Baseline: poly → scale → Ridge"),
                         best_row(pca_search, pca_res, "Poly(15) → scale → PCA → Ridge")]).set_index("Model")
q3_table

# %% [markdown]
# Test MSE is for reference only; selection uses CV MSE.

# %%
# Best CV MSE per degree (baseline) and per number of components (PCA model), minimised over alpha
base_by_deg = base_res.groupby("param_polynomialfeatures__degree")["cv_mse"].min()
pca_by_k = pca_res.groupby("param_pca__n_components")["cv_mse"].min()
alpha_by_deg = base_res.loc[base_res.groupby("param_polynomialfeatures__degree")["cv_mse"].idxmin()] \
    .set_index("param_polynomialfeatures__degree")["param_ridge__alpha"]
alpha_by_k = pca_res.loc[pca_res.groupby("param_pca__n_components")["cv_mse"].idxmin()] \
    .set_index("param_pca__n_components")["param_ridge__alpha"]
profile = pd.DataFrame({"Baseline: best CV MSE at degree d": base_by_deg.values,
                        "best alpha (baseline)": alpha_by_deg.values,
                        "PCA: best CV MSE with k components": pca_by_k.values,
                        "best alpha (PCA)": alpha_by_k.values},
                       index=pd.Index(range(1, 16), name="d or k"))
profile

# %%
# Full alpha x setting grids
base_grid = base_res.pivot(index="param_polynomialfeatures__degree", columns="param_ridge__alpha", values="cv_mse")
pca_grid = pca_res.pivot(index="param_pca__n_components", columns="param_ridge__alpha", values="cv_mse")
print("Baseline CV MSE (rows = degree, cols = alpha)"); display(base_grid)
print("PCA model CV MSE (rows = components, cols = alpha)"); display(pca_grid)

# %%
# One-standard-error check: simplest setting within 1 SE of the best (for interpretation only)
def one_se(r, setting_col):
    best = r.loc[r["cv_mse"].idxmin()]
    ok = r[r["cv_mse"] <= best["cv_mse"] + best["cv_mse_se"]]
    simplest = ok.sort_values([setting_col, "param_ridge__alpha"], ascending=[True, False]).iloc[0]
    return int(simplest[setting_col]), simplest["param_ridge__alpha"], simplest["cv_mse"]

print("Baseline 1-SE simplest (degree, alpha, CV MSE):", one_se(base_res, "param_polynomialfeatures__degree"))
print("PCA      1-SE simplest (k, alpha, CV MSE)     :", one_se(pca_res, "param_pca__n_components"))

# %%
# Paired fold-by-fold comparison of the two selected models (same folds)
def fold_mse(r, i):
    return np.array([-r.loc[i, f"split{j}_test_score"] for j in range(5)])
fb, fp = fold_mse(base_res, base_search.best_index_), fold_mse(pca_res, pca_search.best_index_)
diff = fp - fb
paired = pd.DataFrame({"Baseline": fb, "PCA model": fp, "PCA − baseline": diff},
                      index=[f"Fold {j+1}" for j in range(5)]).T.assign(Mean=lambda d: d.mean(axis=1))
display(paired)
print(f"Mean paired difference (PCA − baseline): {diff.mean():.4f}")
print(f"SD of the 5 paired differences: {diff.std(ddof=1):.4f};  SE of the mean: {diff.std(ddof=1)/np.sqrt(5):.4f}")
print(f"Folds where the baseline has the lower MSE: {(diff > 0).sum()} of 5")
print("Note: the five fold estimates share training rows, so this SE is only a rough guide.")

# %%
# How much of the variance of the 15 scaled polynomial features each component carries
# (fitted on the whole development set, descriptive only)
pp = Pipeline([("polynomialfeatures", PolynomialFeatures(degree=15, include_bias=False)),
               ("standardscaler", StandardScaler()), ("pca", PCA())]).fit(X_dev)
evr = pp.named_steps["pca"].explained_variance_ratio_
print("Cumulative explained variance, first 8 components:", np.round(evr.cumsum()[:8], 4))
print("Smallest eigenvalue / largest:", pp.named_steps["pca"].explained_variance_[-1] /
      pp.named_steps["pca"].explained_variance_[0])
Z = pp.transform(X_dev)
pd.DataFrame({"Explained var. ratio": evr[:8],
              "corr with x": [np.corrcoef(Z[:, k], X_dev[:, 0])[0, 1] for k in range(8)],
              "corr with x³": [np.corrcoef(Z[:, k], X_dev[:, 0]**3)[0, 1] for k in range(8)],
              "corr with x²": [np.corrcoef(Z[:, k], X_dev[:, 0]**2)[0, 1] for k in range(8)],
              "corr with y": [np.corrcoef(Z[:, k], y_dev)[0, 1] for k in range(8)]},
             index=[f"PC{k+1}" for k in range(8)]).assign(
    **{"in-sample R² of y on first k PCs (= cum. sum of corr²)": lambda d: (d["corr with y"]**2).cumsum()})

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# Full range (no clipping), with an inset that zooms in on the flat region near the noise floor
for a_ in [ax[0]]:
    a_.plot(profile.index, profile.iloc[:, 0], marker="o", label="Baseline: degree d")
    a_.plot(profile.index, profile.iloc[:, 2], marker="s", label="PCA: k components (degree 15)")
    a_.axhline(1.0, color="grey", ls="--", lw=1, label="noise variance σ²=1")
ax[0].set(xlabel="degree d / number of components k", ylabel="best CV MSE over alpha",
          title="Q3: CV MSE profile (full range)", ylim=(0, 7)); ax[0].legend(fontsize=8, loc="upper center")
ins = ax[0].inset_axes([0.45, 0.25, 0.52, 0.42])
ins.plot(profile.index, profile.iloc[:, 0], marker="o", ms=3)
ins.plot(profile.index, profile.iloc[:, 2], marker="s", ms=3)
ins.axhline(1.0, color="grey", ls="--", lw=1)
ins.set(xlim=(2.5, 15.5), ylim=(1.0, 1.15), title="zoom: CV MSE 1.00–1.15", xticks=[3, 5, 7, 9, 11, 13, 15])
ins.title.set_fontsize(8); ins.tick_params(labelsize=7)
ax[1].scatter(X_dev[:, 0], y_dev, s=5, alpha=0.25, color="grey")
ax[1].plot(x_grid, y_true, color="black", lw=2, label="true f(x)")
ax[1].plot(x_grid, base_search.predict(x_grid.reshape(-1, 1)), lw=2, label="baseline")
ax[1].plot(x_grid, pca_search.predict(x_grid.reshape(-1, 1)), lw=2, ls="--", label="PCA model")
ax[1].set(xlabel="x", ylabel="y", title="Q3: selected fits"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_q3.png", dpi=150); plt.show()
