# %% [markdown]
# # Credit card customer segmentation with clustering
#
# **Data:** Customer Segmentation Dataset (Kaggle, `hashemi221022/customer-segmentation-data`), file `Customer_Data.csv`:
# 8,950 credit-card holders, 17 behavioural variables plus a customer ID.
#
# **Structure**
# 1. Data preprocessing (cleaning, transformation, scaling, encoding, PCA)
# 2. Clustering algorithms (K-Means, Agglomerative/Ward, Gaussian Mixture, DBSCAN) and choice of *k*
# 3. Interpreting the clusters
# 4. Business validation metrics, recommendations, and robustness checks
#
# All random procedures use seed 5110.
#
# **Evidence standard used throughout.** Cluster labels, profiles and the business proxies are all computed on the same snapshot and share raw variables.
# Differences between clusters are therefore reported as *in-sample descriptions* (effect sizes), not as validation of business value;
# §4 sets out the prospective tests that would provide that validation.

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from scipy.optimize import linear_sum_assignment

SEED = 5110
rng = np.random.default_rng(SEED)
pd.set_option("display.width", 200, "display.max_columns", 40,
              "display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid", font_scale=0.9)

# %% [markdown]
# ## 1. Data preprocessing
# ### 1.1 First look and data-quality audit

# %%
raw = pd.read_csv("Customer_Data.csv")
print(raw.shape)
display(raw.head())
display(raw.describe().T)

# %%
audit = pd.Series({
    "Rows": len(raw),
    "Duplicate rows": raw.duplicated().sum(),
    "Duplicate CUST_ID": raw.CUST_ID.duplicated().sum(),
    "Missing CREDIT_LIMIT": raw.CREDIT_LIMIT.isna().sum(),
    "Missing MINIMUM_PAYMENTS": raw.MINIMUM_PAYMENTS.isna().sum(),
    "  ...of which PAYMENTS == 0": (raw.MINIMUM_PAYMENTS.isna() & (raw.PAYMENTS == 0)).sum(),
    "CASH_ADVANCE_FREQUENCY > 1 (exceeds a share-of-months definition)": (raw.CASH_ADVANCE_FREQUENCY > 1).sum(),
    "|ONEOFF + INSTALLMENTS - PURCHASES| > $1": ((raw.ONEOFF_PURCHASES + raw.INSTALLMENTS_PURCHASES - raw.PURCHASES).abs() > 1).sum(),
    "BALANCE > CREDIT_LIMIT (over limit)": (raw.BALANCE > raw.CREDIT_LIMIT).sum(),
    "TENURE < 12 months": (raw.TENURE < 12).sum(),
})
audit.to_frame("count")

# %% [markdown]
# **What do the frequency fields measure?** The field descriptions on the data page could not be read, so the structure is checked directly.
# If a frequency is (number of months with the activity) / TENURE, then frequency × TENURE should be a whole number and should not exceed the transaction count.

# %%
fq = []
for f, t in [("CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX"), ("PURCHASES_FREQUENCY", "PURCHASES_TRX")]:
    v = raw[f] * raw.TENURE
    fq.append({"field": f, "freq x TENURE is integer": np.isclose(v, v.round(), atol=1e-3).mean(),
               "freq x TENURE <= TRX": (v.round() <= raw[t]).mean(),
               "freq x TENURE < TRX (strictly)": int((v.round() < raw[t]).sum()),
               "freq > 1": int((raw[f] > 1).sum())})
display(pd.DataFrame(fq).round(4))
raw.loc[raw.CASH_ADVANCE_FREQUENCY > 1, ["CASH_ADVANCE_FREQUENCY", "TENURE", "CASH_ADVANCE_TRX"]].assign(
    freq_x_tenure=lambda d: (d.CASH_ADVANCE_FREQUENCY * d.TENURE).round(), trx_per_month=lambda d: d.CASH_ADVANCE_TRX / d.TENURE)

# %% [markdown]
# The values are consistent with TENURE as the denominator (frequency × TENURE is always a whole number), and the implied numerator never exceeds the transaction count.
# It is also strictly smaller than the count for 2,735 customers, including all 8 rows above 1, so the field is **not** "total transactions / TENURE".
# The numerator cannot be identified exactly. For the 8 rows above 1 the numerator exceeds the number of months in tenure, which is inconsistent with a share-of-months definition;
# capping them at 1 is therefore a **modelling assumption**, and §4.3 compares it with keeping the original values.

# %% [markdown]
# **Skewness.** Monetary and count variables are extremely right-skewed (skew of 2 to 14). Distance-based methods
# would be dominated by a handful of very large customers, so these variables are log-transformed below.
# There is no evidence that the extreme values are errors, so they are kept, not deleted.

# %%
num_cols = raw.columns.drop("CUST_ID")
raw[num_cols].skew().sort_values(ascending=False).to_frame("skewness").T

# %%
fig, axes = plt.subplots(3, 6, figsize=(16, 7))
for ax, c in zip(axes.ravel(), num_cols):
    ax.hist(raw[c].dropna(), bins=40, color="#4C72B0"); ax.set_title(c, fontsize=7)
axes.ravel()[-1].axis("off"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.2 Cleaning decisions
#
# | Issue | Treatment | Reason |
# |---|---|---|
# | `CUST_ID` | Dropped from features (kept as index) | Identifier, no behavioural content |
# | 1 missing `CREDIT_LIMIT` | Row dropped (n = 8,949) | One row; imputing a credit limit would invent a key variable |
# | 313 missing `MINIMUM_PAYMENTS` | Imputed with the median `MINIMUM_PAYMENTS` of customers in the same `BALANCE` decile; an imputation flag is kept for reporting | Minimum payment is related to balance (corr 0.40); 240 of the 313 have no payments at all, so the gap looks like "not computed" rather than "zero due". Label sensitivity in §4.3; effect on the payment indicator in §3.1 and §4.1 |
# | 8 rows with `CASH_ADVANCE_FREQUENCY` > 1 | Capped at 1 (modelling assumption) | Inconsistent with a share-of-months definition (see audit above); compared with the uncapped values in §4.3 |
# | 19 rows where one-off + instalment differs from purchases by more than $1 | Kept; `PURCHASES` itself is not used as a clustering input | `PURCHASES` is approximately redundant with the sum of its two parts |
# | Heavy right skew in $ and count variables | `log1p` transform | Reduces the leverage of extreme users on Euclidean distances; keeps zeros |
# | Different units ($, counts, shares, months) | `StandardScaler` (mean 0, sd 1) | Puts every column on the same scale. It does not equalise behaviours: a behaviour measured by several columns (amount, count and frequency) still carries more weight in Euclidean distance; §4.3 refits on a de-duplicated feature set |
# | Categorical encoding | None needed | All variables are numeric. `TENURE` (6 to 12 months) is ordinal and is kept numeric. No sensitive attributes (gender, age) are present |

# %%
df = raw.dropna(subset=["CREDIT_LIMIT"]).set_index("CUST_ID").copy()
dec = pd.qcut(df.BALANCE.rank(method="first"), 10, labels=False)
med_by_dec = df.groupby(dec).MINIMUM_PAYMENTS.median()
df["MINIMUM_PAYMENTS_IMPUTED"] = df.MINIMUM_PAYMENTS.isna()
df["MINIMUM_PAYMENTS"] = df.MINIMUM_PAYMENTS.fillna(dec.map(med_by_dec))
df["CASH_ADVANCE_FREQUENCY"] = df.CASH_ADVANCE_FREQUENCY.clip(upper=1)
print(df.shape, "missing left:", df.isna().sum().sum())

# %% [markdown]
# ### 1.3 Feature set, transformation and scaling
# Clustering features: 16 of the 17 behavioural variables (all except `PURCHASES`). Monetary amounts and transaction counts are `log1p`-transformed;
# frequencies and `PRC_FULL_PAYMENT` are already shares in [0, 1] and are left as they are.
#
# Business KPIs used later for **profiling only** (not clustering inputs, although they are built from the same raw variables):
# * `UTILIZATION` = BALANCE / CREDIT_LIMIT
# * `PAY_TO_MIN` = PAYMENTS / MINIMUM_PAYMENTS. Below 1 means the aggregate payment amount is lower than the recorded or imputed minimum payment amount; reading it as bill-level underpayment would require both fields to cover the same period
# * `CASH_SHARE` = CASH_ADVANCE / (CASH_ADVANCE + PURCHASES)
# * `AVG_TICKET` = PURCHASES / PURCHASES_TRX

# %%
money = ["BALANCE", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
         "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "CASH_ADVANCE_TRX", "PURCHASES_TRX"]
shares = ["BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
          "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
FEATS = money + shares + ["TENURE"]

X_raw = df[FEATS].copy()
X_raw[money] = np.log1p(X_raw[money])
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print("Features:", len(FEATS)); print("Skewness after log1p:")
print(X_raw.skew().round(2).to_dict())

kpi = pd.DataFrame(index=df.index)
kpi["UTILIZATION"] = df.BALANCE / df.CREDIT_LIMIT
kpi["PAY_TO_MIN"] = df.PAYMENTS / df.MINIMUM_PAYMENTS
kpi["CASH_SHARE"] = (df.CASH_ADVANCE / (df.CASH_ADVANCE + df.PURCHASES)).fillna(0)
kpi["AVG_TICKET"] = (df.PURCHASES / df.PURCHASES_TRX).replace([np.inf], np.nan).fillna(0)

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X, columns=FEATS).corr(), cmap="RdBu_r", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 6}); plt.title("Correlation of transformed, scaled features"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.4 Dimensionality reduction (PCA)
# Several features measure the same behaviour (e.g. cash-advance amount, count and frequency; correlation about 0.8 to 0.9).
# PCA summarises this redundancy. Components are kept until about 85% of variance is explained.

# %%
pca_full = PCA(random_state=SEED).fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
N_PC = int(np.searchsorted(cum, 0.85) + 1)
print(f"Components for >=85% variance: {N_PC} (cumulative {cum[N_PC-1]:.1%})")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(range(1, len(cum)+1), pca_full.explained_variance_ratio_, color="#4C72B0", label="per component")
ax.plot(range(1, len(cum)+1), cum, "o-", color="#C44E52", label="cumulative")
ax.axhline(0.85, ls="--", c="grey"); ax.set_xlabel("component"); ax.legend(); plt.tight_layout(); plt.show()

pca = PCA(n_components=N_PC, random_state=SEED)
Z = pca.fit_transform(X)
load = pd.DataFrame(pca.components_[:4].T, index=FEATS, columns=[f"PC{i}" for i in range(1, 5)])
load.round(2)

# %% [markdown]
# Reading the loadings: **PC1** contrasts purchase activity (positive: purchase frequency, counts, instalments) with cash-advance activity (negative: cash-advance amount, count, frequency).
# **PC2** loads positively on almost everything, most on balance, minimum payments, balance frequency and payments: it reflects overall account size/activity rather than one behaviour.
# PC3 separates one-off from instalment buying.
# The main clustering is run on the full scaled feature space; §2.5 repeats it in PCA space and measures how much the clusters change.

# %% [markdown]
# ## 2. Clustering algorithms and the number of clusters
# Four algorithms with different assumptions:
# * **K-Means**: spherical clusters of similar size around centroids; fast, gives a simple assignment rule.
# * **Agglomerative (Ward linkage)**: bottom-up merging that minimises within-cluster variance; no random start; shows the hierarchy.
# * **Gaussian Mixture Model (full covariance)**: soft, elliptical clusters; choose *k* by BIC.
# * **DBSCAN**: density-based, finds arbitrarily shaped clusters and flags noise; no *k* needed.
#
# For *k* = 2 to 10: elbow (inertia), silhouette (higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher better).
#
# **How much weight these scores deserve.** All three are computed in the same Euclidean space. At a fixed *k*, CH = [(n − k)/(k − 1)] × (T − W)/W, so minimising W is equivalent to maximising CH:
# CH is mathematically aligned with the K-Means objective and provides no independent evidence for choosing K-Means. Silhouette (Euclidean distances) and DB (centroids and within-cluster spread)
# tend to reward compact, well-separated clusters. They show how each method performs under this geometric standard, so they are used as supporting evidence only.

# %%
K = range(2, 11)
rows = []
km_models, agg_labels, gmm_models = {}, {}, {}
for k in K:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X); km_models[k] = km
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X); agg_labels[k] = ag
    gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=SEED).fit(X); gmm_models[k] = gm
    for name, lab in [("K-Means", km.labels_), ("Ward", ag), ("GMM", gm.predict(X))]:
        sizes = np.bincount(lab, minlength=k) / len(lab)
        rows.append(dict(method=name, k=k, silhouette=silhouette_score(X, lab),
                         davies_bouldin=davies_bouldin_score(X, lab),
                         calinski_harabasz=calinski_harabasz_score(X, lab),
                         smallest_cluster=sizes.min(),
                         inertia=km.inertia_ if name == "K-Means" else np.nan,
                         bic=gm.bic(X) if name == "GMM" else np.nan))
scores = pd.DataFrame(rows)
scores.pivot(index="k", columns="method", values=["silhouette", "davies_bouldin", "smallest_cluster"]).round(3)

# %%
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
km_s = scores[scores.method == "K-Means"]
ax[0].plot(km_s.k, km_s.inertia, "o-"); ax[0].set_title("K-Means elbow (inertia)")
for m, g in scores.groupby("method"):
    ax[1].plot(g.k, g.silhouette, "o-", label=m); ax[2].plot(g.k, g.davies_bouldin, "o-", label=m)
ax[1].set_title("Silhouette (higher = better)"); ax[2].set_title("Davies-Bouldin (lower = better)"); ax[1].legend()
gm_s = scores[scores.method == "GMM"]
ax[3].plot(gm_s.k, gm_s.bic, "o-", c="C1"); ax[3].set_title("GMM BIC (lower = better)")
for a in ax: a.set_xlabel("k")
plt.tight_layout(); plt.savefig("fig_k_selection.png", dpi=200); plt.show()

# %% [markdown]
# Elbow gains per extra cluster (percentage drop in inertia):

# %%
km_s = km_s.set_index("k")
(km_s.inertia.pct_change() * -100).round(1).to_frame("% inertia reduction vs k-1").T

# %% [markdown]
# ### 2.1 Subsampling stability
# A segmentation that changes when a different sample of customers is drawn is not usable.
# For each method and *k*, we refit on the **same** 20 random 80% subsamples (drawn without replacement) with the **same initialisation settings** as the full-data fit,
# and compare subsample labels with full-data labels on the customers in the subsample (adjusted Rand index, ARI; 1 = identical).
# This measures sensitivity to sample perturbation in a fixed feature space: the scaler is fitted on all customers (fitting it on 80% changes means and SDs negligibly), and ARI is computed on the customers used in the refit.

# %%
SUBS = [rng.choice(len(X), int(0.8 * len(X)), replace=False) for _ in range(20)]

def stability(method, k):
    ref = {"K-Means": km_models[k].labels_, "Ward": agg_labels[k], "GMM": gmm_models[k].predict(X)}[method]
    out = []
    for idx in SUBS:
        Xs = X[idx]
        if method == "K-Means":
            lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xs)
        elif method == "Ward":
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
        else:
            lab = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=SEED).fit(Xs).predict(Xs)
        out.append(adjusted_rand_score(ref[idx], lab))
    return np.mean(out), np.min(out)

stab = []
for m in ["K-Means", "Ward", "GMM"]:
    for k in range(2, 8):
        mu, mn = stability(m, k); stab.append(dict(method=m, k=k, mean_ARI=mu, min_ARI=mn))
stab = pd.DataFrame(stab)
stab.pivot(index="k", columns="method", values="mean_ARI").round(3)

# %% [markdown]
# ### 2.2 Hierarchical clustering: dendrogram (Ward, 2,000-customer sample for readability)

# %%
samp = rng.choice(len(X), 2000, replace=False)
Lk = linkage(X[samp], method="ward")
plt.figure(figsize=(10, 3.5))
dendrogram(Lk, truncate_mode="lastp", p=30, no_labels=True, color_threshold=Lk[-4, 2])
plt.title("Ward dendrogram (top 30 merges); colour cut at 4 clusters"); plt.ylabel("merge distance"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2.3 DBSCAN
# `min_samples` = 2 × dimensions = 32 (common rule). `eps` is read from the knee of the sorted 32-nearest-neighbour distance plot, then a grid around it is tried.

# %%
MS = 2 * X.shape[1]
dist, _ = NearestNeighbors(n_neighbors=MS).fit(X).kneighbors(X)
kd = np.sort(dist[:, -1])
plt.figure(figsize=(6, 3)); plt.plot(kd); plt.ylabel(f"{MS}-NN distance"); plt.xlabel("customers (sorted)")
plt.title("k-distance plot"); plt.tight_layout(); plt.show()
db_rows = []
for eps in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    lab = DBSCAN(eps=eps, min_samples=MS).fit_predict(X)
    core = lab >= 0; ncl = len(set(lab[core]))
    sil = silhouette_score(X[core], lab[core]) if ncl > 1 else np.nan
    sizes = np.bincount(lab[core]) / len(lab) if ncl else []
    db_rows.append(dict(eps=eps, n_clusters=ncl, noise_share=1 - core.mean(),
                        largest_cluster=max(sizes) if ncl else np.nan, silhouette_non_noise=sil))
db_tab = pd.DataFrame(db_rows); db_tab

# %% [markdown]
# **DBSCAN result.** With `min_samples` = 32 and these six `eps` values, DBSCAN either labels a large share as noise with poorly separated clusters
# (`eps` = 1.5: four clusters, 28.9% noise, non-noise silhouette −0.025) or puts almost everyone into one cluster (`eps` ≥ 2.0).
# So no usable density-based segmentation was found under these settings; this does not prove that no density structure exists.
#
# **Who are the noise points?** Being labelled noise only means low local density in this feature space. Their profile (below) shows they are mainly
# **short-tenure, low-payment** customers, not heavy users.

# %%
noise25 = DBSCAN(eps=2.5, min_samples=MS).fit_predict(X) == -1
ncols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_TRX", "TENURE"]
npro = pd.DataFrame({"noise (eps=2.5) median": df[noise25][ncols].median(), "others median": df[~noise25][ncols].median()})
npro.loc["share TENURE < 12"] = [(df[noise25].TENURE < 12).mean(), (df[~noise25].TENURE < 12).mean()]
npro.loc["n"] = [noise25.sum(), (~noise25).sum()]
npro.round(2)

# %% [markdown]
# ### 2.4 Choosing the method and *k*

# %%
summary_k = scores.merge(stab, on=["method", "k"], how="left")
summary_k[summary_k.k.between(2, 8)].pivot(index="k", columns="method",
          values=["silhouette", "davies_bouldin", "mean_ARI", "smallest_cluster"]).round(3)

# %% [markdown]
# **Method selection** (the geometric scores are supporting evidence only; see the note on CH, silhouette and DB above)
#
# * **DBSCAN:** no usable segmentation found under the tested settings (§2.3).
# * **Ward:** weakest silhouette and DB of the three at most *k*, and the least stable under subsampling (mean ARI 0.34 to 0.60 across *k*). At *k* = 4, Ward had a mean subsampling ARI of 0.511, compared with 0.975 for K-Means. This indicates lower assignment stability under the subsampling protocol used here.
# * **GMM:** BIC keeps falling until *k* = 8, where the smallest component holds 4.5% of customers. One possible explanation, not verified here, is that full-covariance Gaussians fit the spikes at 0 and 1 in the share variables. At *k* = 4 GMM is less stable (ARI 0.72) and its smallest cluster is 11.9%.
# * **K-Means: selected.** Among the *k* = 4 candidates it has the highest subsampling stability (0.975 vs 0.511 for Ward and 0.720 for GMM) and the largest smallest cluster (17.2%), and it assigns new customers with a fixed nearest-centroid rule. It also has the best silhouette and DB, but those scores reward compact clusters (and CH is aligned with its objective), so they do not decide the choice.
#
# **Number of clusters: an operational trade-off.** Silhouette is highest at *k* = 2 (0.235). The *k* = 2 profile (above) splits purchase-led customers (median purchases $701, no cash advance) from cash-advance-led customers (median purchases $0, cash advance $1,361).
# *k* = 5 scores slightly better than *k* = 4 on silhouette (0.220 vs 0.218) and DB (1.579 vs 1.662), with similar stability (0.978 vs 0.975), and the elbow has no sharp kink (inertia falls 15.8%, 10.0%, 9.4% for *k* = 3, 4, 5).
# **k = 4 is chosen for operational reasons:** its smallest group is larger (17.2% vs 13.3%), and its four groups map to distinct actions. *k* = 5 splits off about 1,200 customers who both purchase and take cash advances, which is a possible refinement for campaigns.
#
# **Caveat.** A silhouette near 0.22 means the clusters overlap: they are regions of a continuum rather than separated groups. Agreement between methods at *k* = 4 is moderate (ARI 0.38 to 0.50), so segment themes are more robust than exact boundaries.

# %%
K_STAR = 4
km = km_models[K_STAR]
labels_raw = km.labels_
# relabel clusters by size (0 = largest) for stable naming
order = pd.Series(labels_raw).value_counts().index
remap = {old: new for new, old in enumerate(order)}
labels = np.vectorize(remap.get)(labels_raw)
df["cluster"] = labels
print(pd.Series(labels).value_counts().sort_index())

# agreement between methods at k*
agree = pd.DataFrame({
    "Ward vs K-Means": [adjusted_rand_score(labels, agg_labels[K_STAR])],
    "GMM vs K-Means": [adjusted_rand_score(labels, gmm_models[K_STAR].predict(X))],
    "Ward vs GMM": [adjusted_rand_score(agg_labels[K_STAR], gmm_models[K_STAR].predict(X))]}, index=["ARI"])
agree.round(3)

# %% [markdown]
# **What changes with k = 2, 3 or 5?** Profile of the k = 2 solution, cross-tabulation of the k = 4 labels against neighbouring solutions, and the k = 5 profile.

# %%
df2 = df.assign(c2=km_models[2].labels_)
display(df2.groupby("c2")[["BALANCE", "PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT"]].median().T.round(2))
display(df2.c2.value_counts().sort_index())
for k in [3, 5]:
    print(f"k=4 (rows) vs k={k} (columns)"); display(pd.crosstab(labels, km_models[k].labels_))
df5 = df.assign(c5=km_models[5].labels_)
display(df5.groupby("c5")[["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
                          "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT", "PRC_FULL_PAYMENT"]].median().T.round(2))
df5.c5.value_counts().sort_index()

# %% [markdown]
# ### 2.5 Impact of dimensionality reduction
# Re-run K-Means on the PCA scores (N_PC components, about 85% of variance) and on only the first 2 components, and compare with the full-feature solution.

# %%
pca_rows = []
for name, data in [("Full 16 features", X), (f"PCA {N_PC} comps", Z), ("PCA 2 comps", Z[:, :2])]:
    lab = KMeans(n_clusters=K_STAR, n_init=20, random_state=SEED).fit_predict(data)
    pca_rows.append(dict(space=name, ARI_vs_full=adjusted_rand_score(labels, lab),
                         silhouette_own_space=silhouette_score(data, lab),
                         silhouette_full_space=silhouette_score(X, lab)))
pca_cmp = pd.DataFrame(pca_rows); pca_cmp.round(3)

# %% [markdown]
# **Impact of PCA.** With 7 components (89% of variance), K-Means gives almost the same clusters (ARI 0.979 vs full features). The silhouette rises in PCA space (0.249) only because the discarded 11% of variance is removed from the distance; measured in the full feature space the silhouette is unchanged (0.219). So PCA here removes redundancy and speeds up computation but does not create better segments.
# With only 2 components the silhouette looks much better (0.391), but agreement drops to ARI 0.52 and the full-space silhouette falls to 0.189: the 2-D view loses PC3 (one-off vs instalment buying), which is one of the key differences between segments. Two components are therefore used for visualisation only. The final model is fitted on the full 16 scaled features so that centroids can be read directly in terms of the original variables.

# %%
pal = sns.color_palette("Set2", K_STAR)
fig, ax = plt.subplots(figsize=(6.5, 5))
for c in range(K_STAR):
    m = labels == c
    ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5, color=pal[c], label=f"Cluster {c}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%}): purchase (+) vs cash-advance (−) activity")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%}): overall account size / activity")
ax.legend(markerscale=4); ax.set_title(f"K-Means (k={K_STAR}) in PCA space"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Interpreting the clusters
# ### 3.1 Profiles in original units (medians, robust to skew) and KPIs

# %%
prof_cols = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
             "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PURCHASES_TRX", "CREDIT_LIMIT",
             "PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"]
full = df.join(kpi)
prof = full.groupby("cluster")[prof_cols + list(kpi.columns)].median().T
prof.loc["Share of customers"] = full.cluster.value_counts(normalize=True).sort_index()
prof.loc["Mean PRC_FULL_PAYMENT"] = full.groupby("cluster").PRC_FULL_PAYMENT.mean()
prof.loc["Share with PURCHASES = 0"] = full.groupby("cluster").PURCHASES.apply(lambda s: (s == 0).mean())
prof.round(2)

# %% [markdown]
# **Payment-below-minimum indicator and the imputation.** Labels are held fixed; only the indicator is recomputed, first on all customers and then only on customers whose
# MINIMUM_PAYMENTS was observed. The observed-only figure can still be affected by selection, because missingness is related to zero payments.

# %%
full["UNDERPAY"] = full.PAY_TO_MIN < 1
obs = ~full.MINIMUM_PAYMENTS_IMPUTED
gp = full.groupby("cluster")
pay_tab = pd.DataFrame({
    "n": gp.size(),
    "imputed_MIN_share": gp.MINIMUM_PAYMENTS_IMPUTED.mean(),
    "imputed_and_PAYMENTS_0_share": gp.apply(lambda x: (x.MINIMUM_PAYMENTS_IMPUTED & (x.PAYMENTS == 0)).mean()),
    "below_min_all": gp.UNDERPAY.mean(),
    "n_observed": full[obs].groupby("cluster").size(),
    "below_min_observed_only": full[obs].groupby("cluster").UNDERPAY.mean()})
print("Share below minimum among imputed rows:", round(full.loc[~obs, "UNDERPAY"].mean(), 3))
pay_tab.round(3)

# %%
zprof = pd.DataFrame(X, columns=FEATS, index=df.index).groupby(labels).mean().T
plt.figure(figsize=(7, 6.5))
sns.heatmap(zprof, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"label": "mean z-score"})
plt.xlabel("cluster"); plt.title("Cluster centroids (standardised features)"); plt.tight_layout()
plt.savefig("fig_profile.png", dpi=200); plt.show()

# %% [markdown]
# ### 3.2 Which features distinguish the clusters?
# Two views: (a) eta-squared from one-way ANOVA on each scaled feature (share of a feature's variance explained by cluster membership);
# (b) a shallow decision tree trained to predict the cluster, which gives readable rules and tree-based importances.

# %%
Xdf = pd.DataFrame(X, columns=FEATS)
def eta2(x, g):
    grand = x.mean(); ssb = sum(len(x[g == c]) * (x[g == c].mean() - grand) ** 2 for c in np.unique(g))
    return ssb / ((x - grand) ** 2).sum()
imp = pd.DataFrame({"eta_squared": [eta2(Xdf[c].values, labels) for c in FEATS]}, index=FEATS)
tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_raw, labels)
imp["tree_importance"] = tree.feature_importances_
imp = imp.sort_values("eta_squared", ascending=False)
print("Depth-3 tree, agreement with the fixed K-Means labels: training", round(tree.score(X_raw, labels), 3),
      "| 5-fold out-of-sample", round(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED), X_raw, labels, cv=5).mean(), 3))
imp.round(3)

# %%
print(export_text(tree, feature_names=FEATS, decimals=2))
# note: money/count thresholds are on log1p scale; exp(t)-1 converts back to $ or counts

# %% [markdown]
# **Who is in each cluster** (medians unless stated; numbering by size)
#
# | Cluster | Name | Share | Behaviour |
# |---|---|---|---|
# | 0 | **Cash-advance revolvers** | 33% | Median purchases $0 (62% made no purchase); cash advances $1,391 and a $1,750 balance; utilisation 66%; rarely pays in full (mean 3%); for 43% the aggregate payment is below the recorded or imputed minimum (42.7% among the 2,932 with an observed minimum) |
# | 1 | **Active all-round spenders** | 27% | $1,788 purchases (one-off $1,072 plus instalments $465), median recorded purchase frequency 0.92, 26 transactions; highest credit limit ($6,000); pays 3.9× the minimum, but many still revolve |
# | 2 | **Instalment buyers** | 23% | $474 purchases, nearly all instalments, median recorded purchase frequency 0.92; low balance ($150) and limit ($2,000); highest full-payment rate (mean 29%) |
# | 3 | **Low-activity** | 17% | Balance $37, purchases $166, median recorded purchase frequency 0.17, utilisation 1%. 15.9% have an imputed minimum payment; the below-minimum share falls from 26.1% to 16.7% when restricted to the 1,298 customers with observed minimum payments |
#
# **Robustness of the C1/C2 split.** With the reduced-redundancy feature set (§4.3 f), the cash-advance and low-activity profiles remain recognisable; between the two purchasing groups the balance difference becomes more prominent and the original purchase-type boundary weakens (one-off purchase medians still differ, $456 vs $37). The C1/C2 boundary therefore depends partly on the duplicated purchase-type features.
#
# **Most distinguishing features.** Purchase frequency (eta² = 0.72) and purchase transaction count (0.69) separate clusters most, followed by one-off and instalment frequencies, instalment and cash-advance amounts, and balance (eta² about 0.5 to 0.6).
# Credit limit, payments, full-payment share and tenure explain little (eta² below 0.22): the clusters differ mainly in *how* the card is used.
# Eta² supports purchase frequency as the single variable that best separates the current labels; it does not measure how much the clustering would change without it,
# and, like all profiles here, it is descriptive because the labels were formed from these variables.
#
# A depth-3 decision tree gives readable rules: 85.9% agreement with the fixed K-Means labels on the training data and 84.6% in 5-fold out-of-sample agreement
# (labels from the full-data clustering held fixed; preprocessing and clustering are not refitted within folds).
# Its splits use purchase frequency, cash-advance amount, one-off purchases, balance, minimum payments and balance frequency; the PURCHASES_TRX split sends both leaves to C1,
# so it does not change the tree's predictions. (Thresholds are on the log1p scale; exp(t) − 1 converts them to dollars.)

# %% [markdown]
# ## 4. Business validation and discussion
# ### 4.1 In-sample business proxies (descriptive, not validation)
# Geometric scores do not show that segments matter commercially. Two business proxies are computed per cluster. Both are built from the same raw variables used to form
# the clusters, so differences between clusters are **in-sample descriptions** of the fitted segments. Conventional p-values (e.g. Kruskal-Wallis) do not account for the
# data-driven formation of the labels and are therefore not reported; epsilon² is kept as a descriptive effect size.
#
# **Proxy 1: illustrative revenue scenario.** Parameters are assumptions, not issuer data:
# one year of interest on the current balance not paid in full (BALANCE × (1 − PRC_FULL_PAYMENT) × 20% APR; this assumes the full-payment share approximates the share of balance not charged interest)
# \+ 3% fee on CASH_ADVANCE + 1.5% interchange on PURCHASES. Interest is an annual run-rate on a point-in-time balance, while fees and interchange use amounts over the observation window,
# whose length is not documented consistently (data page: last 6 months; frequency fields: TENURE months). Values are per-customer **means** (profiles in §3 are medians).
#
# **Proxy 2: payment indicator.** Share of customers whose aggregate PAYMENTS are below the recorded or imputed MINIMUM_PAYMENTS, and share with utilisation above 80%.

# %%
APR, CA_FEE, INTERCHANGE, LOSS = 0.20, 0.03, 0.015, 0.05
full["REV_INTEREST"] = full.BALANCE * (1 - full.PRC_FULL_PAYMENT) * APR
full["REV_CASHFEE"] = CA_FEE * full.CASH_ADVANCE
full["REV_INTERCHANGE"] = INTERCHANGE * full.PURCHASES
full["REVENUE_PROXY"] = full[["REV_INTEREST", "REV_CASHFEE", "REV_INTERCHANGE"]].sum(1)
full["HIGH_UTIL"] = full.UTILIZATION > 0.8
full["SCENARIO_VALUE"] = full.REVENUE_PROXY - LOSS * full.BALANCE * full.UNDERPAY
print(f"Interest share of the total revenue proxy: {full.REV_INTEREST.sum() / full.REVENUE_PROXY.sum():.1%}")

g = full.groupby("cluster")
biz = pd.DataFrame({
    "share": g.size() / len(full),
    "revenue_proxy_mean": g.REVENUE_PROXY.mean(),
    "revenue_proxy_median": g.REVENUE_PROXY.median(),
    "interest_%": g.REV_INTEREST.sum() / g.REVENUE_PROXY.sum() * 100,
    "cashfee_%": g.REV_CASHFEE.sum() / g.REVENUE_PROXY.sum() * 100,
    "interchange_%": g.REV_INTERCHANGE.sum() / g.REVENUE_PROXY.sum() * 100,
    "below_min_all": g.UNDERPAY.mean(),
    "below_min_observed_only": full[obs].groupby("cluster").UNDERPAY.mean(),
    "high_util_share": g.HIGH_UTIL.mean(),
    "scenario_value_mean": g.SCENARIO_VALUE.mean()})
biz.round(3)

# %% [markdown]
# **Benchmark: a simple rule (BALANCE quartiles).** Both groupings are compared with the same evidence standard: epsilon² from the Kruskal-Wallis H statistic (descriptive only).
# The payment indicator is shown on all customers and on customers with an observed minimum payment (labels held fixed).

# %%
full["BAL_Q"] = pd.qcut(full.BALANCE.rank(method="first"), 4, labels=False)
def eps2(by, col, mask=None):
    d = full if mask is None else full[mask]
    return stats.kruskal(*[x[col].astype(float) for _, x in d.groupby(by)])[0] / (len(d) - 1)
bench = pd.DataFrame({b: {"revenue_proxy": eps2(b, "REVENUE_PROXY"),
                          "below_min (all)": eps2(b, "UNDERPAY"),
                          "below_min (observed MIN only)": eps2(b, "UNDERPAY", obs),
                          "interchange": eps2(b, "REV_INTERCHANGE"),
                          "cash_advance_share": eps2(b, "CASH_SHARE")}
                      for b in ["cluster", "BAL_Q"]}).rename(columns={"cluster": "K-Means (k=4)", "BAL_Q": "Balance quartiles"})
display(bench.round(3))
print("Below-minimum share by balance quartile:", full.groupby("BAL_Q").UNDERPAY.mean().round(3).to_dict())

# %% [markdown]
# Balance quartiles showed stronger in-sample separation on the illustrative revenue proxy (ε² = 0.924 vs. 0.481 for K-Means). This was expected because about 87% of the proxy came from an interest term calculated using balance.
# Among customers with observed minimum payments, balance quartiles also showed greater separation on the payment-to-minimum indicator (ε² = 0.121 vs. 0.063; cluster labels held fixed).
# K-Means showed greater separation on estimated interchange revenue (0.60 vs. 0.03) and cash-advance share (0.64 vs. 0.22), both derived from behaviours used to form the clusters.
# These differences describe the fitted segments and can help formulate offer hypotheses. To assess business value, we would track 12-month retention after assignment and measure incremental net contribution in a randomized campaign.

# %% [markdown]
# ### 4.2 Scenario sensitivity (supporting detail)
# Cluster ranking by mean revenue proxy under alternative APR and interchange assumptions, and the assumed loss rate at which two clusters' mean scenario values would be equal.
# The loss rate is a scenario parameter applied to the balances of customers below the minimum, **not** an estimated default rate.

# %%
sens = []
for apr in [0.15, 0.20, 0.25]:
    for ich in [0.01, 0.015, 0.02]:
        r = full.BALANCE * (1 - full.PRC_FULL_PAYMENT) * apr + CA_FEE * full.CASH_ADVANCE + ich * full.PURCHASES
        m = r.groupby(full.cluster).mean()
        sens.append(dict(APR=apr, interchange=ich, ranking=" > ".join(str(i) for i in m.sort_values(ascending=False).index),
                         **{f"C{c}": round(m[c]) for c in range(K_STAR)}))
display(pd.DataFrame(sens))
exp_bal = (full.BALANCE * full.UNDERPAY).groupby(full.cluster).mean()
rev = full.groupby("cluster").REVENUE_PROXY.mean()
pairs = [dict(higher=a, lower=b, equalising_loss_rate=(rev[a] - rev[b]) / (exp_bal[a] - exp_bal[b]))
         for a in range(K_STAR) for b in range(K_STAR) if a != b and rev[a] > rev[b] and exp_bal[a] > exp_bal[b]]
pd.DataFrame(pairs).round(3)

# %% [markdown]
# Under the baseline parameters, C0 and C1 have equal mean scenario value when the assumed loss on the relevant balances is 17.9%. This is a property of the scenario, not a default threshold,
# and it is not evidence that C0 is the most profitable segment.

# %% [markdown]
# ### 4.3 Robustness of the segmentation
# Each check refits K-Means (k = 4, same settings) and compares labels with the main solution on the **same customers** (ARI), plus the share of customers whose cluster changes
# after matching clusters one-to-one (Hungarian matching on the cross-tab).
# * (a) MINIMUM_PAYMENTS imputed with the overall median instead of the balance-decile median.
# * (b) CASH_ADVANCE_FREQUENCY kept at its original values (no cap); also reports how many of the 8 affected customers change cluster.
# * (c) Drop the DBSCAN low-density points (eps = 2.5).
# * (d) **Same-tenure subsample:** only customers with TENURE = 12, refitted on that subset. This needs no assumption about the observation window, but only shows whether the segmentation is similar among customers with the same tenure; a change could also reflect a different customer mix.
# * (e) **Monthly averages:** flow fields (purchase amounts, cash advance, payments, minimum payments, transaction counts) divided by TENURE; stock fields (BALANCE, CREDIT_LIMIT) unchanged. This *assumes* the flow fields accumulate over the TENURE months.
# * (f) **Reduced-redundancy feature set:** fewer duplicate representations (cash advance: amount only; purchases: one-off amount, instalment amount and purchase frequency; dropping CASH_ADVANCE_TRX, CASH_ADVANCE_FREQUENCY, PURCHASES_TRX and the two purchase-type frequencies): 11 features.
# * (g) No log transform.

# %%
def build(d, feats=FEATS, logcols=money, idx=None):
    Xa = d[feats].copy()
    Xa[[c for c in logcols if c in feats]] = np.log1p(Xa[[c for c in logcols if c in feats]])
    return StandardScaler().fit_transform(Xa)

def compare(Xalt, mask=None):
    lab = KMeans(n_clusters=K_STAR, n_init=20, random_state=SEED).fit_predict(Xalt)
    ref = labels if mask is None else labels[mask]
    ct = pd.crosstab(ref, lab).values
    r_, c_ = linear_sum_assignment(-ct)
    mapping = dict(zip(c_, r_)); matched = np.vectorize(mapping.get)(lab)
    return lab, matched, adjusted_rand_score(ref, lab), (matched != ref).mean()

df_base = raw.dropna(subset=["CREDIT_LIMIT"]).set_index("CUST_ID")
rob, prof_rob = [], {}

alt = df.copy(); alt.loc[alt.MINIMUM_PAYMENTS_IMPUTED, "MINIMUM_PAYMENTS"] = raw.MINIMUM_PAYMENTS.median()
_, mt, a, ch = compare(build(alt)); rob.append(("(a) Overall-median imputation", len(alt), a, ch, ""))

alt = df.copy(); alt["CASH_ADVANCE_FREQUENCY"] = df_base.loc[df.index, "CASH_ADVANCE_FREQUENCY"]
_, mt, a, ch = compare(build(alt)); m8 = (alt.CASH_ADVANCE_FREQUENCY > 1).values
rob.append(("(b) No cap on CASH_ADVANCE_FREQUENCY", len(alt), a, ch, f"{(mt[m8] != labels[m8]).sum()} of {m8.sum()} affected customers change cluster"))

keep = ~noise25
_, mt, a, ch = compare(build(df[keep]), keep); rob.append(("(c) Drop DBSCAN low-density points", keep.sum(), a, ch, ""))

t12 = (df.TENURE == 12).values
F12 = [f for f in FEATS if f != "TENURE"]
_, mt12, a, ch = compare(build(df[t12], F12), t12); rob.append(("(d) TENURE = 12 only (TENURE dropped as constant)", t12.sum(), a, ch, ""))
prof_rob["(d) TENURE = 12"] = mt12

flows = ["ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PAYMENTS", "MINIMUM_PAYMENTS", "CASH_ADVANCE_TRX", "PURCHASES_TRX"]
alt = df.copy(); alt[flows] = alt[flows].div(alt.TENURE, axis=0)
_, mte, a, ch = compare(build(alt)); rob.append(("(e) Monthly averages for flow fields", len(alt), a, ch, "assumes flows accumulate over TENURE"))
prof_rob["(e) Monthly averages"] = mte

FD = ["BALANCE", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
      "MINIMUM_PAYMENTS", "BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "PRC_FULL_PAYMENT", "TENURE"]
_, mtd, a, ch = compare(build(df, FD)); rob.append(("(f) Reduced-redundancy feature set (11 features)", len(df), a, ch, ""))
prof_rob["(f) Reduced-redundancy"] = mtd

lab_g, mtg, a, ch = compare(StandardScaler().fit_transform(df[FEATS]))
rob.append(("(g) No log transform", len(df), a, ch, f"cluster sizes {np.round(np.bincount(lab_g) / len(lab_g), 3)}"))

robust = pd.DataFrame(rob, columns=["check", "n customers", "ARI vs main (same customers)", "share changing cluster (matched)", "note"])
robust.round(3)

# %% [markdown]
# Do the four profiles survive? Medians of key variables by matched cluster, main solution vs checks (d), (e) and (f). (Profiles for check (g) are not shown; overall assignments were sensitive to removing the log transform.)

# %%
key = ["PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "BALANCE", "PURCHASES_FREQUENCY"]
pr = {"main": df.groupby(labels)[key].median()}
pr["(d) TENURE = 12"] = df[t12].groupby(prof_rob["(d) TENURE = 12"])[key].median()
pr["(e) Monthly averages"] = df.groupby(prof_rob["(e) Monthly averages"])[key].median()
pr["(f) Reduced-redundancy"] = df.groupby(prof_rob["(f) Reduced-redundancy"])[key].median()
sizes = {"main": pd.Series(labels).value_counts(normalize=True).sort_index(),
         "(d) TENURE = 12": pd.Series(prof_rob["(d) TENURE = 12"]).value_counts(normalize=True).sort_index(),
         "(e) Monthly averages": pd.Series(prof_rob["(e) Monthly averages"]).value_counts(normalize=True).sort_index(),
         "(f) Reduced-redundancy": pd.Series(prof_rob["(f) Reduced-redundancy"]).value_counts(normalize=True).sort_index()}
out = pd.concat({k: v.assign(share=sizes[k].values) for k, v in pr.items()}, axis=0)
out.round(2)

# %% [markdown]
# **What the checks show**
#
# * Labels change little with the imputation (a: ARI 0.989), the cap (b: retaining the original values changed about 0.4% of assignments overall; none of the eight affected customers changed clusters) and dropping the low-density points (c: ARI 0.976).
# * **Same tenure (d):** among the 7,584 customers with TENURE = 12, ARI 0.889 and 4.5% change cluster; the four profiles are preserved. **Monthly averages (e):** ARI 0.844, 6.2% change, profiles preserved (under its stated assumption).
# * **Reduced-redundancy feature set (f): ARI 0.505, 26% change cluster.** The cash-advance and low-activity profiles remain recognisable. Between the two purchasing groups the balance difference becomes more prominent ($1,469 vs $88 median balance) and the original purchase-type boundary weakens, although one-off purchases still differ ($456 vs $37 median).
#   So **the C1/C2 boundary depends partly on the duplicated purchase-type features**, and offers that rely on it should be treated as hypotheses to test first.
# * **No log transform (g):** ARI 0.281, with one cluster of 5.7%. This shows that the result is sensitive to the transformation; it does not by itself show which version is correct. Untransformed distances are dominated by a few very large values.

# %% [markdown]
# ### 4.4 Recommendations: offer hypotheses to test
# The segments suggest *which offer to test with whom*. None of the actions below is validated by this snapshot.
#
# | Segment | Hypothesis | Offer to test |
# |---|---|---|
# | C0 Cash-advance revolvers (33%) | Structured repayment lowers losses while keeping interest income | Pilot: payment reminders and auto-pay for customers below the minimum; instalment or balance-transfer plans for cash balances; purchase-activation offers. Do not promote cash advances. Evaluate on subsequent repayment and net contribution |
# | C1 Active all-round spenders (27%) | Rewards deepen spend (priority hypothesis; the C1/C2 boundary is sensitive to feature redundancy) | Rewards and premium-card upgrade, merchant offers on large one-off purchases. Limit increases only for customers who meet independent credit-eligibility rules (the historical payment-to-minimum ratio is not sufficient) |
# | C2 Instalment buyers (23%) | Instalment offers raise ticket size (priority hypothesis) | 0% instalment plans with partner merchants; limit-increase evaluation only after independent eligibility checks |
# | C3 Low-activity (17%) | Low-cost activation lifts usage | Test spend-and-get bonus or putting recurring bills on the card. Low usage is not treated as a prediction of attrition |
#
# **Business-grounded validation metrics (prospective)**
#
# | Metric | How to measure | Question answered |
# |---|---|---|
# | 12-month retention rate | Freeze cluster assignments at a base date; track whether each account stays active, with "active" defined in advance; compare with the balance-quartile benchmark | Does the segmentation carry retention information beyond a simple rule? |
# | Incremental net contribution per eligible customer | Within each segment, randomly assign the offer or a control; compare net contribution over the window, including reward cost, contact cost and observable credit losses. Where eligibility allows, run the same offer with treatment and control groups in both C1 and C2 to test directly whether response differs between them | Is choosing offers by segment worth it? |
#
# ### 4.5 If clusters aligned with sensitive attributes (gender, age)
# This file has no such variables, but credit limit, tenure and cash-advance use could act as proxies.
# * Keep sensitive attributes out of the clustering inputs, and test proxies: if the features predict a protected attribute well, review or drop the strongest ones.
# * Present segments by behaviour ("cash-advance revolvers"), never by demographic labels. Behavioural naming is a presentation choice; a diverse within-segment mix does **not** show that targeting is fair.
# * The fairness test is on decisions: where appropriate data exist, compare offer, approval, pricing and adverse-outcome rates for customers who meet the **same eligibility conditions**, reporting denominators and uncertainty. The four-fifths rule can serve only as a screening signal.
# * Segment profiles can shape communication; individual credit decisions follow independent eligibility rules. Document the model and keep human review of targeting rules.
#
# ### 4.6 Stable 5-year segments vs quarterly dynamic campaigns
#
# | Choice | Stable segments (5 years) | Quarterly campaigns |
# |---|---|---|
# | Features | Structural behaviour over long windows (usage mix, ratios averaged over 12+ months); avoid seasonal levels; normalise flows by exposure time | Recent 3-month activity, recency, change vs last quarter, campaign response history |
# | Granularity | Few, broad clusters (k = 3 to 4) | Finer clusters (e.g. k = 5 with the purchase-plus-cash-advance group), soft membership for boundary customers |
# | Fitting and labels | Freeze the base-period preprocessing and centroids; assign customers in later windows to the nearest frozen centroid | Refit each quarter; match new clusters to old ones by profile or centroid before naming (numbering by size can swap names when ranks change) |
# | Validation | Several historical windows: for the same trackable customers, report cluster sizes, profile drift and customer migration rates | Campaign uplift vs hold-out by segment. A move from an active to a low-activity cluster is flagged as declining activity; whether it predicts attrition needs outcome data |
#
# **Limitation.** The data is a single snapshot without dates or demographics. Subsampling stability (§2.1) only measures sensitivity to sample perturbation; it says nothing about stability over time.
# All business numbers are in-sample proxies under stated assumptions.
