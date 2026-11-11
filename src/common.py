"""Shared preprocessing and model utilities for the sub_H pipeline.

Only the three official competition files are read:
    data/train.csv, data/test.csv, data/sample_submission.csv
No external data, APIs or pretrained models are used.

Code is a faithful extraction of notebook 02_linear.ipynb (sections 0-1, 3, 6-10, 12-13, 15).
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__import__("os").environ.get("DATA_DIR", ROOT / "data"))
OUT = ROOT / "output"
CONFIG = ROOT / "config"
T = "AffordabilityPercentageTrue"

# ---------------------------------------------------------------------------
# Hyper-parameters (all fixed; see README "Hyper-parameters")
# ---------------------------------------------------------------------------
RIDGE_ALPHAS = np.logspace(-3, 4, 30)       # RidgeCV grid (efficient LOO inside RidgeCV)
SPLINE_KNOTS, SPLINE_DEGREE = 6, 3
GEO_SELECT = {"te": ["State", "Metro", "zip3"], "knn": [5]}                       # geo config used during feature selection
GEO_FINAL = {"te": ["State", "Metro", "zip3"], "smooth": 5, "knn": [3, 10, 30], "knn_w": True}  # final geo config
NB_K = 10                                     # neighbour-context features: mean of 10 nearest ZIPs
RESID_K, RESID_W = 20, 0.5                    # spatial residual correction
SEED_GEO_OOF, SEED_RESID_OOF = 101, 303       # inner fold seeds
SEED_SELECT = 0                               # outer fold seed used for forward selection

DEMO = ["SampleSize", "AnnualMedianHouseholdIncome", "OwnerPercent", "MedianHomeLengthofResidence",
        "AvgNumberofChildren", "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
        "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent"]
CORE = ["log_rent", "log_income", "OwnerPercent", "AvgNumberofChildren", "AvgNumberofAdults", "SeniorAdultinHouseholdPercent",
        "MedianHomeLengthofResidence", "YoungAdultinHouseholdPercent", "AvgGenerationsinHousehold", "log_size"]
NB_COLS = ["log_rent", "log_income", "OwnerPercent", "AvgNumberofChildren", "SeniorAdultinHouseholdPercent", "AvgNumberofAdults",
           "MedianHomeLengthofResidence", "YoungAdultinHouseholdPercent", "log_size", "AvgGenerationsinHousehold",
           "AvgHouseholdSize", "rent_missing"]


# ---------------------------------------------------------------------------
# Data loading + imputation (notebook 02, section 0)
# ---------------------------------------------------------------------------
def load_raw():
    train = pd.read_csv(DATA / "train.csv", dtype={"Zip": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"Zip": str})
    sub = pd.read_csv(DATA / "sample_submission.csv", dtype={"Zip": str})
    for d in (train, test):
        d["Month"] = pd.to_datetime(d["Month"], format="%m/%d/%Y")
    return train, test, sub


def demo_X(df):
    x = df[DEMO].copy()
    x["SampleSize"] = np.log(x["SampleSize"])
    x["AnnualMedianHouseholdIncome"] = np.log(x["AnnualMedianHouseholdIncome"])
    return x


def geo_mean(known, df, col="log_rent"):
    est = df["zip3"].map(known.groupby("zip3")[col].mean())
    est = est.fillna(df["Metro"].map(known.groupby("Metro")[col].mean()))
    return est.fillna(df["State"].map(known.groupby("State")[col].mean()))


def preprocess(train, test):
    """Missing-value handling and basic transforms on train+test features (no target used)."""
    full = pd.concat([train.assign(src="train"), test.assign(src="test")], ignore_index=True)
    for n in (1, 2, 3, 4):
        full[f"zip{n}"] = full["Zip"].str[:n]
    full["zipn"] = full["Zip"].astype(int)
    full["Metro"] = full["Metro"].fillna("Non-metro")
    full["City"] = full["City"].fillna("Miami")
    full["rent_missing"] = full["RentIndex"].isna().astype(int)

    # RentIndex: 1) same ZIP, nearest month; 2) geo hierarchy zip3 -> Metro -> State + Ridge on demographics
    full["log_rent"] = np.log(full["RentIndex"])
    full = full.sort_values(["Zip", "Month"])
    full["log_rent"] = full.groupby("Zip")["log_rent"].transform(lambda s: s.ffill().bfill())
    full = full.sort_index()
    known = full[full["log_rent"].notna()]
    rent_ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(demo_X(known), known["log_rent"] - geo_mean(known, known))
    m = full["log_rent"].isna()
    full.loc[m, "log_rent"] = geo_mean(known, full[m]) + rent_ridge.predict(demo_X(full[m]))

    full["RentIndex_f"] = np.exp(full["log_rent"])
    full["log_income"] = np.log(full["AnnualMedianHouseholdIncome"])
    full["log_rti"] = full["log_rent"] + np.log(12) - full["log_income"]
    full["rti"] = np.exp(full["log_rti"])
    full["log_size"] = np.log(full["SampleSize"])

    tr = full[full["src"] == "train"].reset_index(drop=True)
    te = full[full["src"] == "test"].reset_index(drop=True)
    return tr, te


# ---------------------------------------------------------------------------
# Engineered features (notebook 02, sections 3 and 6)
# ---------------------------------------------------------------------------
def add_ratio_features(d):
    d["child_share"] = d.AvgNumberofChildren / d.AvgHouseholdSize
    d["children_per_adult"] = d.AvgNumberofChildren / d.AvgNumberofAdults
    d["gen_per_adult"] = d.AvgGenerationsinHousehold / d.AvgNumberofAdults


def add_engineered(d, ref):
    d["owner_logit"] = np.log(d.OwnerPercent / (100 - d.OwnerPercent))
    d["owner_x_rent"] = d.OwnerPercent / 100 * d.log_rent
    d["owner_x_income"] = d.OwnerPercent / 100 * d.log_income
    d["log_rent_sq"] = (d.log_rent - 7.5) ** 2
    d["log_income_sq"] = (d.log_income - 11.4) ** 2
    d["senior_sq"] = (d.SeniorAdultinHouseholdPercent - 10) ** 2
    d["tenure_sq"] = (d.MedianHomeLengthofResidence - 8) ** 2
    d["adults_sq"] = (d.AvgNumberofAdults - 1.8) ** 2
    d["buffer"] = 0.30 - d.rti
    d["disposable_k"] = (d.AnnualMedianHouseholdIncome / 12 - d.RentIndex_f) / 1000
    d["income_per_person_log"] = d.log_income - np.log(d.AvgHouseholdSize)
    d["rent_per_adult_log"] = d.log_rent - np.log(d.AvgNumberofAdults)
    for c in ["log_rent", "log_income", "OwnerPercent", "AvgNumberofChildren"]:
        d[f"{c}_vs_state"] = d[c] - d.State.map(ref[c])


def build_base_tables():
    """train/test after imputation + all engineered features used in the candidate pool."""
    train, test, sub = load_raw()
    tr, te = preprocess(train, test)
    for d in (tr, te):
        add_ratio_features(d)
    state_ref = pd.concat([tr, te]).groupby("State")[["log_rent", "log_income", "OwnerPercent", "AvgNumberofChildren"]].mean()
    for d in (tr, te):
        add_engineered(d, state_ref)
    return train, test, sub, tr, te


def inter_stats(tr):
    return tr[CORE].mean(), tr[CORE].std()


def inter_col(d, a, b, mu, sd):
    return ((d[a] - mu[a]) / sd[a]) * ((d[b] - mu[b]) / sd[b])


# ---------------------------------------------------------------------------
# Folds, metric, geo encoders (notebook 02, section 1)
# ---------------------------------------------------------------------------
def zip_folds(df, seed=0, k=5):
    z = df["Zip"].unique().copy()
    np.random.RandomState(seed).shuffle(z)
    fold_of = pd.Series(np.arange(len(z)) % k, index=z)
    f = df["Zip"].map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def target_encode(A, B, col, smooth=20):
    g = A.groupby(col)[T].agg(["mean", "count"])
    prior = A[T].mean()
    enc = (g["mean"] * g["count"] + prior * smooth) / (g["count"] + smooth)
    return B[col].map(enc).fillna(prior).values


def knn_by_zip(ref_zip, ref_val, query_zip, k):
    """k nearest ZIPs by ZIP-code number (adjacent numbers ~ adjacent geography)."""
    o = np.argsort(ref_zip); rz, rv = ref_zip[o], ref_val[o]; n = len(rz)
    pos = np.searchsorted(rz, query_zip)
    idx = np.clip(pos[:, None] + np.arange(-k, k)[None, :], 0, n - 1)
    d = np.abs(rz[idx] - query_zip[:, None]).astype(float)
    dup = np.zeros_like(d, bool); dup[:, 1:] = idx[:, 1:] == idx[:, :-1]; d[dup] = np.inf
    sel = np.argsort(d, axis=1)[:, :k]
    return np.take_along_axis(rv[idx], sel, 1), np.take_along_axis(d, sel, 1)


def geo_features(A, B, cfg):
    out = {}
    for c in cfg.get("te", []):
        out[f"te_{c}"] = target_encode(A, B, c, cfg.get("smooth", 20))
    ref = A[A["Month"] == A["Month"].max()].drop_duplicates("Zip")
    for k in cfg.get("knn", []):
        v, d = knn_by_zip(ref["zipn"].values.astype(float), ref[T].values, B["zipn"].values.astype(float), k)
        out[f"knn{k}"] = v.mean(1)
        if cfg.get("knn_w"):
            w = 1 / (1 + d); out[f"knnw{k}"] = (v * w).sum(1) / w.sum(1)
    return pd.DataFrame(out, index=B.index)


def with_geo(A, B, cfg):
    """Target-derived geo features: out-of-fold inside A, full-A for B."""
    parts = [geo_features(A.iloc[a], A.iloc[b], cfg) for a, b in zip_folds(A, seed=SEED_GEO_OOF)]
    return A.join(pd.concat(parts).loc[A.index]), B.join(geo_features(A, B, cfg))


def geo_cols(cfg):
    cols = [f"te_{c}" for c in cfg.get("te", [])] + [f"knn{k}" for k in cfg.get("knn", [])]
    return cols + ([f"knnw{k}" for k in cfg.get("knn", [])] if cfg.get("knn_w") else [])


def nb_features(df, pool, cols, k=NB_K):
    """Mean feature values of the k nearest other ZIPs (features only, no target)."""
    allz = pool.drop_duplicates("Zip", keep="last")
    out = {}
    for c in cols:
        v, d = knn_by_zip(allz["zipn"].values.astype(float), allz[c].values, df["zipn"].values.astype(float), k + 1)
        out[f"nb{k}_{c}"] = np.nanmean(np.where(d == 0, np.nan, v), 1)  # exclude the ZIP itself
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Model and target transforms (notebook 02, section 1 / 9)
# ---------------------------------------------------------------------------
def make_model(feats, splines=None, n_knots=SPLINE_KNOTS):
    steps = []
    if splines:
        steps.append(ColumnTransformer([("spl", SplineTransformer(n_knots=n_knots, degree=SPLINE_DEGREE), splines)],
                                       remainder="passthrough"))
    steps += [StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)]
    return make_pipeline(*steps)


TRANS = {
    "raw": (lambda y: np.asarray(y, float), lambda z: z),
    "asin": (lambda y: np.arcsin(np.sqrt(np.clip(np.asarray(y, float), 0, 100) / 100)),
             lambda z: 100 * np.sin(np.clip(z, 0, np.pi / 2)) ** 2),
}
