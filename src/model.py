"""sub_H model: tables, Seen-ZIP drift, New-ZIP model (notebook 02, sections 8-10, 12-13, 15)."""
import json

import numpy as np
import pandas as pd

from common import (CONFIG, GEO_FINAL, GEO_SELECT, NB_COLS, NB_K, RESID_K, RESID_W, SEED_RESID_OOF, T, TRANS,
                    build_base_tables, geo_cols, inter_col, inter_stats, knn_by_zip, make_model, nb_features,
                    with_geo, zip_folds)


def load_features():
    cfg = json.load(open(CONFIG / "features.json"))
    return cfg["FWD"], cfg["INTER"]


def build_tables():
    """Returns everything needed to train and predict: tables, final feature list, spline columns."""
    train, test, sub, tr, te = build_base_tables()
    FWD, INTER = load_features()

    mu, sd = inter_stats(tr)
    for d in (tr, te):
        for n_ in INTER:
            a, b = n_.split("*"); d[n_] = inter_col(d, a, b, mu, sd)

    pool = pd.concat([tr, te])
    for d in (tr, te):
        nb = nb_features(d, pool, NB_COLS, NB_K)
        for c in nb:
            d[c] = nb[c]

    NOGEO = [f for f in FWD if f not in geo_cols(GEO_SELECT)] + INTER
    NB = [f"nb{NB_K}_{c}" for c in NB_COLS]
    BEST = NOGEO + geo_cols(GEO_FINAL) + NB
    # splines only on continuous features (binary flags break SplineTransformer)
    SPL = [c for c in BEST if c.startswith(("te_", "knn")) or tr[c].nunique() > 10]

    is_seen = te["Zip"].isin(train["Zip"]).values
    return dict(train=train, test=test, sub=sub, tr=tr, te=te, BEST=BEST, SPL=SPL, is_seen=is_seen)


# ---------------------------------------------------------------------------
# Seen ZIPs: June target + hierarchical drift State -> Metro -> zip3 estimated on May -> June
# ---------------------------------------------------------------------------
def may_june_drift(tr):
    may = tr[tr.Month.dt.month == 5].set_index("Zip")
    jun = tr[tr.Month.dt.month == 6].set_index("Zip").loc[may.index]
    dy = jun[T] - may[T]
    may["zip3"] = may.index.str[:3]
    return may, dy


def seen_hier_drift(tr, te, is_seen):
    may, dy = may_june_drift(tr)
    jun_all = tr[tr.Month == tr.Month.max()].set_index("Zip")[T]
    m_ = may.assign(d=dy.values)
    st_ = m_.groupby("State")["d"].agg(["mean", "count"]); p = dy.mean()
    enc_s = (st_["mean"] * st_["count"] + p * 20) / (st_["count"] + 20)          # State, shrink 20 to global mean
    r = m_["d"] - m_["State"].map(enc_s)
    te_seen = te[is_seen]
    out = te_seen["State"].map(enc_s).fillna(p).values
    for col in ("Metro", "zip3"):                                                 # residual drift, shrink 50 to 0
        g = pd.DataFrame({"r": r.values, "s": m_[col].values}).groupby("s")["r"].agg(["mean", "count"])
        enc = g["mean"] * g["count"] / (g["count"] + 50)
        out = out + te_seen[col].map(enc).fillna(0).values
        r = r - m_[col].map(enc).fillna(0)
    return te_seen["Zip"].map(jun_all).values + out


# ---------------------------------------------------------------------------
# New ZIPs: train on July feature rows of Seen ZIPs with pseudo-label (June target + mean May->June drift)
# ---------------------------------------------------------------------------
def july_pseudo_labelled(tr, te, is_seen):
    _, dy = may_june_drift(tr)
    jun_t = tr[tr.Month == tr.Month.max()].set_index("Zip")[T]
    jul = te[is_seen].copy().reset_index(drop=True)
    jul[T] = jul["Zip"].map(jun_t).values + dy.mean()
    return jul


def asin_splines(A, B, BEST, SPL, resid=(RESID_K, RESID_W)):
    """Ridge + cubic splines on arcsin(sqrt(y/100)); geo features OOF; optional spatial residual correction."""
    fA, iA = TRANS["asin"]
    A, B = with_geo(A, B, GEO_FINAL)
    fit = lambda X: make_model(BEST, splines=SPL).fit(X[BEST], fA(X[T]))
    p = iA(fit(A).predict(B[BEST]))
    if resid:
        k, w = resid; rA = np.zeros(len(A))
        for a2, b2 in zip_folds(A, seed=SEED_RESID_OOF):
            rA[b2] = A[T].values[b2] - iA(fit(A.iloc[a2]).predict(A.iloc[b2][BEST]))
        ref = A.assign(r=rA)
        ref = ref[ref.Month == ref.Month.max()].drop_duplicates("Zip")
        v, _ = knn_by_zip(ref.zipn.values.astype(float), ref.r.values, B.zipn.values.astype(float), k)
        p = p + w * v.mean(1)
    return p
