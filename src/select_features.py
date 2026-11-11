"""Step 1 — feature selection (the only "tuning" step of sub_H).

Greedy forward selection over a fixed candidate pool, then greedy addition of pairwise
interactions. Scored by 5-fold GroupKFold-by-ZIP RMSE on the train months (May+June),
Ridge on raw target, geo config GEO_SELECT, fold seed 0.
Notebook 02_linear.ipynb, sections 6-7 (cells 29-34).

Output: config/features.json (FWD, INTER). Runtime ~5 min on Apple M3 Pro.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from common import (CONFIG, CORE, GEO_SELECT, SEED_SELECT, T, build_base_tables, geo_cols, inter_col, inter_stats,
                    make_model, rmse, with_geo, zip_folds)

_, _, _, tr, te = build_base_tables()

POOL = ["log_size", "log_rent", "log_income", "log_rti", "OwnerPercent", "owner_logit", "MedianHomeLengthofResidence",
        "AvgNumberofChildren", "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
        "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent", "rent_missing", "child_share",
        "children_per_adult", "gen_per_adult", "owner_x_rent", "owner_x_income", "log_rent_sq", "log_income_sq",
        "senior_sq", "tenure_sq", "adults_sq", "buffer", "disposable_k", "income_per_person_log", "rent_per_adult_log",
        "log_rent_vs_state", "log_income_vs_state", "OwnerPercent_vs_state", "AvgNumberofChildren_vs_state"] + geo_cols(GEO_SELECT)

GEO_CACHE = {}


def fold_data(a, b, cfg, seed, fi):
    A, B = tr.iloc[a], tr.iloc[b]
    key = (repr(sorted(cfg.items())), seed, fi)
    if key not in GEO_CACHE:
        gA, gB = with_geo(A, B, cfg)
        cols = geo_cols(cfg)
        GEO_CACHE[key] = (gA[cols], gB[cols])
    gA, gB = GEO_CACHE[key]
    return A.join(gA), B.join(gB)


def cv_new(feats, geo=GEO_SELECT, seed=SEED_SELECT, extra=None):
    """RMSE on held-out ZIPs (GroupKFold by ZIP), Ridge on raw target."""
    oof = np.zeros(len(tr))
    for fi, (a, b) in enumerate(zip_folds(tr, seed)):
        A, B = fold_data(a, b, geo, seed, fi)
        if extra is not None:
            A, B = A.join(extra.loc[A.index]), B.join(extra.loc[B.index])
        m = make_model(feats).fit(A[feats], A[T])
        oof[b] = m.predict(B[feats])
    return rmse(tr[T], np.clip(oof, 0, 100))


# --- forward selection: add the best feature while RMSE improves by > 0.01 ---
t0 = time.time(); FWD, best = [], np.inf
while True:
    scores = {f: cv_new(FWD + [f]) for f in POOL if f not in FWD}
    f, s = min(scores.items(), key=lambda x: x[1])
    if s > best - 0.01:
        break
    FWD.append(f); best = s
    print(f"  + {f:32s} {s:.3f}", flush=True)
print(f"forward: {len(FWD)} features, RMSE {best:.3f}, {time.time() - t0:.0f}s")

# --- interactions: rank all 45 pairs of CORE, greedily add top-12 while RMSE improves by > 0.01 ---
mu, sd = inter_stats(tr)
base_fwd = cv_new(FWD)
gains = []
for a, b in itertools.combinations(CORE, 2):
    n_ = f"{a}*{b}"
    ex = pd.DataFrame({n_: inter_col(tr, a, b, mu, sd)}, index=tr.index)
    gains.append((n_, base_fwd - cv_new(FWD + [n_], extra=ex)))
gains = pd.DataFrame(gains, columns=["inter", "gain"]).sort_values("gain", ascending=False)

INTER, cur, ex_all = [], base_fwd, pd.DataFrame(index=tr.index)
for n_ in gains["inter"].head(12):
    a, b = n_.split("*"); ex_all[n_] = inter_col(tr, a, b, mu, sd)
    s = cv_new(FWD + INTER + [n_], extra=ex_all)
    if s < cur - 0.01:
        INTER.append(n_); cur = s
print(f"interactions: {INTER}, RMSE {cur:.3f}")

CONFIG.mkdir(exist_ok=True)
out = CONFIG / "features.json"
json.dump({"FWD": FWD, "INTER": INTER}, open(out, "w"), indent=2)
print("saved", out)
