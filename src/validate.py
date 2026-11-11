"""Step 3 (optional) — validation scores reported in the write-up (notebook 02, cells 50 and 63).

* Seen ZIPs: predict June from May, 5-fold KFold over ZIPs.
* New ZIPs, "July-CV": train on July rows of ZIP group A (pseudo-label), predict July rows of held-out ZIPs B,
  score vs pseudo-label (June target + mean drift; label noise ~1.2 RMSE). 5 folds, seed 0.
Runtime ~1 min.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import T, rmse
from model import asin_splines, build_tables, july_pseudo_labelled, may_june_drift

D = build_tables()
tr, te, is_seen, BEST, SPL = D["tr"], D["te"], D["is_seen"], D["BEST"], D["SPL"]

# --- Seen: June from May ---
may, dy = may_june_drift(tr)
jun = tr[tr.Month.dt.month == 6].set_index("Zip").loc[may.index]
y0, y1 = may[T], jun[T]
kf = KFold(5, shuffle=True, random_state=0)


def group_drift(a, b, col, smooth, resid=None):
    s = may[col].values; r = dy.values[a] if resid is None else resid
    g = pd.DataFrame({"d": r, "s": s[a]}).groupby("s")["d"].agg(["mean", "count"])
    prior = r.mean() if resid is None else 0.0
    enc = (g["mean"] * g["count"] + prior * smooth) / (g["count"] + smooth)
    return pd.Series(s[b]).map(enc).fillna(prior).values, pd.Series(s[a]).map(enc).fillna(prior).values


def hier_drift(a, b):
    pb, pa = group_drift(a, b, "State", 20); r = dy.values[a] - pa
    for col in ("Metro", "zip3"):
        qb, qa = group_drift(a, b, col, 50, resid=r); pb, r = pb + qb, r - qa
    return pb


o = np.zeros(len(y0))
for a, b in kf.split(y0):
    o[b] = y0.values[b] + hier_drift(a, b)
print(f"Seen RMSE (June from May, hierarchical drift): {rmse(y1, o):.4f}   [notebook: 1.155]")

# --- New: July-CV ---
jul = july_pseudo_labelled(tr, te, is_seen)
zips = jul["Zip"].values.copy(); np.random.RandomState(0).shuffle(zips)
fold_of = pd.Series(np.arange(len(zips)) % 5, index=zips)
oof = np.zeros(len(jul))
for k in range(5):
    a_zips = set(fold_of.index[fold_of != k])
    B = jul[jul["Zip"].map(fold_of) == k]
    oof[B.index] = asin_splines(jul[jul.Zip.isin(a_zips)], B, BEST, SPL)
oof = np.clip(oof, 0, 100)
print(f"July-CV RMSE New: {rmse(jul[T], oof):.3f}, bias {np.mean(oof - jul[T]):.2f}   [notebook: 12.224, -0.05]")
