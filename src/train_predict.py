"""Step 2 — train the final model and write the submission (notebook 02, cells 53 and 68).

Seen ZIPs: June target + hierarchical May->June drift.
New ZIPs:  Ridge + splines (arcsin target) trained on July rows of Seen ZIPs with pseudo-labels,
           + spatial residual correction (k=20, w=0.5).
Output: output/sub_H_july_trained_resid.csv. Runtime ~1 min.
"""
import numpy as np

from common import OUT, T
from model import asin_splines, build_tables, july_pseudo_labelled, seen_hier_drift

D = build_tables()
tr, te, sub, is_seen = D["tr"], D["te"], D["sub"], D["is_seen"]
print(f"features: {len(D['BEST'])} (splines on {len(D['SPL'])}); test rows: {len(te)}, seen {is_seen.sum()}, new {(~is_seen).sum()}")

seen_pred = seen_hier_drift(tr, te, is_seen)
jul = july_pseudo_labelled(tr, te, is_seen)
new_pred = asin_splines(jul, te[~is_seen].copy(), D["BEST"], D["SPL"])

pred = np.zeros(len(te))
pred[~is_seen] = new_pred
pred[is_seen] = seen_pred
out = sub[["Zip", "State"]].copy()
assert (out["Zip"].values == te["Zip"].values).all()
out[T] = np.clip(pred, 0, 100)

OUT.mkdir(exist_ok=True)
path = OUT / "sub_H_july_trained_resid.csv"
out.to_csv(path, index=False)
print("saved", path)
print(out[T].describe().round(2))
