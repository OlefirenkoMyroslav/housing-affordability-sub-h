# sub_H — reproduction pipeline

**Predicting Housing Affordability in the U.S.** (Kaggle, Melissa Data).
Submission `sub_H_july_trained_resid.csv`: public LB **10.447** (submitted 2026-10-29).

**Data used:** only the official competition files `train.csv`, `test.csv` and `sample_submission.csv`.
No external datasets, APIs, pretrained models or hand-labelling.
`grep read_csv src/*.py` lists every file the pipeline reads.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && cp /path/to/{train,test,sample_submission}.csv data/
bash run_all.sh            # or: PYTHON=/path/to/python DATA_DIR=/path/to/data bash run_all.sh
```

| Step | Script | Output | Time |
|---|---|---|---|
| 1. Feature selection (the only tuning step) | `src/select_features.py` | `config/features.json` | ~5 min |
| 2. Train + inference + submission | `src/train_predict.py` | `output/sub_H_july_trained_resid.csv` | ~10 s |
| 3. Validation scores (optional) | `src/validate.py` | stdout | ~1 min |

`config/features.json` is committed, so step 2 also runs on its own. Step 1 regenerates the same file.

**Verified:** step 1 reproduces the committed feature list exactly. Step 2 reproduces the submitted CSV byte-for-byte (max abs diff 0.0).
The whole pipeline is deterministic: every split uses a fixed seed, and Ridge has no randomness.

### Compute environment
- Apple M3 Pro (12 CPU), 18 GB RAM, macOS 27.2. CPU only, single process, peak RAM < 2 GB.
- Python 3.13.5; numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0.

## Methodology

### Problem setup
- Train: May and June 2026, 6,830 ZIPs. Test: July 2026, 8,534 ZIPs.
  - **Seen** ZIPs (6,830) appear in train.
  - **New** ZIPs (1,704) are a geographic holdout.
- Metric: `0.2·RMSE_Seen + 0.8·RMSE_New`.
- The target moves very little month to month (May vs June corr 0.999).
- July demographic features shift sharply compared with June (`SeniorAdult%` +10%, `YoungAdult%` +11%). Models trained on May/June therefore under-predict July by 1.5–3 p.p.

### Seen ZIPs
**Prediction = June target + hierarchical drift.** The drift is the May→June target change, estimated in three nested levels:
1. **State** mean, shrunk toward the global mean with weight 20.
2. **Metro** mean of the remaining residual, shrunk toward 0 with weight 50.
3. **zip3** mean of the remaining residual, shrunk toward 0 with weight 50.

Validation (June predicted from May, 5-fold): RMSE 1.155, vs 1.208 for a global mean drift.

### New ZIPs
1. **Training rows.** July feature rows of Seen ZIPs.
   - Pseudo-label = June target + mean May→June drift (−0.33). It comes from train labels only.
   - Training on July features removes the month-shift bias.
2. **Preprocessing.**
   - Missing `RentIndex`: same ZIP's nearest month first. Otherwise a zip3 → Metro → State mean plus a Ridge on demographics.
   - Missing `Metro` → "Non-metro"; missing `City` → "Miami".
   - Log of rent, income and sample size; `rent_missing` flag.
3. **Features (42).**
   - Engineered: rent/income ratio, squared terms, owner × income, deviation from the state mean, per-adult ratios.
   - 3 tenure interactions: `log_income × tenure`, `log_rent × tenure`, `log_income × generations`.
   - Target encoding for State/Metro/zip3 (smoothing 5).
   - kNN target by ZIP-code number: k = 3/10/30, plain and distance-weighted.
   - Mean demographics of the 10 nearest ZIPs (features only, no target).
   - All target-derived features are computed out-of-fold (inner 5-fold by ZIP, seed 101).
4. **Model.**
   - Ridge (`RidgeCV`, α ∈ logspace(−3, 4, 30)) after `StandardScaler`.
   - Cubic B-spline expansion (6 knots) on 41 continuous features.
   - Target transform `arcsin(√(y/100))`, inverted after prediction, clipped to [0, 100].
5. **Spatial residual correction.**
   - Compute out-of-fold training residuals (inner 5-fold, seed 303).
   - Add 0.5 × mean residual of the 20 nearest training ZIPs.

### Feature selection (step 1)
1. Greedy forward selection from a pool of 36 candidates.
   - Scored by 5-fold GroupKFold-by-ZIP RMSE on May+June, Ridge on raw target, seed 0.
   - A feature is added while it improves RMSE by > 0.01.
2. All 45 pairwise interactions of 10 core features are ranked. The top 12 are added greedily under the same > 0.01 rule.

Spline knots, smoothing, kNN scales, k/w of the residual correction and the target transform were chosen by comparing a small grid of options on the same CV. The grids and results are in notebook `02_linear.ipynb`, sections 8–10. Final values are fixed in `src/common.py`.

### Validation
| | Score |
|---|---|
| Seen, June predicted from May | RMSE 1.155 |
| New, July-CV (train July rows of 80% of Seen ZIPs, predict the held-out 20%) | RMSE 12.22, bias −0.05 |
| Public LB | 10.447 |

July-CV includes ~1.2 RMSE of pseudo-label noise.
