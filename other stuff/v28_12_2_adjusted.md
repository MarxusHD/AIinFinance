```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, confusion_matrix, classification_report
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# Project: Next-year financial distress prediction (firm-year panel)
# Data Science Lifecycle focus in this cell:
#   - Setup & configuration (temporal split conventions, leakage control parameters)
#   - Panel integrity (types, deduplication, stable firm identifier)
#   - Define label_year used consistently for splitting and target alignment
# =============================================================================

# ----------------------------
# Configuration (split & preprocessing parameters)
# ----------------------------
FILE_NAME = "data.csv"

TRAIN_CUTOFF_LABEL_YEAR = 2022   # label_year <= cutoff → train/val pool; after cutoff → test
VAL_YEARS = 1                    # last N years within the pool are validation
N_SPLITS_TIME_CV = 4             # rolling time-based folds for sanity checks

WINSOR_LOWER_Q = 0.01            # winsorization lower quantile (train-only)
WINSOR_UPPER_Q = 0.99            # winsorization upper quantile (train-only)

REQUIRED_KEYS = ["gvkey", "fyear"]
TARGET_COL = "target_next_year_distress"


# ----------------------------
# Utilities (robust numeric ops for ratios)
# ----------------------------

def to_float_numpy(x) -> np.ndarray:
    """Convert series/array-like to float numpy array, coercing non-numeric to NaN."""
    s = pd.to_numeric(x, errors="coerce")
    return s.to_numpy(dtype=float) if hasattr(s, "to_numpy") else np.asarray(s, dtype=float)
def safe_divide(a, b) -> np.ndarray:
    """Elementwise divide a/b with NaN when division is invalid (0 or non-finite)."""
    a = to_float_numpy(a)
    b = to_float_numpy(b)
    out = np.full_like(a, np.nan, dtype=float)
    np.divide(a, b, out=out, where=(b != 0) & np.isfinite(a) & np.isfinite(b))
    return out

def rolling_year_folds(
    df_in: pd.DataFrame, year_col: str = "label_year", n_splits: int = 4, min_train_years: int = 3
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Create expanding-window time folds:
      train years: first (min_train_years + k) years
      val year:    next year
    Returns: list of (train_idx, val_idx, train_years, val_year)
    """
    years_sorted = np.sort(df_in[year_col].dropna().unique())
    if len(years_sorted) <= min_train_years:
        return []
    n_splits = min(n_splits, len(years_sorted) - min_train_years)

    folds_out = []
    for k in range(n_splits):
        train_years = years_sorted[: min_train_years + k]
        val_year = int(years_sorted[min_train_years + k])

        train_idx = df_in.index[df_in[year_col].isin(train_years)].to_numpy()
        val_idx = df_in.index[df_in[year_col] == val_year].to_numpy()
        folds_out.append((train_idx, val_idx, train_years, val_year))

    return folds_out


# =============================================================================
# 1) Data acquisition & panel hygiene
# =============================================================================
df = pd.read_csv(FILE_NAME, low_memory=False)

# Convert datadate if present
if "datadate" in df.columns:
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

# Create stable firm id + de-duplicate firm-year (keep last record)
df["firm_id"] = df["gvkey"]
df = (
    df.sort_values(["firm_id", "fyear"])
      .drop_duplicates(subset=["firm_id", "fyear"], keep="last")
      .reset_index(drop=True)
)

# Label year: predict distress in the next fiscal year
df["label_year"] = df["fyear"] + 1

# =============================================================================
# 2) Split scaffolding (define train/val pool years via label_year)
# =============================================================================
pool_mask = df["label_year"] <= TRAIN_CUTOFF_LABEL_YEAR
pool_years = np.sort(df.loc[pool_mask, "label_year"].dropna().unique())
val_years = pool_years[-VAL_YEARS:] if len(pool_years) else np.array([], dtype=int)

# This mask is ONLY used for imputations (train-only information)
train_mask_for_imputation = pool_mask & (~df["label_year"].isin(val_years))

```


```python
# =============================================================================
# Data Cleaning & Missing-Data Handling (leakage-aware)
# Purpose:
#   - Quantify missingness and distribution properties before intervention
#   - Preserve informative missingness via miss_* indicators
#   - Impute financial statement inputs using TRAIN-only information:
#       (1) within-firm past values (lag-1) where economically sensible
#       (2) peer medians by year×size_decile on size-scaled ratios (ratio space)
#       (3) KNN imputation (TRAIN-fit) for selected core balance-sheet items
#   - Re-run EDA after imputation to audit how strongly imputations alter the data
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

RAW_INPUTS_FOR_FE = [
    "aco","act","ao","aoloch","ap","apalch","aqc","at","caps","capx","ceq","che","chech","csho","cstk","cstke",
    "datadate","dlc","dlcch","dltis","dltr","dltt","do","dp","dpc","dv","dvc","dvp","dvt","esubc","exre",
    "fiao","fincf","fopo","fyear","gvkey","ib","ibadj","ibc","intan","invch","invt","ismod","ivaco","ivaeq",
    "ivao","ivch","ivncf","ivstch","lco","lct","lt","mibt","mkvalt","niadj","nopi","oancf","oibdp","ppent",
    "prcc_c","prcc_f","prstkc","pstk","pstkn","pstkr","re","recch","rect","seq","siv","spi","sppe","sppiv",
    "sstk","tstk","txach","txbcof","txdc","txditc","txp","txt","xi","xido","xidoc","xint",
    # optional identifiers present in many extracts:
    "conm","consol","datafmt","indfmt",
]
raw = [c for c in RAW_INPUTS_FOR_FE if c in df.columns]

# ---------------------------------------------------------------------------
# 3.0 Ensure keys exist + types
# ---------------------------------------------------------------------------
if "firm_id" not in df.columns:
    if "gvkey" in df.columns:
        df["firm_id"] = df["gvkey"]
    else:
        raise ValueError("Need either firm_id or gvkey in df to run panel imputations.")

if "fyear" in df.columns:
    df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce")

# ---------------------------------------------------------------------------
# 3.1 Drop rows with missing critical identifiers (do not impute these)
# ---------------------------------------------------------------------------
NON_IMPUTE_DROP = [c for c in ["gvkey", "datadate", "fyear", "conm", "datafmt", "indfmt", "consol"] if c in df.columns]
if NON_IMPUTE_DROP:
    before_n = df.shape[0]
    df = df.dropna(subset=NON_IMPUTE_DROP).copy()
    after_n = df.shape[0]
    if after_n < before_n:
        print(f"[INFO] Dropped {before_n - after_n:,} rows due to missing non-imputable ID/meta fields: {NON_IMPUTE_DROP}")

# Ensure train mask aligns after drops
if isinstance(train_mask_for_imputation, pd.Series):
    train_mask_for_imputation = train_mask_for_imputation.reindex(df.index).fillna(False).astype(bool)

# Rebuild raw after potential drop
raw = [c for c in RAW_INPUTS_FOR_FE if c in df.columns]

# ---------------------------------------------------------------------------
# 3.2 EDA BEFORE imputation (missingness + distribution snapshot)
# ---------------------------------------------------------------------------
df_raw_pre = df[raw].copy(deep=True)

pre_miss = pd.DataFrame(
    {
        "col": raw,
        "n": [int(df_raw_pre[c].shape[0]) for c in raw],
        "n_na_pre": [int(df_raw_pre[c].isna().sum()) for c in raw],
        "pct_na_pre": [float(df_raw_pre[c].isna().mean() * 100.0) for c in raw],
        "train_n": [int(train_mask_for_imputation.sum()) for _ in raw],
        "train_pct_na_pre": [
            float(df_raw_pre.loc[train_mask_for_imputation, c].isna().mean() * 100.0) for c in raw
        ],
    }
).sort_values("pct_na_pre", ascending=False)

print("\n=== EDA (BEFORE imputation): Missingness on raw inputs ===")
print(pre_miss.round(4).head(50))

# Numeric distribution summary (coerce non-numeric to NaN)
if raw:
    x_pre = df_raw_pre[raw].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    q_pre = x_pre.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    pre_dist = pd.DataFrame(
        {
            "n_nonmiss_pre": x_pre.notna().sum(),
            "mean_pre": x_pre.mean(),
            "std_pre": x_pre.std(ddof=0),
            "min_pre": x_pre.min(),
            "p01_pre": q_pre[0.01],
            "p05_pre": q_pre[0.05],
            "p25_pre": q_pre[0.25],
            "p50_pre": q_pre[0.50],
            "p75_pre": q_pre[0.75],
            "p95_pre": q_pre[0.95],
            "p99_pre": q_pre[0.99],
            "max_pre": x_pre.max(),
        }
    )
    print("\n=== EDA (BEFORE imputation): Distribution summary (raw inputs) ===")
    print(pre_dist.round(4).sort_values("n_nonmiss_pre", ascending=True).head(50))

# ---------------------------------------------------------------------------
# 3.3 Missingness flags (ALWAYS create before imputations)
# ---------------------------------------------------------------------------
for c in raw:
    df[f"miss_{c}"] = df[c].isna().astype("int8")

# Helper: originally-observed indicator
def _obs(col: str) -> pd.Series:
    m = f"miss_{col}"
    if m in df.columns:
        return (df[m] == 0)
    return df[col].notna()

# ---------------------------------------------------------------------------
# 3.4 Grouping key for peer-based imputations (no industry used)
#   - Use TRAIN-derived size_decile to avoid mixing microcaps with mega-caps
# ---------------------------------------------------------------------------
df = df.sort_values(["firm_id", "fyear"]).copy()

# TRAIN-derived size deciles on log(at) using observed TRAIN values only
if "at" in df.columns:
    at_train = pd.to_numeric(df.loc[train_mask_for_imputation, "at"], errors="coerce")
    log_at_train = np.log(at_train.where(at_train > 0))
    qs = np.linspace(0, 1, 11)
    edges = np.unique(np.nanquantile(log_at_train.dropna(), qs))
    if edges.size >= 3:
        log_at_all = np.log(pd.to_numeric(df["at"], errors="coerce").where(pd.to_numeric(df["at"], errors="coerce") > 0))
        df["size_decile"] = pd.cut(log_at_all, bins=edges, include_lowest=True, labels=False)
    else:
        df["size_decile"] = np.nan
else:
    df["size_decile"] = np.nan

group_cols = ["fyear", "size_decile"]

# ---------------------------------------------------------------------------
# 3.5 Step 1: Construct / reconcile FIRST (no leakage: contemporaneous or lag only)
# ---------------------------------------------------------------------------
# 3.5.1 Mild lag-1 fill for prices, then mkvalt = prcc_f * csho if missing
for px in ["prcc_f", "prcc_c"]:
    if px in df.columns:
        df[px] = df[px].where(df[px].notna(), df.groupby("firm_id")[px].shift(1))

if all(c in df.columns for c in ["mkvalt", "prcc_f", "csho"]):
    mkvalt_miss = df["mkvalt"].isna()
    mkvalt_calc = pd.to_numeric(df["prcc_f"], errors="coerce") * pd.to_numeric(df["csho"], errors="coerce")
    df.loc[mkvalt_miss & mkvalt_calc.notna(), "mkvalt"] = mkvalt_calc.loc[mkvalt_miss & mkvalt_calc.notna()]

# 3.5.2 Reconstruct change variables from level differences (fill only if change var is missing)
def _fill_change_from_levels(change_col, level_col):
    if change_col in df.columns and level_col in df.columns:
        miss = df[change_col].isna()
        lvl = pd.to_numeric(df[level_col], errors="coerce")
        lag_lvl = pd.to_numeric(df.groupby("firm_id")[level_col].shift(1), errors="coerce")
        recon = lvl - lag_lvl
        df.loc[miss & recon.notna(), change_col] = recon.loc[miss & recon.notna()]

_fill_change_from_levels("dlcch", "dlc")
_fill_change_from_levels("recch", "rect")
_fill_change_from_levels("invch", "invt")
_fill_change_from_levels("chech", "che")

# ---------------------------------------------------------------------------
# 3.6 Sparse zero-fill for structurally zero-inflated items (TRAIN-validated)
#   - Keeps miss_* as informative signal
# ---------------------------------------------------------------------------
SPARSE_CANDIDATES = [c for c in [
    "txach","txdc","txbcof","spi","siv","sppiv","ivstch","sppe","esubc","dvp","dv","dvc","dvt"
] if c in df.columns]

def _zero_fill_if_sparse(col: str, zero_share_thresh: float = 0.70, min_obs: int = 1000) -> bool:
    mflag = f"miss_{col}"
    if mflag not in df.columns:
        return False
    obs_mask = train_mask_for_imputation & (df[mflag] == 0)
    s = pd.to_numeric(df.loc[obs_mask, col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < min_obs:
        return False
    zero_share = float((s == 0).mean())
    med = float(s.median()) if s.notna().any() else np.nan
    if np.isfinite(med) and (med == 0.0) and (zero_share >= zero_share_thresh):
        df.loc[df[col].isna(), col] = 0.0
        return True
    return False

for c in SPARSE_CANDIDATES:
    _zero_fill_if_sparse(c)

# Explicit “tax components -> 0” rule (FFO proxy safety)
for c in ["txt", "txdc", "txach"]:
    if c in df.columns:
        df.loc[df[c].isna(), c] = 0.0

# ---------------------------------------------------------------------------
# 3.7 Step 3: Stocks — lag-1 fill -> peer median of ratio (x/at) with ratio-space clipping
# ---------------------------------------------------------------------------
STOCKS = [c for c in [
    "aco","act","ao","ap","at","caps","ceq","che","csho","cstk","dlc","dltt","intan","invt","lco","lct","lt",
    "mibt","ppent","pstk","pstkn","pstkr","re","rect","seq","tstk","ivaeq","mkvalt"
] if c in df.columns]

# lag-1 fill (past-only)
if STOCKS:
    lag1 = df.groupby("firm_id")[STOCKS].shift(1)
    df[STOCKS] = df[STOCKS].where(df[STOCKS].notna(), lag1)

# non-negativity only where economically necessary
NONNEG_STOCKS = set([c for c in STOCKS if c in {
    "aco","act","ao","ap","at","caps","che","csho","cstk","dlc","dltt","intan","invt","lco","lct","lt",
    "mibt","mkvalt","ppent","pstk","pstkn","pstkr","rect","tstk","ivaeq"
}])

def _fit_ratio_stats(train_df: pd.DataFrame, col: str, base: str = "at", qlo=0.01, qhi=0.99):
    mcol = f"miss_{col}"
    mbase = f"miss_{base}"
    s = pd.to_numeric(train_df[col], errors="coerce")
    b = pd.to_numeric(train_df[base], errors="coerce") if base in train_df.columns else None

    obs_s = (train_df[mcol] == 0) if mcol in train_df.columns else s.notna()

    if b is None:
        tr = train_df.loc[obs_s, group_cols + [col]].copy()
        tr[col] = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grp = tr.groupby(group_cols)[col].median()
        overall = float(tr[col].median()) if tr[col].notna().any() else 0.0
        return ("level", overall, grp, np.nan, np.nan)

    obs_b = (train_df[mbase] == 0) if mbase in train_df.columns else b.notna()
    valid = obs_s & obs_b & s.notna() & b.notna() & (b > 0)

    if valid.sum() < 200:
        tr = train_df.loc[obs_s, group_cols + [col]].copy()
        tr[col] = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grp = tr.groupby(group_cols)[col].median()
        overall = float(tr[col].median()) if tr[col].notna().any() else 0.0
        return ("level", overall, grp, np.nan, np.nan)

    ratio = (s[valid] / b[valid]).replace([np.inf, -np.inf], np.nan).dropna()
    overall = float(ratio.median()) if ratio.notna().any() else 0.0
    r_lo = float(ratio.quantile(qlo))
    r_hi = float(ratio.quantile(qhi))

    tmp = train_df.loc[valid, group_cols].copy()
    tmp["_ratio_"] = ratio.values
    grp = tmp.groupby(group_cols)["_ratio_"].median()
    return ("ratio", overall, grp, r_lo, r_hi)

def _apply_ratio_stats(df_all: pd.DataFrame, col: str, fit_obj, base: str = "at", nonneg: bool = False):
    kind, overall, grp, r_lo, r_hi = fit_obj
    miss = df_all[col].isna()
    if not miss.any():
        return

    if kind == "ratio" and base in df_all.columns:
        b = pd.to_numeric(df_all.loc[miss, base], errors="coerce")
        g = df_all.loc[miss, group_cols]

        if len(group_cols) == 1:
            mapped = g[group_cols[0]].map(grp)
        else:
            keys = list(map(tuple, g[group_cols].to_numpy()))
            mapped = pd.Series(keys, index=g.index).map(grp)

        r = pd.to_numeric(mapped, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(overall)

        # clip in ratio space (TRAIN-observed band)
        if np.isfinite(r_lo) and np.isfinite(r_hi) and (r_lo < r_hi):
            r = r.clip(r_lo, r_hi)

        fill = r * b
        fill = fill.where(b.notna() & (b > 0), np.nan)
        df_all.loc[miss & fill.notna(), col] = fill.loc[miss & fill.notna()].to_numpy()

    # fallback: group level medians (TRAIN-fit, observed-only)
    miss2 = df_all[col].isna()
    if miss2.any():
        mflag = f"miss_{col}"
        obs_train_mask = train_mask_for_imputation
        if mflag in df_all.columns:
            obs_train_mask = obs_train_mask & (df_all[mflag] == 0)

        tr = df_all.loc[obs_train_mask, group_cols + [col]].copy()
        tr[col] = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        lvl_overall = float(tr[col].median()) if tr[col].notna().any() else 0.0
        lvl_grp = tr.groupby(group_cols)[col].median()

        g2 = df_all.loc[miss2, group_cols]
        if len(group_cols) == 1:
            mapped2 = g2[group_cols[0]].map(lvl_grp)
        else:
            keys2 = list(map(tuple, g2[group_cols].to_numpy()))
            mapped2 = pd.Series(keys2, index=g2.index).map(lvl_grp)

        fill2 = pd.to_numeric(mapped2, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(lvl_overall)
        if nonneg:
            fill2 = fill2.clip(lower=0.0)
        df_all.loc[miss2, col] = fill2.to_numpy()

# Fit on training
tr_all = df.loc[train_mask_for_imputation].copy()
stock_fits = {c: _fit_ratio_stats(tr_all, c, base="at") for c in STOCKS}

# Apply to full df
for c in STOCKS:
    _apply_ratio_stats(df, c, stock_fits[c], base="at", nonneg=(c in NONNEG_STOCKS))

# ---------------------------------------------------------------------------
# 3.8 Step 4: Flows / income variables — ratio-median imputation (leakage-aware)
# ---------------------------------------------------------------------------
FLOWS = [c for c in ["ib","ibadj","ibc","niadj","nopi","oibdp","dp","oancf","fincf","ivncf","xint"] if c in df.columns]

# Debt-aware base for xint
if "xint" in df.columns and all(c in df.columns for c in ["dlc", "dltt"]):
    df["_td_for_xint"] = (
        pd.to_numeric(df["dlc"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["dltt"], errors="coerce").fillna(0.0)
    )
else:
    df["_td_for_xint"] = np.nan

# If total debt <= 0 and xint missing -> 0
if "xint" in df.columns and "_td_for_xint" in df.columns:
    td = pd.to_numeric(df["_td_for_xint"], errors="coerce").fillna(0.0)
    df.loc[df["xint"].isna() & (td <= 0), "xint"] = 0.0

flow_fits = {}
for c in FLOWS:
    base = "_td_for_xint" if (c == "xint" and "_td_for_xint" in df.columns) else "at"
    flow_fits[c] = _fit_ratio_stats(tr_all, c, base=base)

for c in FLOWS:
    base = "_td_for_xint" if (c == "xint" and "_td_for_xint" in df.columns) else "at"
    _apply_ratio_stats(df, c, flow_fits[c], base=base, nonneg=False)

# ---------------------------------------------------------------------------
# 3.9 KNN imputation (TRAIN-fit) for selected core balance-sheet items
#   - Applies ONLY to originally-missing rows (miss_* == 1)
#   - Uses a robust signed-log transform for magnitudes
# ---------------------------------------------------------------------------
KNN_TARGETS = [c for c in [
    "at","act","lct","lt","seq","dlc","dltt","che","ppent","rect","invt","re","ceq","caps"
] if c in df.columns]

# Non-negativity for clearly non-negative magnitudes (do NOT force seq/ceq/re to be non-negative)
KNN_NONNEG = set([c for c in KNN_TARGETS if c in {"at","act","lct","lt","dlc","dltt","che","ppent","rect","invt","caps"}])

if KNN_TARGETS:
    # KNN feature set (kept compact and available in your raw inputs)
    knn_feats = []
    for c in ["fyear", "size_decile", "mkvalt", "csho", "prcc_f", "prcc_c"]:
        if c in df.columns:
            knn_feats.append(c)
    knn_feats = list(dict.fromkeys(knn_feats))  # preserve order, unique

    knn_cols = list(dict.fromkeys(knn_feats + KNN_TARGETS))
    X = df[knn_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # For KNN, re-instate NaN only for the targets that were originally missing
    for t in KNN_TARGETS:
        mflag = f"miss_{t}"
        if mflag in df.columns:
            X.loc[df[mflag] == 1, t] = np.nan

    # Fill size_decile NaN with TRAIN median decile for distance stability
    if "size_decile" in X.columns:
        sd_train = X.loc[train_mask_for_imputation, "size_decile"]
        sd_fill = float(sd_train.median()) if sd_train.notna().any() else 5.0
        X["size_decile"] = X["size_decile"].fillna(sd_fill)

    # Robust transform: signed log1p for magnitudes; keep fyear/size_decile in levels
    def _signed_log1p(a: pd.Series) -> pd.Series:
        v = pd.to_numeric(a, errors="coerce")
        return np.sign(v) * np.log1p(np.abs(v))

    Z = X.copy()
    for c in Z.columns:
        if c not in {"fyear", "size_decile"}:
            Z[c] = _signed_log1p(Z[c])

    # Standardize using TRAIN-only stats (nan-safe)
    mu = Z.loc[train_mask_for_imputation].mean(skipna=True)
    sd = Z.loc[train_mask_for_imputation].std(skipna=True, ddof=0).replace(0.0, 1.0)
    sd = sd.fillna(1.0)
    mu = mu.fillna(0.0)

    Zs = (Z - mu) / sd

    # Fit KNN on TRAIN only; transform all rows
    imputer = KNNImputer(n_neighbors=25, weights="distance", metric="nan_euclidean")
    imputer.fit(Zs.loc[train_mask_for_imputation].to_numpy(dtype=float))
    Zs_imp = imputer.transform(Zs.to_numpy(dtype=float))
    Zs_imp = pd.DataFrame(Zs_imp, index=Zs.index, columns=Zs.columns)

    # Unscale
    Z_imp = Zs_imp * sd + mu

    # Invert signed log for magnitudes
    X_imp = X.copy()
    for c in Z_imp.columns:
        if c in {"fyear", "size_decile"}:
            X_imp[c] = Z_imp[c]
        else:
            v = pd.to_numeric(Z_imp[c], errors="coerce")
            X_imp[c] = np.sign(v) * (np.expm1(np.abs(v)))

    # Write back ONLY for originally-missing targets
    for t in KNN_TARGETS:
        mflag = f"miss_{t}"
        if mflag in df.columns:
            miss_mask = (df[mflag] == 1)
            df.loc[miss_mask, t] = pd.to_numeric(X_imp.loc[miss_mask, t], errors="coerce").to_numpy()

            if t in KNN_NONNEG:
                df.loc[miss_mask, t] = pd.to_numeric(df.loc[miss_mask, t], errors="coerce").clip(lower=0.0)

# ---------------------------------------------------------------------------
# 3.10 Change variables AFTER level imputations:
#   - Fill only if change is missing
#   - Reconstruct only when both levels were originally observed; otherwise impute change directly
# ---------------------------------------------------------------------------
def _fill_change_after_levels(change_col, level_col):
    if change_col in df.columns and level_col in df.columns:
        lvl = pd.to_numeric(df[level_col], errors="coerce")
        lag_lvl = pd.to_numeric(df.groupby("firm_id")[level_col].shift(1), errors="coerce")
        recon = lvl - lag_lvl

        miss_now = df[change_col].isna()

        # only reconstruct when both levels were originally observed
        obs_now = _obs(level_col)
        obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)
        can_recon = miss_now & obs_now & obs_lag & recon.notna()
        df.loc[can_recon, change_col] = recon.loc[can_recon]

        # remaining missing: direct group-median impute of change (TRAIN-fit, observed-only if possible)
        miss2 = df[change_col].isna()
        if miss2.any():
            mflag = f"miss_{change_col}"
            obs_train = train_mask_for_imputation
            if mflag in df.columns:
                obs_train = obs_train & (df[mflag] == 0)

            tr = df.loc[obs_train, group_cols + [change_col]].copy()
            tr[change_col] = pd.to_numeric(tr[change_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            overall = float(tr[change_col].median()) if tr[change_col].notna().any() else 0.0
            grp = tr.groupby(group_cols)[change_col].median()

            g = df.loc[miss2, group_cols]
            if len(group_cols) == 1:
                mapped = g[group_cols[0]].map(grp)
            else:
                keys = list(map(tuple, g[group_cols].to_numpy()))
                mapped = pd.Series(keys, index=g.index).map(grp)

            fill = pd.to_numeric(mapped, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(overall)
            df.loc[miss2, change_col] = fill.to_numpy()

_fill_change_after_levels("dlcch", "dlc")
_fill_change_after_levels("recch", "rect")
_fill_change_after_levels("invch", "invt")
_fill_change_after_levels("chech", "che")
if "apalch" in df.columns and "ap" in df.columns:
    _fill_change_after_levels("apalch", "ap")

# ---------------------------------------------------------------------------
# 3.11 Guardrail: cap ONLY-imputed values (miss_* == 1) to TRAIN-observed quantile band
# ---------------------------------------------------------------------------
REG_NONNEG = set(["at","act","lct","lt","dlc","dltt","che","ppent","rect","invt","caps"])  # for this script

def _cap_imputed_to_train_quantiles(
    df_all: pd.DataFrame,
    col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    nonneg: bool = False,
) -> None:
    flag = f"miss_{col}"
    if col not in df_all.columns or flag not in df_all.columns:
        return

    miss_mask = df_all[flag].astype(bool)
    if not miss_mask.any():
        return

    # bounds from TRAIN & originally-observed values
    obs = pd.to_numeric(df_all.loc[train_mask_for_imputation & (~miss_mask), col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if obs.notna().sum() < 200:
        obs = pd.to_numeric(df_all.loc[train_mask_for_imputation, col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if obs.notna().sum() == 0:
        return

    lo = float(obs.quantile(lower_q))
    hi = float(obs.quantile(upper_q))

    if nonneg:
        lo = max(0.0, lo) if np.isfinite(lo) else 0.0

    s = pd.to_numeric(df_all.loc[miss_mask, col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.clip(lo, hi)
    if nonneg:
        s = s.clip(lower=0.0)
    df_all.loc[miss_mask, col] = s.to_numpy()

CAP_COLS = sorted(
    (
        set(STOCKS + FLOWS + ["dlcch", "apalch", "recch", "invch", "chech"])
        | set(KNN_TARGETS)
    ) & set(df.columns)
)

for c in CAP_COLS:
    _cap_imputed_to_train_quantiles(
        df,
        c,
        lower_q=0.01,
        upper_q=0.99,
        nonneg=(c in NONNEG_STOCKS) or (c in REG_NONNEG),
    )

# ---------------------------------------------------------------------------
# 3.12 EDA AFTER imputation (missingness reduction + distribution deltas)
# ---------------------------------------------------------------------------
df_raw_post = df[raw].copy(deep=True)

post_miss = pd.DataFrame(
    {
        "col": raw,
        "n_na_post": [int(df_raw_post[c].isna().sum()) for c in raw],
        "pct_na_post": [float(df_raw_post[c].isna().mean() * 100.0) for c in raw],
        "train_pct_na_post": [
            float(df_raw_post.loc[train_mask_for_imputation, c].isna().mean() * 100.0) for c in raw
        ],
    }
)

changes = pre_miss.merge(post_miss, on="col", how="left")
changes["n_imputed"] = changes["n_na_pre"] - changes["n_na_post"]
changes["pct_points_na_reduction"] = changes["pct_na_pre"] - changes["pct_na_post"]

x_post = df_raw_post[raw].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
q_post = x_post.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
post_dist = pd.DataFrame(
    {
        "n_nonmiss_post": x_post.notna().sum(),
        "mean_post": x_post.mean(),
        "std_post": x_post.std(ddof=0),
        "min_post": x_post.min(),
        "p01_post": q_post[0.01],
        "p05_post": q_post[0.05],
        "p25_post": q_post[0.25],
        "p50_post": q_post[0.50],
        "p75_post": q_post[0.75],
        "p95_post": q_post[0.95],
        "p99_post": q_post[0.99],
        "max_post": x_post.max(),
    }
)

pre_dist_key = pre_dist[["n_nonmiss_pre","mean_pre","std_pre","p01_pre","p50_pre","p99_pre"]].copy() if raw else pd.DataFrame()
post_dist_key = post_dist[["n_nonmiss_post","mean_post","std_post","p01_post","p50_post","p99_post"]].copy() if raw else pd.DataFrame()

dist_delta = pre_dist_key.join(post_dist_key, how="outer")
dist_delta["delta_mean"] = dist_delta["mean_post"] - dist_delta["mean_pre"]
dist_delta["delta_std"]  = dist_delta["std_post"]  - dist_delta["std_pre"]
dist_delta["delta_p50"]  = dist_delta["p50_post"]  - dist_delta["p50_pre"]

# Imputed-only diagnostics (based on original missingness in df_raw_pre)
rows = []
for c in raw:
    pre_na_mask = df_raw_pre[c].isna()
    n_imp = int(pre_na_mask.sum())
    if n_imp == 0:
        rows.append((c, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    imp_vals = pd.to_numeric(df_raw_post.loc[pre_na_mask, c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    obs_vals = pd.to_numeric(df_raw_pre.loc[~pre_na_mask, c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    rows.append(
        (
            c,
            n_imp,
            float(imp_vals.mean()) if imp_vals.notna().any() else np.nan,
            float(imp_vals.median()) if imp_vals.notna().any() else np.nan,
            float(imp_vals.std(ddof=0)) if imp_vals.notna().any() else np.nan,
            float(obs_vals.mean()) if obs_vals.notna().any() else np.nan,
            float(obs_vals.median()) if obs_vals.notna().any() else np.nan,
        )
    )

imputed_only = pd.DataFrame(
    rows,
    columns=["col","n_imputed","imputed_mean","imputed_median","imputed_std","observed_mean_pre","observed_median_pre"],
).set_index("col")

print("\n=== EDA (AFTER imputation): Missingness on raw inputs + change ===")
cols_show = [
    "col", "n", "n_na_pre", "pct_na_pre", "n_na_post", "pct_na_post",
    "n_imputed", "pct_points_na_reduction", "train_pct_na_pre", "train_pct_na_post",
]
print(
    changes[cols_show]
    .sort_values(["n_imputed","pct_points_na_reduction"], ascending=[False, False])
    .round(4)
    .head(50)
)

print("\n=== Change analysis: Distribution deltas (post - pre) on raw inputs ===")
print(
    dist_delta[["n_nonmiss_pre","n_nonmiss_post","delta_mean","delta_std","delta_p50"]]
    .sort_values("delta_mean", key=lambda s: s.abs(), ascending=False)
    .round(6)
    .head(50)
)

print("\n=== Change analysis: Imputed-only vs observed (pre) summary ===")
print(
    imputed_only.assign(
        mean_gap_imputed_minus_observed=lambda d: d["imputed_mean"] - d["observed_mean_pre"],
        median_gap_imputed_minus_observed=lambda d: d["imputed_median"] - d["observed_median_pre"],
    )
    .sort_values("n_imputed", ascending=False)
    .round(6)
    .head(50)
)

```

    
    === EDA (BEFORE imputation): Missingness on raw inputs ===
           col      n  n_na_pre  pct_na_pre  train_n  train_pct_na_pre
    18   dlcch  75005     33143     44.1877    48458           42.9630
    5   apalch  75005     30371     40.4920    48458           39.0214
    75   txach  75005     22791     30.3860    48458           29.2501
    48  ivstch  75005     19194     25.5903    48458           23.0282
    66   recch  75005     12589     16.7842    48458           16.5938
    53  mkvalt  75005     12350     16.4656    48458           17.0189
    71    sppe  75005     12239     16.3176    48458           16.4307
    1      act  75005     10721     14.2937    48458           14.6581
    50     lct  75005     10695     14.2590    48458           14.5982
    84    xint  75005     10536     14.0471    48458           13.9234
    78  txditc  75005      9069     12.0912    48458           12.2044
    79     txp  75005      7942     10.5886    48458           10.6917
    29   esubc  75005      6798      9.0634    48458            8.8964
    49     lco  75005      6779      9.0381    48458            9.3277
    0      aco  75005      6778      9.0367    48458            9.3277
    60  prcc_f  75005      5956      7.9408    48458            8.8902
    59  prcc_c  75005      5942      7.9221    48458            8.8819
    40   invch  75005      5315      7.0862    48458            7.3445
    72   sppiv  75005      3967      5.2890    48458            4.9631
    44   ivaeq  75005      3548      4.7304    48458            4.8826
    61  prstkc  75005      3410      4.5464    48458            4.9486
    8     caps  75005      3074      4.0984    48458            4.2655
    6      aqc  75005      2812      3.7491    48458            3.2647
    23      dp  75005      2722      3.6291    48458            3.7909
    65      re  75005      2596      3.4611    48458            3.6650
    46    ivch  75005      2540      3.3864    48458            3.3844
    45    ivao  75005      2529      3.3718    48458            3.1821
    77    txdc  75005      2472      3.2958    48458            3.4050
    69     siv  75005      2299      3.0651    48458            3.1285
    57   oibdp  75005      2142      2.8558    48458            2.9366
    14    cstk  75005      1925      2.5665    48458            3.0294
    24     dpc  75005      1757      2.3425    48458            2.4268
    19   dltis  75005      1756      2.3412    48458            2.4681
    58   ppent  75005      1723      2.2972    48458            2.3814
    73    sstk  75005      1436      1.9145    48458            1.8593
    20    dltr  75005      1388      1.8505    48458            2.0616
    25      dv  75005      1137      1.5159    48458            1.5911
    70     spi  75005      1126      1.5012    48458            1.3785
    67    rect  75005       880      1.1733    48458            1.1247
    52    mibt  75005       842      1.1226    48458            1.3022
    13    csho  75005       804      1.0719    48458            1.2939
    28     dvt  75005       706      0.9413    48458            1.1412
    26     dvc  75005       703      0.9373    48458            1.1350
    39   intan  75005       654      0.8719    48458            0.8977
    41    invt  75005       652      0.8693    48458            0.8255
    76  txbcof  75005       619      0.8253    48458            1.0071
    9     capx  75005       503      0.6706    48458            0.7429
    83   xidoc  75005       477      0.6360    48458            0.7202
    3   aoloch  75005       462      0.6160    48458            0.6893
    38     ibc  75005       454      0.6053    48458            0.6810
    
    === EDA (BEFORE imputation): Distribution summary (raw inputs) ===
             n_nonmiss_pre      mean_pre       std_pre       min_pre  \
    consol               0           NaN           NaN           NaN   
    datafmt              0           NaN           NaN           NaN   
    conm                 0           NaN           NaN           NaN   
    indfmt               0           NaN           NaN           NaN   
    dlcch            41862  2.168121e+03  1.155250e+06 -8.555368e+07   
    apalch           44634  8.385212e+03  3.253932e+05 -4.849894e+07   
    txach            52214  3.214308e+02  3.140000e+04 -3.282293e+06   
    ivstch           55811  6.871673e+04  3.980372e+06 -1.752327e+07   
    recch            62416 -1.318528e+04  3.129434e+05 -2.297888e+07   
    mkvalt           62655  9.288907e+05  2.201298e+07  0.000000e+00   
    sppe             62766  6.939724e+03  1.245321e+05 -5.895100e+04   
    act              64284  6.440956e+05  4.645197e+06 -2.498000e+03   
    lct              64310  4.755088e+05  3.980895e+06  0.000000e+00   
    xint             64469  2.759126e+04  1.196406e+05 -7.284730e+05   
    txditc           65936  4.384545e+04  3.265589e+05 -1.616000e+03   
    txp              67063  1.062620e+04  1.001099e+05 -2.217000e+03   
    esubc            68207 -2.220041e+03  8.665624e+04 -8.775761e+06   
    lco              68226  2.039805e+05  1.784063e+06 -8.900000e+01   
    aco              68227  5.573244e+04  9.308496e+05 -2.500000e+01   
    prcc_f           69029  6.838158e+02  1.018087e+04  1.000000e-04   
    prcc_c           69039  7.067634e+02  1.003861e+04  1.000000e-04   
    invch            69690 -3.860008e+04  1.008997e+06 -8.472324e+07   
    sppiv            71038 -7.943608e+03  1.753101e+05 -1.598761e+07   
    ivaeq            71457  6.873078e+04  9.162568e+05 -1.551500e+04   
    prstkc           71595  2.902337e+04  2.408411e+05 -1.131100e+04   
    caps             71931  5.224117e+05  2.924028e+06 -2.333792e+07   
    aqc              72193  2.583727e+04  3.427732e+05 -2.002407e+07   
    dp               72283  7.989028e+04  7.010127e+05 -4.121000e+03   
    re               72409  4.063079e+05  4.773704e+06 -8.108442e+07   
    ivch             72465  4.336588e+05  1.267386e+07  0.000000e+00   
    ivao             72476  1.117387e+06  1.830844e+07  0.000000e+00   
    txdc             72533 -2.552908e+03  8.582746e+04 -8.127405e+06   
    siv              72706  3.204806e+05  9.796796e+06 -3.080000e+02   
    oibdp            72863  2.472644e+05  1.847387e+06 -6.234977e+06   
    cstk             73080  1.440901e+05  1.035999e+06 -1.466100e+04   
    dpc              73248  8.009984e+04  6.495898e+05 -1.551370e+05   
    dltis            73249  2.584506e+05  2.896443e+06 -4.257400e+04   
    ppent            73282  5.870571e+05  5.242606e+06  0.000000e+00   
    sstk             73569  2.949253e+04  2.324514e+05 -5.012830e+05   
    dltr             73617  2.679871e+05  2.856270e+06 -2.844300e+04   
    dv               73868  4.155940e+04  3.023818e+05 -1.631969e+06   
    spi              73879 -1.290533e+04  1.992509e+05 -1.354613e+07   
    rect             74125  2.121967e+06  3.060638e+07 -7.300000e+01   
    mibt             74163  5.164076e+04  6.023643e+05 -1.237174e+06   
    csho             74201  2.254364e+05  5.059555e+06  0.000000e+00   
    dvt              74299  4.154428e+04  3.087856e+05 -1.329470e+05   
    dvc              74302  4.081004e+04  3.071635e+05 -2.473000e+03   
    intan            74351  3.830857e+05  2.603599e+06  0.000000e+00   
    invt             74353  1.471455e+05  1.356403e+06  0.000000e+00   
    txbcof           74386  2.081901e+02  4.374733e+03 -9.510600e+04   
    
                  p01_pre      p05_pre     p25_pre     p50_pre      p75_pre  \
    consol            NaN          NaN         NaN         NaN          NaN   
    datafmt           NaN          NaN         NaN         NaN          NaN   
    conm              NaN          NaN         NaN         NaN          NaN   
    indfmt            NaN          NaN         NaN         NaN          NaN   
    dlcch   -2.265422e+05   -8701.6500      0.0000      0.0000       1.0000   
    apalch  -1.053346e+05  -14648.8000   -119.0000     50.0000    1693.0000   
    txach   -9.192830e+03    -193.0000      0.0000      0.0000       0.0000   
    ivstch  -2.099582e+05  -10799.5000      0.0000      0.0000       0.0000   
    recch   -2.890472e+05  -52494.2500  -2252.2500    -17.0000      54.0000   
    mkvalt   5.154000e-01       3.3549     48.9974    467.1767    4760.4262   
    sppe     0.000000e+00       0.0000      0.0000      0.0000       6.8300   
    act      1.700000e-01      21.0000   1732.0000  23074.0000  214833.2500   
    lct      6.600000e-01      26.5090   1277.0000  11405.0000  105630.0000   
    xint     0.000000e+00       0.0000     15.0000    433.0000    8992.0000   
    txditc   0.000000e+00       0.0000      0.0000      0.0000    1883.5000   
    txp      0.000000e+00       0.0000      0.0000      0.0000     152.0000   
    esubc   -3.866582e+04    -484.7000      0.0000      0.0000       0.0000   
    lco      0.000000e+00       0.0000    301.0000   3986.5000   46136.5000   
    aco      0.000000e+00       0.0000     58.0000    946.0000   10293.0000   
    prcc_f   1.000000e-02       0.1016      2.6000     13.9200      42.8400   
    prcc_c   1.000000e-02       0.1000      2.6200     14.0100      43.0250   
    invch   -3.362948e+05  -33105.1500   -172.0000      0.0000       0.0000   
    sppiv   -1.356993e+05  -11708.4500    -13.0000      0.0000       0.0000   
    ivaeq    0.000000e+00       0.0000      0.0000      0.0000       0.0000   
    prstkc   0.000000e+00       0.0000      0.0000      0.0000     846.5000   
    caps     0.000000e+00       0.0000   1645.0000  26381.0000  268051.0000   
    aqc     -1.760872e+04       0.0000      0.0000      0.0000       0.0000   
    dp       0.000000e+00       0.0000     68.0000   1079.4700   15628.5000   
    re      -2.484404e+06 -701966.6000 -80817.0000  -1251.0000   29686.0000   
    ivch     0.000000e+00       0.0000      0.0000      0.0000     275.0000   
    ivao     0.000000e+00       0.0000      0.0000      0.0000    3979.7500   
    txdc    -1.337068e+05  -18598.4000   -105.0000      0.0000       5.0000   
    siv      0.000000e+00       0.0000      0.0000      0.0000      61.0000   
    oibdp   -2.060061e+05  -47857.8000  -1873.5000    800.4000   46909.0000   
    cstk     0.000000e+00       0.0000      8.0000    197.0000   13957.2500   
    dpc      0.000000e+00       0.0000    100.3000   1542.0000   21382.2500   
    dltis    0.000000e+00       0.0000      0.0000      0.1500    1740.0000   
    ppent    0.000000e+00       0.0000    514.0000   8837.5000   93080.2500   
    sstk     0.000000e+00       0.0000      0.0000     58.0000    4216.0000   
    dltr     0.000000e+00       0.0000      0.0000    140.7000    8902.0000   
    dv       0.000000e+00       0.0000      0.0000      0.0000     884.0000   
    spi     -3.215187e+05  -47237.4000  -1071.0000      0.0000       0.0000   
    rect     0.000000e+00       0.0000    225.0000   5841.0000   97807.0000   
    mibt    -1.887380e+03       0.0000      0.0000      0.0000       0.0000   
    csho     3.150000e+00      28.0000   7219.0000  36145.0000  107343.0000   
    dvt      0.000000e+00       0.0000      0.0000      0.0000    1178.4450   
    dvc      0.000000e+00       0.0000      0.0000      0.0000     621.0000   
    intan    0.000000e+00       0.0000      0.0000   1365.0000   45975.0000   
    invt     0.000000e+00       0.0000      0.0000    275.3000   12041.0000   
    txbcof   0.000000e+00       0.0000      0.0000      0.0000       0.0000   
    
                 p95_pre      p99_pre       max_pre  
    consol           NaN          NaN           NaN  
    datafmt          NaN          NaN           NaN  
    conm             NaN          NaN           NaN  
    indfmt           NaN          NaN           NaN  
    dlcch      12288.900    239569.72  1.674825e+08  
    apalch     40626.050    280635.99  2.085900e+07  
    txach        544.000     16622.89  3.282293e+06  
    ivstch     14654.500    206862.80  3.964813e+08  
    recch      14636.250    102245.15  3.777953e+07  
    mkvalt   1137140.389  16710068.76  3.522211e+09  
    sppe        7315.750     96130.00  7.819213e+06  
    act      1887389.100  11141665.87  2.648894e+08  
    lct      1150175.850   8749902.35  2.275619e+08  
    xint      126670.400    475515.16  3.895731e+06  
    txditc    138738.750    824518.10  1.309682e+07  
    txp        20801.100    229018.76  5.843978e+06  
    esubc        503.700     18641.34  3.314734e+06  
    lco       528718.500   3510482.25  1.976466e+08  
    aco       124386.000    926127.42  1.973027e+08  
    prcc_f       271.900     17260.04  1.369125e+06  
    prcc_c       275.000     17524.72  1.369125e+06  
    invch       9558.700     99906.23  1.919409e+07  
    sppiv        779.000     12781.32  7.378669e+06  
    ivaeq      69436.200   1094948.44  4.184002e+07  
    prstkc     87371.300    608747.74  1.551860e+07  
    caps     2034165.500   7334828.10  1.724641e+08  
    aqc        71387.800    578083.72  5.188218e+07  
    dp        220915.500   1361138.02  3.714574e+07  
    re       1402781.800  11353646.12  2.219454e+08  
    ivch      441795.800   3486163.72  1.169666e+09  
    ivao     1193829.500  10658327.50  1.099816e+09  
    txdc       11204.800     85023.32  4.507005e+06  
    siv       237122.750   2260226.70  9.068858e+08  
    oibdp     727543.600   4683708.38  7.350011e+07  
    cstk      420270.900   2924964.10  3.904769e+07  
    dpc       252347.000   1229121.52  3.648387e+07  
    dltis     692136.800   4321096.08  1.818726e+08  
    ppent    1817488.900  10065753.57  2.926841e+08  
    sstk      111376.400    525775.68  2.036114e+07  
    dltr      761854.400   3911400.36  1.796217e+08  
    dv        122377.700    899427.56  1.296791e+07  
    spi         2226.100     54763.82  1.148318e+07  
    rect     1917288.600  17121476.16  1.218880e+09  
    mibt       66241.300   1067080.28  3.075527e+07  
    csho      533438.000   2530459.00  6.064077e+08  
    dvt       124549.500    879155.68  1.485872e+07  
    dvc       120572.700    869661.95  1.437446e+07  
    intan    1345701.500   7366060.00  9.849622e+07  
    invt      385661.800   2910600.48  1.043358e+08  
    txbcof         0.000      3680.50  6.321680e+05  


    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2286657439.py:434: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)
    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2286657439.py:434: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)
    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2286657439.py:434: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)
    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2286657439.py:434: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)
    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2286657439.py:434: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      obs_lag = obs_now.groupby(df["firm_id"]).shift(1).fillna(False)


    
    === EDA (AFTER imputation): Missingness on raw inputs + change ===
           col      n  n_na_pre  pct_na_pre  n_na_post  pct_na_post  n_imputed  \
    0    dlcch  75005     33143     44.1877          0       0.0000      33143   
    1   apalch  75005     30371     40.4920          0       0.0000      30371   
    2    txach  75005     22791     30.3860          0       0.0000      22791   
    4    recch  75005     12589     16.7842          0       0.0000      12589   
    5   mkvalt  75005     12350     16.4656          0       0.0000      12350   
    7      act  75005     10721     14.2937          0       0.0000      10721   
    8      lct  75005     10695     14.2590          0       0.0000      10695   
    9     xint  75005     10536     14.0471          0       0.0000      10536   
    12   esubc  75005      6798      9.0634          0       0.0000       6798   
    13     lco  75005      6779      9.0381          0       0.0000       6779   
    14     aco  75005      6778      9.0367          0       0.0000       6778   
    17   invch  75005      5315      7.0862          0       0.0000       5315   
    19   ivaeq  75005      3548      4.7304          0       0.0000       3548   
    21    caps  75005      3074      4.0984          0       0.0000       3074   
    23      dp  75005      2722      3.6291          0       0.0000       2722   
    24      re  75005      2596      3.4611          0       0.0000       2596   
    27    txdc  75005      2472      3.2958          0       0.0000       2472   
    28     siv  75005      2299      3.0651          0       0.0000       2299   
    29   oibdp  75005      2142      2.8558          0       0.0000       2142   
    30    cstk  75005      1925      2.5665          0       0.0000       1925   
    33   ppent  75005      1723      2.2972          0       0.0000       1723   
    38    rect  75005       880      1.1733          0       0.0000        880   
    39    mibt  75005       842      1.1226          0       0.0000        842   
    40    csho  75005       804      1.0719          0       0.0000        804   
    43   intan  75005       654      0.8719          0       0.0000        654   
    44    invt  75005       652      0.8693          0       0.0000        652   
    45  txbcof  75005       619      0.8253          0       0.0000        619   
    49     ibc  75005       454      0.6053          0       0.0000        454   
    54   ivncf  75005       316      0.4213          0       0.0000        316   
    55   fincf  75005       315      0.4200          0       0.0000        315   
    56   chech  75005       304      0.4053          0       0.0000        304   
    57   oancf  75005       303      0.4040          0       0.0000        303   
    58    pstk  75005       219      0.2920          0       0.0000        219   
    59   pstkn  75005       211      0.2813          0       0.0000        211   
    60    dltt  75005       188      0.2506          0       0.0000        188   
    61    tstk  75005       180      0.2400          0       0.0000        180   
    62     ceq  75005       149      0.1987          0       0.0000        149   
    15  prcc_f  75005      5956      7.9408       5826       7.7675        130   
    16  prcc_c  75005      5942      7.9221       5813       7.7501        129   
    63      lt  75005       101      0.1347          0       0.0000        101   
    64      ap  75005        88      0.1173          0       0.0000         88   
    65   pstkr  75005        75      0.1000          0       0.0000         75   
    66     dvp  75005        36      0.0480          0       0.0000         36   
    67     dlc  75005        31      0.0413          0       0.0000         31   
    68     txt  75005        15      0.0200          0       0.0000         15   
    69    nopi  75005        12      0.0160          0       0.0000         12   
    70     che  75005         6      0.0080          0       0.0000          6   
    71      ao  75005         5      0.0067          0       0.0000          5   
    74     seq  75005         3      0.0040          0       0.0000          3   
    3   ivstch  75005     19194     25.5903      19194      25.5903          0   
    
        pct_points_na_reduction  train_pct_na_pre  train_pct_na_post  
    0                   44.1877           42.9630             0.0000  
    1                   40.4920           39.0214             0.0000  
    2                   30.3860           29.2501             0.0000  
    4                   16.7842           16.5938             0.0000  
    5                   16.4656           17.0189             0.0000  
    7                   14.2937           14.6581             0.0000  
    8                   14.2590           14.5982             0.0000  
    9                   14.0471           13.9234             0.0000  
    12                   9.0634            8.8964             0.0000  
    13                   9.0381            9.3277             0.0000  
    14                   9.0367            9.3277             0.0000  
    17                   7.0862            7.3445             0.0000  
    19                   4.7304            4.8826             0.0000  
    21                   4.0984            4.2655             0.0000  
    23                   3.6291            3.7909             0.0000  
    24                   3.4611            3.6650             0.0000  
    27                   3.2958            3.4050             0.0000  
    28                   3.0651            3.1285             0.0000  
    29                   2.8558            2.9366             0.0000  
    30                   2.5665            3.0294             0.0000  
    33                   2.2972            2.3814             0.0000  
    38                   1.1733            1.1247             0.0000  
    39                   1.1226            1.3022             0.0000  
    40                   1.0719            1.2939             0.0000  
    43                   0.8719            0.8977             0.0000  
    44                   0.8693            0.8255             0.0000  
    45                   0.8253            1.0071             0.0000  
    49                   0.6053            0.6810             0.0000  
    54                   0.4213            0.4788             0.0000  
    55                   0.4200            0.4767             0.0000  
    56                   0.4053            0.4623             0.0000  
    57                   0.4040            0.4602             0.0000  
    58                   0.2920            0.2951             0.0000  
    59                   0.2813            0.2868             0.0000  
    60                   0.2506            0.2889             0.0000  
    61                   0.2400            0.2414             0.0000  
    62                   0.1987            0.2043             0.0000  
    15                   0.1733            8.8902             8.6735  
    16                   0.1720            8.8819             8.6652  
    63                   0.1347            0.1259             0.0000  
    64                   0.1173            0.0991             0.0000  
    65                   0.1000            0.1032             0.0000  
    66                   0.0480            0.0495             0.0000  
    67                   0.0413            0.0495             0.0000  
    68                   0.0200            0.0144             0.0000  
    69                   0.0160            0.0227             0.0000  
    70                   0.0080            0.0124             0.0000  
    71                   0.0067            0.0103             0.0000  
    74                   0.0040            0.0062             0.0000  
    3                    0.0000           23.0282            23.0282  
    
    === Change analysis: Distribution deltas (post - pre) on raw inputs ===
            n_nonmiss_pre  n_nonmiss_post     delta_mean     delta_std   delta_p50
    mkvalt          62655           75005  339908.830337 -1.766668e+06    119.9347
    rect            74125           75005  -24635.263120 -1.792322e+05   -216.0000
    act             64284           75005  -22454.818040 -3.155523e+05  14532.0000
    lco             68226           75005   21036.613834 -6.219595e+04   1871.5000
    caps            71931           75005  -19698.650266 -5.887712e+04  -2173.0000
    lct             64310           75005  -17372.380933 -2.682944e+05   6589.0000
    re              72409           75005  -12813.089960 -8.183700e+04    273.0000
    ppent           73282           75005  -12344.410435 -5.990450e+04    238.5000
    siv             72706           75005   -9823.145072 -1.511525e+05      0.0000
    lt              74904           75005   -6279.209873 -4.631426e+04     62.0000
    aco             68227           75005    4416.620600 -4.056789e+04    459.0000
    apalch          44634           75005    3844.702802 -7.037396e+04     15.0000
    cstk            73080           75005   -3371.893764 -1.293280e+04      0.3800
    ivaeq           71457           75005   -3175.746266 -2.180294e+04      0.0000
    recch           62416           75005    2935.206610 -2.586869e+04      6.7200
    invch           69690           75005    2632.929381 -3.631501e+04      0.0000
    intan           74351           75005   -2435.410689 -1.065323e+04     17.0000
    ap              74917           75005   -2409.458333 -2.223757e+04      4.0000
    dltt            74817           75005   -2284.657106 -1.061125e+04    -33.0000
    ceq             74856           75005    1948.514470 -2.647329e+03    -16.5000
    dp              72283           75005    1514.527654 -1.139971e+04    141.5300
    xint            64469           75005   -1348.022514 -7.828917e+03    -15.0000
    csho            74201           75005   -1198.257302 -2.703964e+04    -64.0000
    invt            74353           75005    -988.862996 -5.809818e+03      7.0900
    oibdp           72863           75005    -792.923284 -2.530035e+04    170.6000
    oancf           74702           75005    -727.783680 -4.392208e+03     -5.2000
    mibt            74163           75005    -579.008278 -3.365934e+03      0.0000
    ivncf           74689           75005     558.836145 -3.279674e+03      7.0000
    esubc           68207           75005     201.211062 -4.017797e+03      0.0000
    ibc             74551           75005    -190.518158 -2.001482e+03     -0.1700
    chech           74701           75005    -184.196906 -4.120815e+03     -0.1100
    dlcch           41862           75005     144.280720 -2.911747e+05      0.0000
    tstk            74825           75005    -140.814051 -5.864455e+02      0.0000
    txach           52214           75005     -97.669890 -5.200987e+03      0.0000
    txdc            72533           75005      84.138237 -1.424961e+03      0.0000
    che             74999           75005     -70.624838 -6.643516e+02     -2.0000
    dlc             74974           75005     -60.471895 -1.545875e+03      1.0000
    fincf           74690           75005     -43.033951 -3.252390e+03      0.0700
    seq             75002           75005     -38.472134 -1.212666e+02     -3.5000
    ao              75000           75005     -37.929044 -4.493026e+02     -0.1000
    pstk            74786           75005     -33.132292 -3.322790e+02      0.0000
    pstkn           74794           75005     -12.933761 -2.168597e+02      0.0000
    pstkr           74930           75005      -7.086945 -8.506999e+01      0.0000
    nopi            74993           75005       6.575298 -6.715941e+01      0.0000
    txt             74990           75005      -5.758203 -2.315719e+01      0.0000
    txbcof          74386           75005      -1.718148 -1.804855e+01      0.0000
    prcc_c          69039           69167      -1.207154 -9.250138e+00     -0.0100
    prcc_f          69029           69158      -1.191174 -9.460768e+00     -0.0200
    dvp             74969           75005      -0.331571 -3.001890e+00      0.0000
    at              75005           75005       0.000000  0.000000e+00      0.0000
    
    === Change analysis: Imputed-only vs observed (pre) summary ===
            n_imputed  imputed_mean  imputed_median   imputed_std  \
    col                                                             
    dlcch       33143  2.494638e+03        0.000000  6.300460e+04   
    apalch      30371  1.788019e+04      115.700000  7.037716e+04   
    txach       22791  0.000000e+00        0.000000  0.000000e+00   
    ivstch      19194           NaN             NaN           NaN   
    recch       12589  4.302616e+03        0.000000  7.211934e+04   
    mkvalt      12350  2.993252e+06     5729.901897  5.252785e+06   
    sppe        12239           NaN             NaN           NaN   
    act         10721  4.869998e+05   139778.546064  1.320445e+06   
    lct         10695  3.536747e+05    68153.642247  1.165729e+06   
    xint        10536  1.799479e+04      418.000000  3.653868e+04   
    txditc       9069           NaN             NaN           NaN   
    txp          7942           NaN             NaN           NaN   
    esubc        6798  0.000000e+00        0.000000  0.000000e+00   
    lco          6779  4.367362e+05    88275.927500  8.491035e+05   
    aco          6778  1.046065e+05    20443.802494  2.162451e+05   
    prcc_f       5956  4.521732e+01        3.050000  1.277017e+02   
    prcc_c       5942  5.445709e+01        2.100000  1.993917e+02   
    invch        5315 -1.444317e+03        0.000000  3.507483e+04   
    sppiv        3967           NaN             NaN           NaN   
    ivaeq        3548  1.595255e+03        0.000000  2.544047e+04   
    prstkc       3410           NaN             NaN           NaN   
    caps         3074  4.176843e+04     6931.009560  1.082224e+05   
    aqc          2812           NaN             NaN           NaN   
    dp           2722  1.216232e+05    53082.966710  2.300360e+05   
    re           2596  3.610538e+04      -63.518032  5.242238e+05   
    ivch         2540           NaN             NaN           NaN   
    ivao         2529           NaN             NaN           NaN   
    txdc         2472  0.000000e+00        0.000000  0.000000e+00   
    siv          2299  0.000000e+00        0.000000  0.000000e+00   
    oibdp        2142  2.194991e+05    90782.404109  4.015096e+05   
    cstk         1925  1.270890e+04      204.974675  1.374580e+05   
    dpc          1757           NaN             NaN           NaN   
    dltis        1756           NaN             NaN           NaN   
    ppent        1723  4.968479e+04    15615.380231  1.276160e+05   
    sstk         1436           NaN             NaN           NaN   
    dltr         1388           NaN             NaN           NaN   
    dv           1137           NaN             NaN           NaN   
    spi          1126           NaN             NaN           NaN   
    rect          880  2.223032e+04      140.926861  1.245324e+05   
    mibt          842  6.294546e+01        0.000000  7.999841e+02   
    csho          804  1.136512e+05    34012.802470  3.595284e+05   
    dvt           706           NaN             NaN           NaN   
    dvc           703           NaN             NaN           NaN   
    intan         654  1.037768e+05     3307.075685  5.935900e+05   
    invt          652  3.338837e+04      614.036198  1.333228e+05   
    txbcof        619  0.000000e+00        0.000000  0.000000e+00   
    capx          503           NaN             NaN           NaN   
    xidoc         477           NaN             NaN           NaN   
    aoloch        462           NaN             NaN           NaN   
    ibc           454  4.592746e+04        0.070076  2.269067e+05   
    
            observed_mean_pre  observed_median_pre  \
    col                                              
    dlcch        2.168121e+03               0.0000   
    apalch       8.385212e+03              50.0000   
    txach        3.214308e+02               0.0000   
    ivstch       6.871673e+04               0.0000   
    recch       -1.318528e+04             -17.0000   
    mkvalt       9.288907e+05             467.1767   
    sppe         6.939724e+03               0.0000   
    act          6.440956e+05           23074.0000   
    lct          4.755088e+05           11405.0000   
    xint         2.759126e+04             433.0000   
    txditc       4.384545e+04               0.0000   
    txp          1.062620e+04               0.0000   
    esubc       -2.220041e+03               0.0000   
    lco          2.039805e+05            3986.5000   
    aco          5.573244e+04             946.0000   
    prcc_f       6.838158e+02              13.9200   
    prcc_c       7.067634e+02              14.0100   
    invch       -3.860008e+04               0.0000   
    sppiv       -7.943608e+03               0.0000   
    ivaeq        6.873078e+04               0.0000   
    prstkc       2.902337e+04               0.0000   
    caps         5.224117e+05           26381.0000   
    aqc          2.583727e+04               0.0000   
    dp           7.989028e+04            1079.4700   
    re           4.063079e+05           -1251.0000   
    ivch         4.336588e+05               0.0000   
    ivao         1.117387e+06               0.0000   
    txdc        -2.552908e+03               0.0000   
    siv          3.204806e+05               0.0000   
    oibdp        2.472644e+05             800.4000   
    cstk         1.440901e+05             197.0000   
    dpc          8.009984e+04            1542.0000   
    dltis        2.584506e+05               0.1500   
    ppent        5.870571e+05            8837.5000   
    sstk         2.949253e+04              58.0000   
    dltr         2.679871e+05             140.7000   
    dv           4.155940e+04               0.0000   
    spi         -1.290533e+04               0.0000   
    rect         2.121967e+06            5841.0000   
    mibt         5.164076e+04               0.0000   
    csho         2.254364e+05           36145.0000   
    dvt          4.154428e+04               0.0000   
    dvc          4.081004e+04               0.0000   
    intan        3.830857e+05            1365.0000   
    invt         1.471455e+05             275.3000   
    txbcof       2.081901e+02               0.0000   
    capx         8.862010e+04             887.0000   
    xidoc        8.325909e+02               0.0000   
    aoloch       3.689360e+04              -0.0100   
    ibc          7.740282e+04               2.0000   
    
            mean_gap_imputed_minus_observed  median_gap_imputed_minus_observed  
    col                                                                         
    dlcch                      3.265177e+02                           0.000000  
    apalch                     9.494977e+03                          65.700000  
    txach                     -3.214308e+02                           0.000000  
    ivstch                              NaN                                NaN  
    recch                      1.748790e+04                          17.000000  
    mkvalt                     2.064361e+06                        5262.725197  
    sppe                                NaN                                NaN  
    act                       -1.570958e+05                      116704.546064  
    lct                       -1.218341e+05                       56748.642247  
    xint                      -9.596472e+03                         -15.000000  
    txditc                              NaN                                NaN  
    txp                                 NaN                                NaN  
    esubc                      2.220041e+03                           0.000000  
    lco                        2.327557e+05                       84289.427500  
    aco                        4.887410e+04                       19497.802494  
    prcc_f                    -6.385985e+02                         -10.870000  
    prcc_c                    -6.523063e+02                         -11.910000  
    invch                      3.715576e+04                           0.000000  
    sppiv                               NaN                                NaN  
    ivaeq                     -6.713553e+04                           0.000000  
    prstkc                              NaN                                NaN  
    caps                      -4.806432e+05                      -19449.990440  
    aqc                                 NaN                                NaN  
    dp                         4.173297e+04                       52003.496710  
    re                        -3.702025e+05                        1187.481968  
    ivch                                NaN                                NaN  
    ivao                                NaN                                NaN  
    txdc                       2.552908e+03                           0.000000  
    siv                       -3.204806e+05                           0.000000  
    oibdp                     -2.776527e+04                       89982.004109  
    cstk                      -1.313812e+05                           7.974675  
    dpc                                 NaN                                NaN  
    dltis                               NaN                                NaN  
    ppent                     -5.373723e+05                        6777.880231  
    sstk                                NaN                                NaN  
    dltr                                NaN                                NaN  
    dv                                  NaN                                NaN  
    spi                                 NaN                                NaN  
    rect                      -2.099736e+06                       -5700.073139  
    mibt                      -5.157781e+04                           0.000000  
    csho                      -1.117852e+05                       -2132.197530  
    dvt                                 NaN                                NaN  
    dvc                                 NaN                                NaN  
    intan                     -2.793088e+05                        1942.075685  
    invt                      -1.137572e+05                         338.736198  
    txbcof                    -2.081901e+02                           0.000000  
    capx                                NaN                                NaN  
    xidoc                               NaN                                NaN  
    aoloch                              NaN                                NaN  
    ibc                       -3.147536e+04                          -1.929924  



```python
# =============================================================================
# Feature Engineering & Target Construction
#   - Build leverage / coverage / cash-flow-to-debt ratios commonly used in credit analysis
#   - Define a 'highly leveraged' distress proxy from multiple ratio-based conditions
#   - Create the supervised-learning target: next-year distress within the same firm
# =============================================================================

firm_col = "firm_id"

dlc = pd.to_numeric(df.get("dlc", np.nan), errors="coerce")
dltt = pd.to_numeric(df.get("dltt", np.nan), errors="coerce")
df["total_debt"] = pd.concat([dlc, dltt], axis=1).sum(axis=1, min_count=1)

seq = pd.to_numeric(df.get("seq", np.nan), errors="coerce")
mibt = pd.to_numeric(df.get("mibt", 0.0), errors="coerce")
df["equity_plus_mi_sp"] = seq + mibt
df["total_capital_sp"] = df["total_debt"] + df["equity_plus_mi_sp"]

# --- CHANGED (minimal): handle total capital <= 0 as extreme leverage + flag ---
cap_s = pd.to_numeric(df["total_capital_sp"], errors="coerce")
df["cap_nonpos_flag"] = (cap_s.notna() & (cap_s <= 0)).astype("int8")
df["sp_debt_to_capital"] = safe_divide(df["total_debt"], cap_s)
df.loc[df["cap_nonpos_flag"] == 1, "sp_debt_to_capital"] = np.inf
# ---------------------------------------------------------------------------

oibdp = pd.to_numeric(df.get("oibdp", np.nan), errors="coerce")
xint = pd.to_numeric(df.get("xint", np.nan), errors="coerce")

# --- CHANGED (minimal): handle EBITDA <= 0 as extreme leverage + flag ---
df["ebitda_nonpos_flag"] = (oibdp.notna() & (oibdp <= 0)).astype("int8")
df["sp_debt_to_ebitda"] = safe_divide(df["total_debt"], oibdp)
df.loc[df["ebitda_nonpos_flag"] == 1, "sp_debt_to_ebitda"] = np.inf
# ---------------------------------------------------------------------------

# --- OPTIONAL transparency (simple): flag if tax proxy components missing (no complex proxying) ---
txt_raw = pd.to_numeric(df.get("txt", np.nan), errors="coerce")
txdc_raw = pd.to_numeric(df.get("txdc", np.nan), errors="coerce")
txach_raw = pd.to_numeric(df.get("txach", np.nan), errors="coerce")
df["tax_proxy_incomplete"] = (txt_raw.isna() | txdc_raw.isna() | txach_raw.isna()).astype("int8")

txt = txt_raw.fillna(0.0)
txdc = txdc_raw.fillna(0.0)
txach = txach_raw.fillna(0.0)
df["cash_tax_paid_proxy"] = txt - txdc - txach

df["ffo_proxy"] = oibdp - xint - pd.to_numeric(df["cash_tax_paid_proxy"], errors="coerce")
df["sp_ffo_to_debt"] = safe_divide(df["ffo_proxy"], df["total_debt"])

oancf = pd.to_numeric(df.get("oancf", np.nan), errors="coerce")
capx = pd.to_numeric(df.get("capx", np.nan), errors="coerce")
df["sp_cfo_to_debt"] = safe_divide(oancf, df["total_debt"])
df["focf"] = oancf - capx
df["sp_focf_to_debt"] = safe_divide(df["focf"], df["total_debt"])

# --- CHANGED (minimal): make outflows robust to sign conventions ---
dv = pd.to_numeric(df.get("dv", 0.0), errors="coerce").fillna(0.0).abs()
prstkc = pd.to_numeric(df.get("prstkc", 0.0), errors="coerce").fillna(0.0).abs()
df["dcf"] = df["focf"] - dv - prstkc
df["sp_dcf_to_debt"] = safe_divide(df["dcf"], df["total_debt"])
# ------------------------------------------------------------------

# Log transforms (log1p handles 0 nicely). Negative → NaN.
for c in ["at", "mkvalt"]:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        df[f"log_{c}"] = np.where(s >= 0, np.log1p(s), np.nan)

# Interest coverage: EBITDA / |interest expense|
# Interest coverage can explode when interest expense is near zero.
# Stabilize by flooring the denominator, capping extreme values, and log-transforming.
INT_FLOOR = 1.0  # minimum |interest expense| to avoid blow-ups
IC_CAP = 100.0  # cap extreme coverage magnitudes before log transform
denom_ic = np.maximum(xint.abs(), INT_FLOOR)
df["sp_interest_coverage_raw"] = safe_divide(oibdp, denom_ic)
df["sp_interest_coverage_is_capped"] = (df["sp_interest_coverage_raw"].abs() > IC_CAP).astype("int8")
df["sp_interest_coverage_denom_floored"] = (xint.abs() < INT_FLOOR).astype("int8")
df["sp_interest_coverage"] = np.sign(df["sp_interest_coverage_raw"]) * np.log1p(
    np.minimum(df["sp_interest_coverage_raw"].abs(), IC_CAP)
)

# Distress proxy: 'highly leveraged' condition using multiple credit-ratio thresholds
td = pd.to_numeric(df["total_debt"], errors="coerce").to_numpy(dtype=float)
cap = pd.to_numeric(df["total_capital_sp"], errors="coerce").to_numpy(dtype=float)
eb = pd.to_numeric(oibdp, errors="coerce").to_numpy(dtype=float)
ffo = pd.to_numeric(df["ffo_proxy"], errors="coerce").to_numpy(dtype=float)

ffo_to_debt_pct = 100.0 * safe_divide(ffo, td)

debt_to_capital_pct = 100.0 * safe_divide(td, cap)
# --- CHANGED (minimal): cap <= 0 => extreme leverage (so hl_cap can trigger) ---
debt_to_capital_pct = np.where(np.isfinite(cap) & (cap <= 0), np.inf, debt_to_capital_pct)

debt_to_ebitda = safe_divide(td, eb)
# --- CHANGED (minimal): EBITDA <= 0 => extreme leverage (so hl_deb can trigger) ---
debt_to_ebitda = np.where(np.isfinite(eb) & (eb <= 0), np.inf, debt_to_ebitda)

# Three “highly leveraged” conditions (S&P table)
hl_ffo = (td > 0) & (ffo_to_debt_pct < 15.0)  # FFO/total debt < 15%
hl_cap = (td > 0) & (debt_to_capital_pct > 55.0)  # TD/total capital > 55% (or cap<=0 => inf)
hl_deb = (td > 0) & (debt_to_ebitda > 4.5)  # TD/EBITDA > 4.5 (or EBITDA<=0 => inf)

is_highly_leveraged = hl_ffo & hl_cap & hl_deb

# Equity strictly negative and not missing
# --- CHANGED (minimal): use standard negative equity cutoff and actually use it ---
is_equity_negative = seq.notna() & (seq < 0)

# Final distress rule: highly leveraged OR negative equity (simple, avoids computed-but-unused flag)
df["distress_dummy"] = (is_highly_leveraged | is_equity_negative.to_numpy(dtype=bool)).astype("int8")
# ------------------------------------------------------------------------------

# Target: next year's distress (within firm)
df["target_next_year_distress"] = (
    df.groupby(firm_col)["distress_dummy"].shift(-1)
)

# Indicator for whether the next-year label is observed (attrition / survivorship considerations)
df["has_next_year_obs"] = df["target_next_year_distress"].notna().astype("int8")
n_total = len(df)
n_missing_next = int((df["has_next_year_obs"] == 0).sum())
print(
    f"Next-year label availability: {n_total - n_missing_next:,}/{n_total:,} observed "
    f"({(1 - n_missing_next / max(n_total, 1)):.1%}); missing next-year={n_missing_next:,}."
)

# Modeling sample: restrict to observed next-year labels (do NOT overwrite df)
df_model = df[df["has_next_year_obs"] == 1].copy().reset_index(drop=True)
df_model["target_next_year_distress"] = df_model["target_next_year_distress"].astype("int8")

```

    Next-year label availability: 63,602/75,005 observed (84.8%); missing next-year=11,403.



```python
# =============================================================================
# 4. Distinct moments / event indicators (interpretable "drivers & levers")
# Purpose:
#   - Translate continuous accounting ratios into discrete, management-relevant "events"
#   - Use TRAIN pool only for any distribution-based thresholds (no look-ahead)
# =============================================================================

df_events = df.copy()


# ----------------------------
# Helpers
# ----------------------------
def _as_series(x, index) -> pd.Series:
    """Ensure x is a pandas Series aligned to index (handles scalars, arrays, Series)."""
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if isinstance(x, (pd.Index, list, tuple, np.ndarray)):
        return pd.Series(x, index=index, dtype="float64")
    return pd.Series([x] * len(index), index=index, dtype="float64")


def lag(series: pd.Series, k: int = 1) -> pd.Series:
    """Firm-level lag (t-k) using firm_id."""
    s = _as_series(series, df_events.index)
    return s.groupby(df_events["firm_id"]).shift(k)


def ratio(a, b) -> pd.Series:
    """Leakage-safe ratio helper that preserves the DataFrame index."""
    a_s = _as_series(a, df_events.index)
    b_s = _as_series(b, df_events.index)
    return pd.Series(safe_divide(a_s, b_s), index=df_events.index, dtype="float64")


# Define training mask for distribution-aware thresholds (exclude val/test)
pool_mask = (df_events["label_year"] <= TRAIN_CUTOFF_LABEL_YEAR)
train_mask = pool_mask.copy()
if "val_years" in globals() and len(val_years) > 0:
    train_mask = train_mask & (~df_events["label_year"].isin(val_years))

# =============================================================================
# 4.1 Dividend "moments"
# =============================================================================
# Distress literature treats dividend cuts as a signal of cash-flow stress / financing constraints.
# Given the mass at zero (many firms pay no dividends), define events conditional on prior payment.

div = pd.to_numeric(
    _as_series(df_events["dv"] if "dv" in df_events.columns else 0.0, df_events.index),
    errors="coerce",
).fillna(0.0)
div_l1 = lag(div, 1).fillna(0.0)

# Distribution-aware cut threshold among dividend payers (TRAIN pool only)
payer_mask_train = train_mask & (div_l1 > 0)
pct_change = (div - div_l1) / div_l1.replace(0, np.nan)
df_events["div_pct_change"] = pct_change

cut_q = pct_change.loc[payer_mask_train].quantile(0.10)  # 10th percentile of YoY pct change for payers
cut_threshold = float(cut_q) if pd.notna(cut_q) else -0.25
cut_threshold = min(cut_threshold, -0.10)  # enforce a meaningful negative cut

df_events["evt_div_cut"] = ((div_l1 > 0) & (pct_change <= cut_threshold)).astype("int8")
df_events["evt_div_suspend"] = ((div_l1 > 0) & (div <= 0)).astype("int8")
df_events["evt_div_initiate"] = ((div_l1 <= 0) & (div > 0)).astype("int8")

# =============================================================================
# 4.2 Coverage / leverage / liquidity / profitability "moments"
# =============================================================================

# --- Interest coverage ---
cov = pd.to_numeric(
    _as_series(
        df_events["sp_interest_coverage"] if "sp_interest_coverage" in df_events.columns else np.nan,
        df_events.index,
    ),
    errors="coerce",
)
cov_l1 = lag(cov, 1)

df_events["evt_cov_breach"] = (cov.notna() & (cov < 1.0)).astype("int8")  # EBITDA/interest < 1

# Distribution-aware collapse threshold: cov_t / cov_{t-1} unusually low (TRAIN pool only)
cov_ratio = ratio(cov, cov_l1)
df_events["cov_ratio"] = cov_ratio
cov_drop_q = cov_ratio.loc[train_mask].quantile(0.05)
cov_drop_thr = float(cov_drop_q) if pd.notna(cov_drop_q) else 0.5
cov_drop_thr = min(cov_drop_thr, 0.7)  # enforce meaningful drop

df_events["evt_cov_collapse"] = (cov_l1.notna() & (cov_ratio < cov_drop_thr)).astype("int8")

# --- Leverage spike (debt-to-capital) ---
lev = pd.to_numeric(
    _as_series(
        df_events["sp_debt_to_capital"] if "sp_debt_to_capital" in df_events.columns else np.nan,
        df_events.index,
    ),
    errors="coerce",
)
lev_l1 = lag(lev, 1)

dlev = lev - lev_l1
df_events["lev_delta"] = dlev
dlev_q = dlev.loc[train_mask].quantile(0.95)
dlev_thr = float(dlev_q) if pd.notna(dlev_q) else 10.0  # percentage points
dlev_thr = max(dlev_thr, 5.0)  # minimum meaningful spike

df_events["evt_lev_spike"] = (dlev.notna() & (dlev >= dlev_thr)).astype("int8")

# --- Liquidity squeeze (current ratio and quick ratio) ---
act = pd.to_numeric(_as_series(df_events["act"] if "act" in df_events.columns else np.nan, df_events.index),
                    errors="coerce")
lct = pd.to_numeric(_as_series(df_events["lct"] if "lct" in df_events.columns else np.nan, df_events.index),
                    errors="coerce")
invt = pd.to_numeric(_as_series(df_events["invt"] if "invt" in df_events.columns else np.nan, df_events.index),
                     errors="coerce")

current_ratio = ratio(act, lct)
quick_ratio = ratio(act - invt, lct)

df_events["evt_liquidity_squeeze"] = (current_ratio.notna() & (current_ratio < 1.0)).astype("int8")
df_events["evt_quick_squeeze"] = (quick_ratio.notna() & (quick_ratio < 0.8)).astype("int8")

# --- Profitability / cash-flow stress ---
oibdp = pd.to_numeric(_as_series(df_events["oibdp"] if "oibdp" in df_events.columns else np.nan, df_events.index),
                      errors="coerce")
oancf = pd.to_numeric(_as_series(df_events["oancf"] if "oancf" in df_events.columns else np.nan, df_events.index),
                      errors="coerce")

oibdp_l1 = lag(oibdp, 1)
oancf_l1 = lag(oancf, 1)

# Large negative EBITDA shock: below 5th percentile of YoY change (TRAIN pool only)
doibdp = oibdp - oibdp_l1
doibdp_q = doibdp.loc[train_mask].quantile(0.05)

df_events["evt_ebitda_shock"] = (doibdp.notna() & (doibdp <= doibdp_q)).astype("int8")

# CFO turns negative or collapses
df_events["evt_cfo_negative"] = (oancf.notna() & (oancf < 0)).astype("int8")

cfo_ratio = ratio(oancf, oancf_l1)
cfo_drop_q = cfo_ratio.loc[train_mask].quantile(0.05)
cfo_drop_thr = float(cfo_drop_q) if pd.notna(cfo_drop_q) else 0.5
cfo_drop_thr = min(cfo_drop_thr, 0.7)

df_events["evt_cfo_collapse"] = (oancf_l1.notna() & (cfo_ratio < cfo_drop_thr)).astype("int8")

# =============================================================================
# 4.3 Event summary: prevalence + conditional distress rate
# =============================================================================
event_cols = [c for c in df_events.columns if c.startswith("evt_")]

_target_default = (
    "target_next_year_distress"
    if "target_next_year_distress" in df_events.columns
    else ("distress_dummy" if "distress_dummy" in df_events.columns else None)
)


def event_summary(df_in: pd.DataFrame, target: str | None = _target_default) -> pd.DataFrame:
    rows = []
    has_target = (target is not None) and (target in df_in.columns)

    for c in event_cols:
        n_events = int(df_in[c].sum())
        event_rate = float(df_in[c].mean())
        distress_rate = float(df_in.loc[df_in[c] == 1, target].mean()) if (n_events > 0 and has_target) else np.nan
        rows.append((c, n_events, event_rate, distress_rate))

    return (
        pd.DataFrame(rows, columns=["event", "n_events", "event_rate", "distress_rate_given_event"])
        .sort_values(["distress_rate_given_event", "n_events"], ascending=[False, False])
        .reset_index(drop=True)
    )


summary_pool = event_summary(df_events.loc[pool_mask].copy())
summary_all = event_summary(df_events)

print("Dividend cut threshold (pct change) used:", round(cut_threshold, 3))
print("Coverage collapse threshold (ratio) used:", round(cov_drop_thr, 3))
print("Leverage spike threshold (pp) used:", round(dlev_thr, 3))

display(summary_pool.head(15))

```

    Dividend cut threshold (pct change) used: -0.999
    Coverage collapse threshold (ratio) used: -0.69
    Leverage spike threshold (pp) used: 5.0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event</th>
      <th>n_events</th>
      <th>event_rate</th>
      <th>distress_rate_given_event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>evt_lev_spike</td>
      <td>1138</td>
      <td>0.020575</td>
      <td>0.747711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>evt_liquidity_squeeze</td>
      <td>15399</td>
      <td>0.278418</td>
      <td>0.436592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>evt_quick_squeeze</td>
      <td>16392</td>
      <td>0.296371</td>
      <td>0.416678</td>
    </tr>
    <tr>
      <th>3</th>
      <td>evt_cfo_negative</td>
      <td>20841</td>
      <td>0.376810</td>
      <td>0.348213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>evt_cov_breach</td>
      <td>27315</td>
      <td>0.493862</td>
      <td>0.341434</td>
    </tr>
    <tr>
      <th>5</th>
      <td>evt_cfo_collapse</td>
      <td>2335</td>
      <td>0.042217</td>
      <td>0.281644</td>
    </tr>
    <tr>
      <th>6</th>
      <td>evt_div_suspend</td>
      <td>1093</td>
      <td>0.019762</td>
      <td>0.267068</td>
    </tr>
    <tr>
      <th>7</th>
      <td>evt_ebitda_shock</td>
      <td>2195</td>
      <td>0.039686</td>
      <td>0.259722</td>
    </tr>
    <tr>
      <th>8</th>
      <td>evt_div_cut</td>
      <td>1746</td>
      <td>0.031568</td>
      <td>0.243227</td>
    </tr>
    <tr>
      <th>9</th>
      <td>evt_cov_collapse</td>
      <td>2331</td>
      <td>0.042145</td>
      <td>0.241171</td>
    </tr>
    <tr>
      <th>10</th>
      <td>evt_div_initiate</td>
      <td>4266</td>
      <td>0.077130</td>
      <td>0.206888</td>
    </tr>
  </tbody>
</table>
</div>



```python

# =============================================================================
# Train / Validation / Test Split (out-of-time)
#   - Split by label_year to respect the t → t+1 prediction structure
#   - Keep the last label year(s) inside the training pool as validation
# =============================================================================
# Use the observed-label sample for modeling to avoid silently conditioning on future existence
df_split = df_events[df_events["has_next_year_obs"] == 1].copy().reset_index(drop=True)

train_pool = df_split[df_split["label_year"] <= TRAIN_CUTOFF_LABEL_YEAR].copy()
test = df_split[df_split["label_year"] > TRAIN_CUTOFF_LABEL_YEAR].copy()

years = np.sort(train_pool["label_year"].dropna().unique())
val_years = years[-VAL_YEARS:] if len(years) else np.array([], dtype=int)

val = train_pool[train_pool["label_year"].isin(val_years)].copy()
train = train_pool[~train_pool["label_year"].isin(val_years)].copy()

print(
    "Split:",
    f"train={len(train):,}",
    f"val={len(val):,}",
    f"test={len(test):,}",
    "| val_years:",
    list(val_years),
)
```

    Split: train=44,783 val=6,415 test=12,404 | val_years: [np.int64(2022)]



```python

# =============================================================================
# Modeling-Ready Preprocessing (fit on TRAIN only)
#   - Handle infinities and remaining NaNs
#   - Winsorize continuous features using TRAIN quantile bounds
#   - Standardize continuous features to z-scores using TRAIN stats
#   - Keep binary (event) features as-is
# =============================================================================
continuous_feats = [
    "sp_debt_to_capital",
    "sp_ffo_to_debt",
    "sp_cfo_to_debt",
    "sp_focf_to_debt",
    "sp_dcf_to_debt",
    "sp_debt_to_ebitda",
    "sp_interest_coverage",
    "log_at",
    "log_mkvalt",
]
# Ensure they exist in all splits
continuous_feats = [c for c in continuous_feats if c in train.columns and c in val.columns and c in test.columns]

event_feats = []
if "event_cols" in globals():
    event_feats = [c for c in event_cols if c in train.columns and c in val.columns and c in test.columns]

# Total features to be used
all_feats = continuous_feats + event_feats

# Replace +/-inf with NaN
for d in (train, val, test):
    d[all_feats] = d[all_feats].replace([np.inf, -np.inf], np.nan)

# Impute NaNs (train-only medians)
fill = train[all_feats].median(numeric_only=True)
for d in (train, val, test):
    d[all_feats] = d[all_feats].fillna(fill)

# Winsorize ONLY continuous features
bounds = {}
for c in continuous_feats:
    s = pd.to_numeric(train[c], errors="coerce")
    bounds[c] = (s.quantile(WINSOR_LOWER_Q), s.quantile(WINSOR_UPPER_Q))

for d in (train, val, test):
    for c, (lo, hi) in bounds.items():
        s = pd.to_numeric(d[c], errors="coerce")
        d[c] = s.clip(lo, hi)

# Standardize (z-scores) ONLY continuous features
scaler = StandardScaler().fit(train[continuous_feats].to_numpy(dtype=float))

# Map to new columns
z_cols_cont = [f"z_{c}" for c in continuous_feats]
train[z_cols_cont] = scaler.transform(train[continuous_feats].to_numpy(dtype=float))
val[z_cols_cont] = scaler.transform(val[continuous_feats].to_numpy(dtype=float))
test[z_cols_cont] = scaler.transform(test[continuous_feats].to_numpy(dtype=float))

# Final model features: scaled continuous + raw binary events
MODEL_FEATS = z_cols_cont + event_feats
# Update z_cols for compatibility with later cells
z_cols = MODEL_FEATS
```


```python

# =============================================================================
# Diagnostics & Monitoring Proxies
#   - Correlation screen (TRAIN) for rough signal strength and sanity checks
#   - Expanding-window time folds for temporal stability checks
#   - Dataset overview (rows/firms/years/target rate) + target rate by year
#   - Distribution summaries, collinearity scan, and simple drift proxy (SMD) Train→Test
# =============================================================================
t = "target_next_year_distress"

feats = [c for c in (all_feats if "all_feats" in globals() else z_cols) if c in train.columns and c in test.columns]

corr = (
    train[[t] + feats]
    .corr(numeric_only=True)[t]
    .drop(t)
    .sort_values(key=np.abs, ascending=False)
)
print("Correlation with target:")
print(corr)


# Multicollinearity: Variance Inflation Factor (VIF)
def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data.sort_values("VIF", ascending=False)


vif_df = calculate_vif(train, z_cols)
print("\n=== Multicollinearity Diagnostic (VIF) ===")
print(vif_df)

folds = rolling_year_folds(train_pool, n_splits=N_SPLITS_TIME_CV, min_train_years=3)
for i, (tr_idx, va_idx, tr_years, va_year) in enumerate(folds, 1):
    print(
        f"Fold {i}: train_years={tr_years[0]}..{tr_years[-1]} (n={len(tr_idx)}), "
        f"val_year={va_year} (n={len(va_idx)})"
    )


def _overview(d: pd.DataFrame, name: str) -> None:
    n_rows = len(d)
    n_firms = d["firm_id"].nunique() if "firm_id" in d.columns else np.nan
    n_years = d["fyear"].nunique() if "fyear" in d.columns else np.nan
    target_rate = float(d[t].mean()) if t in d.columns else np.nan

    print(f"\n=== {name} === rows={n_rows:,} | firms={n_firms:,} | years={n_years} | target_rate={target_rate:.4f}")

    if "label_year" in d.columns:
        by_year = d.groupby("label_year")[t].agg(["mean", "count"])
        print("\nTarget by label_year (tail):")
        print(by_year.tail(12))


_overview(train, "TRAIN")
_overview(val, "VAL")
_overview(test, "TEST")

post_miss = pd.DataFrame({"col": raw})
post_miss["train_pct_na"] = [train[c].isna().mean() * 100 if c in train.columns else np.nan for c in raw]
post_miss["val_pct_na"] = [val[c].isna().mean() * 100 if c in val.columns else np.nan for c in raw]
post_miss["test_pct_na"] = [test[c].isna().mean() * 100 if c in test.columns else np.nan for c in raw]
if not post_miss.empty:
    post_miss = post_miss.sort_values("train_pct_na", ascending=False)
    print("\nPost-imputation missingness on raw inputs (pct):")
    print(post_miss.head(50).round(4))


def _dist(d: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    x = d[cols].replace([np.inf, -np.inf], np.nan)
    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    out = pd.DataFrame(
        {
            "n": x.notna().sum(),
            "mean": x.mean(),
            "std": x.std(ddof=0),
            "min": x.min(),
            "p01": q[0.01],
            "p05": q[0.05],
            "p25": q[0.25],
            "p50": q[0.50],
            "p75": q[0.75],
            "p95": q[0.95],
            "p99": q[0.99],
            "max": x.max(),
            "skew": x.skew(numeric_only=True),
            "kurt": x.kurtosis(numeric_only=True),
        }
    )
    print(f"\nDistribution summary ({name})")
    print(out.round(4).sort_values("skew", key=lambda s: s.abs(), ascending=False))
    return out


_ = _dist(train, feats, "TRAIN | winsorized raw feats")
_ = _dist(train, z_cols, "TRAIN | standardized feats")


def _hi_corr(d: pd.DataFrame, cols: list[str], thr: float = 0.80) -> list[tuple[str, str, float]]:
    cm = d[cols].corr(numeric_only=True)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = cm.iloc[i, j]
            if np.isfinite(r) and abs(r) >= thr:
                pairs.append((cols[i], cols[j], float(r)))
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


pairs = _hi_corr(train, feats, thr=0.80)
print("\nHigh collinearity pairs among feats (|corr|>=0.80) [top 25]:")
for a, b, r in pairs[:25]:
    print(f"{a} vs {b}: r={r:.3f}")


def _drift_smd(a_df: pd.DataFrame, b_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        a = pd.to_numeric(a_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(b_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        ma, mb = float(a.mean()), float(b.mean())
        sa, sb = float(a.std(ddof=0)), float(b.std(ddof=0))
        sp = np.sqrt(0.5 * (sa ** 2 + sb ** 2))
        smd = (mb - ma) / sp if sp > 0 else np.nan

        rows.append((c, ma, mb, smd, abs(smd) if np.isfinite(smd) else np.nan))

        out = pd.DataFrame(rows, columns=["feature", "mean_train", "mean_test", "smd", "abs_smd"])
    return out.sort_values("abs_smd", ascending=False)


drift = _drift_smd(train, test, feats)
print("\nTrain→Test drift (SMD) [top 15]:")
print(drift.head(15).round(4))


def _group_diff(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    g = d.groupby(t)[cols].mean(numeric_only=True)
    if 0 in g.index and 1 in g.index:
        return (g.loc[1] - g.loc[0]).sort_values(key=np.abs, ascending=False)
    return pd.Series(dtype="float64")


diff = _group_diff(train, feats)
if not diff.empty:
    print("\nMean difference (target=1 minus target=0) on TRAIN feats [top 15]:")
    print(diff.head(15).round(4))

```

    Correlation with target:
    sp_debt_to_capital       0.380218
    evt_liquidity_squeeze    0.263844
    evt_quick_squeeze        0.247242
    evt_cov_breach           0.195518
    log_at                  -0.170695
    evt_cfo_negative         0.167042
    evt_lev_spike            0.161332
    sp_interest_coverage    -0.152523
    log_mkvalt              -0.118970
    sp_cfo_to_debt          -0.075422
    sp_ffo_to_debt          -0.065969
    sp_focf_to_debt         -0.041535
    evt_div_initiate        -0.031913
    sp_debt_to_ebitda        0.027249
    sp_dcf_to_debt          -0.018879
    evt_cfo_collapse         0.014950
    evt_div_cut             -0.007761
    evt_ebitda_shock         0.007009
    evt_cov_collapse        -0.001902
    evt_div_suspend          0.001658
    Name: target_next_year_distress, dtype: float64
    
    === Multicollinearity Diagnostic (VIF) ===
                       feature        VIF
    3        z_sp_focf_to_debt  10.893009
    4         z_sp_dcf_to_debt   6.531875
    16       evt_quick_squeeze   3.845334
    12          evt_cov_breach   3.818077
    15   evt_liquidity_squeeze   3.810864
    2         z_sp_cfo_to_debt   3.420485
    18        evt_cfo_negative   3.232385
    9              evt_div_cut   2.584775
    10         evt_div_suspend   2.541363
    6   z_sp_interest_coverage   2.143007
    1         z_sp_ffo_to_debt   2.072122
    7                 z_log_at   1.379671
    8             z_log_mkvalt   1.332765
    17        evt_ebitda_shock   1.173054
    5      z_sp_debt_to_ebitda   1.141891
    11        evt_div_initiate   1.104041
    0     z_sp_debt_to_capital   1.081994
    14           evt_lev_spike   1.076851
    13        evt_cov_collapse   1.070668
    19        evt_cfo_collapse   1.066291
    Fold 1: train_years=2015..2017 (n=19775), val_year=2018 (n=6337)
    Fold 2: train_years=2015..2018 (n=26112), val_year=2019 (n=6173)
    Fold 3: train_years=2015..2019 (n=32285), val_year=2020 (n=6233)
    Fold 4: train_years=2015..2020 (n=38518), val_year=2021 (n=6265)
    
    === TRAIN === rows=44,783 | firms=9,220 | years=7 | target_rate=0.2527
    
    Target by label_year (tail):
                    mean  count
    label_year                 
    2015        0.261036   6773
    2016        0.249619   6570
    2017        0.237718   6432
    2018        0.240019   6337
    2019        0.276365   6173
    2020        0.277715   6233
    2021        0.226656   6265
    
    === VAL === rows=6,415 | firms=6,415 | years=1 | target_rate=0.2469
    
    Target by label_year (tail):
                    mean  count
    label_year                 
    2022        0.246921   6415
    
    === TEST === rows=12,404 | firms=6,633 | years=2 | target_rate=0.2627
    
    Target by label_year (tail):
                    mean  count
    label_year                 
    2023        0.266635   6327
    2024        0.258516   6077
    
    Post-imputation missingness on raw inputs (pct):
             col  train_pct_na  val_pct_na  test_pct_na
    48    ivstch       23.4218     30.1949      30.3773
    71      sppe       16.6559     16.3055      16.1641
    78    txditc       12.4668     11.9719      12.1251
    79       txp       10.9283     10.4910      10.7546
    60    prcc_f        8.6506      6.0483       7.1993
    59    prcc_c        8.6417      6.0171       7.1832
    61    prstkc        4.9997      4.4895       3.3538
    72     sppiv        4.9997      5.5495       5.9174
    46      ivch        3.4723      3.3359       3.5069
    6        aqc        3.3227      4.6921       4.5953
    45      ivao        3.2646      3.4451       3.7730
    19     dltis        2.4585      2.3071       1.9832
    24       dpc        2.4250      2.2136       2.1928
    20      dltr        2.0276      1.2627       1.6608
    73      sstk        1.8869      1.9174       2.0477
    25        dv        1.5899      1.4030       1.3463
    70       spi        1.4112      1.6680       1.8381
    28       dvt        0.8374      0.5924       0.4837
    26       dvc        0.8329      0.5924       0.4837
    9       capx        0.7079      0.4365       0.5160
    83     xidoc        0.6967      0.4677       0.4434
    3     aoloch        0.6677      0.4677       0.4353
    33      fopo        0.6319      0.4209       0.3950
    30      exre        0.5940      0.3118       0.2902
    31      fiao        0.4868      0.3118       0.2660
    43     ivaco        0.4712      0.2962       0.2660
    15     cstke        0.0089      0.0000       0.0000
    82      xido        0.0067      0.0000       0.0000
    22        do        0.0022      0.0000       0.0000
    81        xi        0.0022      0.0000       0.0000
    23        dp        0.0000      0.0000       0.0000
    14      cstk        0.0000      0.0000       0.0000
    16  datadate        0.0000      0.0000       0.0000
    10       ceq        0.0000      0.0000       0.0000
    11       che        0.0000      0.0000       0.0000
    2         ao        0.0000      0.0000       0.0000
    1        act        0.0000      0.0000       0.0000
    4         ap        0.0000      0.0000       0.0000
    5     apalch        0.0000      0.0000       0.0000
    7         at        0.0000      0.0000       0.0000
    8       caps        0.0000      0.0000       0.0000
    0        aco        0.0000      0.0000       0.0000
    21      dltt        0.0000      0.0000       0.0000
    18     dlcch        0.0000      0.0000       0.0000
    17       dlc        0.0000      0.0000       0.0000
    12     chech        0.0000      0.0000       0.0000
    13      csho        0.0000      0.0000       0.0000
    38       ibc        0.0000      0.0000       0.0000
    37     ibadj        0.0000      0.0000       0.0000
    40     invch        0.0000      0.0000       0.0000
    
    Distribution summary (TRAIN | winsorized raw feats)
                               n      mean       std        min        p01  \
    evt_div_suspend        44783    0.0183    0.1339     0.0000     0.0000   
    sp_dcf_to_debt         44783  -26.1977  206.7528 -1724.5155 -1724.0928   
    evt_lev_spike          44783    0.0200    0.1400     0.0000     0.0000   
    sp_debt_to_ebitda      44783  130.0003  728.4237     0.0000     0.0000   
    evt_div_cut            44783    0.0302    0.1712     0.0000     0.0000   
    sp_ffo_to_debt         44783  -14.8417  231.4420 -1852.1280 -1850.7110   
    sp_focf_to_debt        44783  -13.6823  181.8769 -1436.7039 -1435.7623   
    evt_cfo_collapse       44783    0.0390    0.1937     0.0000     0.0000   
    evt_cov_collapse       44783    0.0394    0.1946     0.0000     0.0000   
    evt_ebitda_shock       44783    0.0406    0.1973     0.0000     0.0000   
    evt_div_initiate       44783    0.0837    0.2769     0.0000     0.0000   
    sp_debt_to_capital     44783    0.3837    0.4446     0.0000     0.0000   
    evt_liquidity_squeeze  44783    0.2754    0.4467     0.0000     0.0000   
    evt_quick_squeeze      44783    0.2932    0.4552     0.0000     0.0000   
    log_mkvalt             44783    7.0526    4.0063     0.4260     0.4260   
    log_at                 44783   11.0730    3.5650     0.6931     0.6931   
    evt_cfo_negative       44783    0.3598    0.4799     0.0000     0.0000   
    sp_interest_coverage   44783    0.3670    3.0059    -4.6151    -4.6151   
    sp_cfo_to_debt         44783    4.8503  158.0296  -954.5940  -954.3091   
    evt_cov_breach         44783    0.4815    0.4997     0.0000     0.0000   
    
                               p05     p25      p50      p75      p95        p99  \
    evt_div_suspend         0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    sp_dcf_to_debt        -34.0200 -0.2690  -0.0162   0.0922   3.1968   357.6861   
    evt_lev_spike           0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    sp_debt_to_ebitda       0.0000  1.6697   2.5050   3.5542  31.1958  5667.0715   
    evt_div_cut             0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    sp_ffo_to_debt        -22.0781 -0.1073   0.0984   0.2931  12.7810   784.0926   
    sp_focf_to_debt       -20.7545 -0.1556   0.0470   0.1927   7.2503   613.8415   
    evt_cfo_collapse        0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    evt_cov_collapse        0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    evt_ebitda_shock        0.0000  0.0000   0.0000   0.0000   0.0000     1.0000   
    evt_div_initiate        0.0000  0.0000   0.0000   0.0000   1.0000     1.0000   
    sp_debt_to_capital      0.0000  0.0150   0.2890   0.5571   1.0013     2.7859   
    evt_liquidity_squeeze   0.0000  0.0000   0.0000   1.0000   1.0000     1.0000   
    evt_quick_squeeze       0.0000  0.0000   0.0000   1.0000   1.0000     1.0000   
    log_mkvalt              1.5667  4.1728   6.3628   9.0370  15.6682    16.4558   
    log_at                  4.3694  8.8541  11.4487  13.7914  16.0542    17.8535   
    evt_cfo_negative        0.0000  0.0000   0.0000   1.0000   1.0000     1.0000   
    sp_interest_coverage   -4.6151 -2.0107   1.1031   2.5024   4.6151     4.6151   
    sp_cfo_to_debt         -7.6060  0.0001   0.1333   0.3443  17.9883   944.6857   
    evt_cov_breach          0.0000  0.0000   0.0000   1.0000   1.0000     1.0000   
    
                                 max    skew     kurt  
    evt_div_suspend           1.0000  7.1951  49.7712  
    sp_dcf_to_debt          357.7370 -6.8666  51.2077  
    evt_lev_spike             1.0000  6.8560  45.0066  
    sp_debt_to_ebitda      5667.1637  6.4206  41.9768  
    evt_div_cut               1.0000  5.4893  28.1335  
    sp_ffo_to_debt          784.2414 -5.4265  44.3318  
    sp_focf_to_debt         614.0150 -5.3263  42.4355  
    evt_cfo_collapse          1.0000  4.7604  20.6626  
    evt_cov_collapse          1.0000  4.7330  20.4019  
    evt_ebitda_shock          1.0000  4.6573  19.6913  
    evt_div_initiate          1.0000  3.0067   7.0408  
    sp_debt_to_capital        2.7876  2.4008   8.9148  
    evt_liquidity_squeeze     1.0000  1.0055  -0.9890  
    evt_quick_squeeze         1.0000  0.9085  -1.1747  
    log_mkvalt               16.4558  0.7284  -0.1123  
    log_at                   17.8537 -0.6152   0.1416  
    evt_cfo_negative          1.0000  0.5844  -1.6586  
    sp_interest_coverage      4.6151 -0.4038  -0.9945  
    sp_cfo_to_debt          944.8613  0.2324  27.9858  
    evt_cov_breach            1.0000  0.0741  -1.9946  
    
    Distribution summary (TRAIN | standardized feats)
                                n    mean     std     min     p01     p05     p25  \
    evt_div_suspend         44783  0.0183  0.1339  0.0000  0.0000  0.0000  0.0000   
    z_sp_dcf_to_debt        44783  0.0000  1.0000 -8.2142 -8.2122 -0.0378  0.1254   
    evt_lev_spike           44783  0.0200  0.1400  0.0000  0.0000  0.0000  0.0000   
    z_sp_debt_to_ebitda     44783 -0.0000  1.0000 -0.1785 -0.1785 -0.1785 -0.1762   
    evt_div_cut             44783  0.0302  0.1712  0.0000  0.0000  0.0000  0.0000   
    z_sp_ffo_to_debt        44783  0.0000  1.0000 -7.9384 -7.9323 -0.0313  0.0637   
    z_sp_focf_to_debt       44783 -0.0000  1.0000 -7.8241 -7.8189 -0.0389  0.0744   
    evt_cfo_collapse        44783  0.0390  0.1937  0.0000  0.0000  0.0000  0.0000   
    evt_cov_collapse        44783  0.0394  0.1946  0.0000  0.0000  0.0000  0.0000   
    evt_ebitda_shock        44783  0.0406  0.1973  0.0000  0.0000  0.0000  0.0000   
    evt_div_initiate        44783  0.0837  0.2769  0.0000  0.0000  0.0000  0.0000   
    z_sp_debt_to_capital    44783 -0.0000  1.0000 -0.8628 -0.8628 -0.8628 -0.8291   
    evt_liquidity_squeeze   44783  0.2754  0.4467  0.0000  0.0000  0.0000  0.0000   
    evt_quick_squeeze       44783  0.2932  0.4552  0.0000  0.0000  0.0000  0.0000   
    z_log_mkvalt            44783  0.0000  1.0000 -1.6541 -1.6540 -1.3693 -0.7188   
    z_log_at                44783  0.0000  1.0000 -2.9116 -2.9116 -1.8804 -0.6224   
    evt_cfo_negative        44783  0.3598  0.4799  0.0000  0.0000  0.0000  0.0000   
    z_sp_interest_coverage  44783 -0.0000  1.0000 -1.6574 -1.6574 -1.6574 -0.7910   
    z_sp_cfo_to_debt        44783  0.0000  1.0000 -6.0713 -6.0695 -0.0788 -0.0307   
    evt_cov_breach          44783  0.4815  0.4997  0.0000  0.0000  0.0000  0.0000   
    
                               p50     p75     p95     p99     max    skew  \
    evt_div_suspend         0.0000  0.0000  0.0000  1.0000  1.0000  7.1951   
    z_sp_dcf_to_debt        0.1266  0.1272  0.1422  1.8567  1.8570 -6.8666   
    evt_lev_spike           0.0000  0.0000  0.0000  1.0000  1.0000  6.8560   
    z_sp_debt_to_ebitda    -0.1750 -0.1736 -0.1356  7.6014  7.6016  6.4206   
    evt_div_cut             0.0000  0.0000  0.0000  1.0000  1.0000  5.4893   
    z_sp_ffo_to_debt        0.0646  0.0654  0.1194  3.4520  3.4526 -5.4265   
    z_sp_focf_to_debt       0.0755  0.0763  0.1151  3.4503  3.4512 -5.3263   
    evt_cfo_collapse        0.0000  0.0000  0.0000  1.0000  1.0000  4.7604   
    evt_cov_collapse        0.0000  0.0000  0.0000  1.0000  1.0000  4.7330   
    evt_ebitda_shock        0.0000  0.0000  0.0000  1.0000  1.0000  4.6573   
    evt_div_initiate        0.0000  0.0000  1.0000  1.0000  1.0000  3.0067   
    z_sp_debt_to_capital   -0.2129  0.3900  1.3891  5.4026  5.4065  2.4008   
    evt_liquidity_squeeze   0.0000  1.0000  1.0000  1.0000  1.0000  1.0055   
    evt_quick_squeeze       0.0000  1.0000  1.0000  1.0000  1.0000  0.9085   
    z_log_mkvalt           -0.1722  0.4953  2.1505  2.3471  2.3471  0.7284   
    z_log_at                0.1054  0.7625  1.3973  1.9020  1.9020 -0.6152   
    evt_cfo_negative        0.0000  1.0000  1.0000  1.0000  1.0000  0.5844   
    z_sp_interest_coverage  0.2449  0.7104  1.4132  1.4132  1.4132 -0.4038   
    z_sp_cfo_to_debt       -0.0298 -0.0285  0.0831  5.9472  5.9483  0.2324   
    evt_cov_breach          0.0000  1.0000  1.0000  1.0000  1.0000  0.0741   
    
                               kurt  
    evt_div_suspend         49.7712  
    z_sp_dcf_to_debt        51.2077  
    evt_lev_spike           45.0066  
    z_sp_debt_to_ebitda     41.9768  
    evt_div_cut             28.1335  
    z_sp_ffo_to_debt        44.3318  
    z_sp_focf_to_debt       42.4355  
    evt_cfo_collapse        20.6626  
    evt_cov_collapse        20.4019  
    evt_ebitda_shock        19.6913  
    evt_div_initiate         7.0408  
    z_sp_debt_to_capital     8.9148  
    evt_liquidity_squeeze   -0.9890  
    evt_quick_squeeze       -1.1747  
    z_log_mkvalt            -0.1123  
    z_log_at                 0.1416  
    evt_cfo_negative        -1.6586  
    z_sp_interest_coverage  -0.9945  
    z_sp_cfo_to_debt        27.9858  
    evt_cov_breach          -1.9946  
    
    High collinearity pairs among feats (|corr|>=0.80) [top 25]:
    sp_focf_to_debt vs sp_dcf_to_debt: r=0.899
    sp_interest_coverage vs evt_cov_breach: r=-0.822
    
    Train→Test drift (SMD) [top 15]:
                      feature  mean_train  mean_test     smd  abs_smd
    11       evt_div_initiate      0.0837     0.0322 -0.2215   0.2215
    2          sp_cfo_to_debt      4.8503    -8.2457 -0.0923   0.0923
    7                  log_at     11.0730    11.3644  0.0836   0.0836
    18       evt_cfo_negative      0.3598     0.3968  0.0764   0.0764
    19       evt_cfo_collapse      0.0390     0.0539  0.0708   0.0708
    0      sp_debt_to_capital      0.3837     0.4146  0.0699   0.0699
    17       evt_ebitda_shock      0.0406     0.0507  0.0486   0.0486
    15  evt_liquidity_squeeze      0.2754     0.2583 -0.0387   0.0387
    8              log_mkvalt      7.0526     7.2055  0.0377   0.0377
    13       evt_cov_collapse      0.0394     0.0465  0.0349   0.0349
    16      evt_quick_squeeze      0.2932     0.2793 -0.0309   0.0309
    1          sp_ffo_to_debt    -14.8417   -20.7313 -0.0273   0.0273
    3         sp_focf_to_debt    -13.6823   -18.3124 -0.0271   0.0271
    5       sp_debt_to_ebitda    130.0003   136.9402  0.0093   0.0093
    12         evt_cov_breach      0.4815     0.4854  0.0079   0.0079
    
    Mean difference (target=1 minus target=0) on TRAIN feats [top 15]:
    sp_debt_to_ebitda        45.6783
    sp_ffo_to_debt          -35.1363
    sp_cfo_to_debt          -27.4287
    sp_focf_to_debt         -17.3843
    sp_dcf_to_debt           -8.9825
    log_at                   -1.4004
    log_mkvalt               -1.0969
    sp_interest_coverage     -1.0551
    sp_debt_to_capital        0.3891
    evt_liquidity_squeeze     0.2712
    evt_quick_squeeze         0.2590
    evt_cov_breach            0.2248
    evt_cfo_negative          0.1845
    evt_lev_spike             0.0520
    evt_div_initiate         -0.0203
    dtype: float64



```python
# =============================================================================
# Visual EDA: Feature distributions by distress flag
#   - Quick separation check: do distressed vs non-distressed firms differ in levels?
#   - Uses a horizontal boxplot for comparability across features
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Objective: visualize feature distribution differences for distressed vs non-distressed observations.

plot_df = train_pool.copy() if "train_pool" in globals() else df.copy()

flag_col = "distress_dummy" if "distress_dummy" in plot_df.columns else (
    "target_next_year_distress" if "target_next_year_distress" in plot_df.columns else None
)
if flag_col is None:
    raise KeyError("No distress flag found. Expected 'distress_dummy' or 'target_next_year_distress' in the data.")

plot_feats = [c for c in (feats if "feats" in globals() else []) if c in plot_df.columns]
if not plot_feats:
    plot_feats = [c for c in (z_cols if "z_cols" in globals() else []) if c in plot_df.columns]
if not plot_feats:
    raise KeyError(
        "No feature columns found to plot. Expected 'feats' or 'z_cols' to exist and be present in the data.")

tmp = plot_df[[flag_col] + plot_feats].copy()
tmp[flag_col] = pd.to_numeric(tmp[flag_col], errors="coerce").astype("Int64")
tmp = tmp[tmp[flag_col].isin([0, 1])].copy()

long = tmp.melt(id_vars=[flag_col], value_vars=plot_feats, var_name="feature", value_name="value")
long["value"] = pd.to_numeric(long["value"], errors="coerce")
long = long.dropna(subset=["value"])

fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(plot_feats))))
sns.boxplot(
    data=long,
    x="value",
    y="feature",
    hue=flag_col,
    orient="h",
    showfliers=False,
    ax=ax,
)
ax.set_title(f"Feature distributions by {flag_col} (0=No distress, 1=Distress)")
ax.set_xlabel("Feature value")
ax.set_ylabel("")
ax.legend(title=flag_col, loc="best")
plt.tight_layout()
plt.show()
```


    
![png](v28_12_2_adjusted_files/v28_12_2_adjusted_7_0.png)
    



```python
# =============================================================================
# Sanity checks on the distress proxy
#   - Component prevalence (each condition and joint condition)
#   - Distress rate by firm size decile (log assets) to check monotonic patterns
# =============================================================================

print(pd.Series({
    "hl_ffo": hl_ffo.mean(),
    "hl_cap": hl_cap.mean(),
    "hl_deb": hl_deb.mean(),
    "hl_all": is_highly_leveraged.mean()
}))
df["size_decile"] = pd.qcut(df["log_at"], 10, duplicates="drop")
print(df.groupby("size_decile")["distress_dummy"].mean())
```

    hl_ffo    0.495567
    hl_cap    0.316846
    hl_deb    0.482848
    hl_all    0.229065
    dtype: float64
    size_decile
    (-0.001, 6.528]     0.523391
    (6.528, 8.356]      0.351160
    (8.356, 9.487]      0.266898
    (9.487, 10.488]     0.206933
    (10.488, 11.476]    0.195174
    (11.476, 12.507]    0.196933
    (12.507, 13.39]     0.178533
    (13.39, 14.213]     0.208106
    (14.213, 15.218]    0.227733
    (15.218, 21.886]    0.225703
    Name: distress_dummy, dtype: float64


    /var/folders/p1/_cwwbdbj51q1lwpynfnzdxpm0000gn/T/ipykernel_7676/2342790164.py:14: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      print(df.groupby("size_decile")["distress_dummy"].mean())


### 5.5 Endogeneity & Simultaneity Bias: A Critical Note
As highlighted by recent literature (e.g., Roberts 2015, Jiang 2017), modeling capital structure as a predictor of credit outcomes is prone to **endogeneity**. 

1. **Simultaneity**: Firms choose their leverage (Debt/Capital, etc.) based on their expected future credit risk and investment opportunities. Thus, leverage and distress outcomes are determined simultaneously in equilibrium.
2. **Reverse Causality**: While high leverage may lead to distress, the *anticipation* of distress may also force firms to seek more debt (if they follow pecking order) or prevent them from accessing debt (if credit constrained).

**Mitigation in this notebook:**
- While we use a standard Logistic Regression for transparency and baseline benchmarking, we acknowledge that the coefficients should be interpreted as **predictive associations** rather than pure causal effects.
- Future iterations should explore **Instrumental Variables (IV)** or **GMM** estimation (e.g., using industry-median leverage or historical tax changes as instruments) to better identify the causal impact of leverage on distress.



```python
# =============================================================================
# Persistence Benchmark (Economic Sanity Check)
#   - Does the ML model add value over simply assuming distress persists?
#   - We use 'distress_dummy' (current year) to predict 'target_next_year_distress'
# =============================================================================
from sklearn.metrics import f1_score, accuracy_score

for name, d in [("VAL", val), ("TEST", test)]:
    y_true = d[TARGET_COL].astype(int)
    y_bench = d["distress_dummy"].astype(int)

    auc_bench = roc_auc_score(y_true, y_bench)
    f1_bench = f1_score(y_true, y_bench)
    acc_bench = accuracy_score(y_true, y_bench)

    print(f"--- Persistence Benchmark ({name}) ---")
    print(f"AUC: {auc_bench:.4f} | F1: {f1_bench:.4f} | Accuracy: {acc_bench:.4f}")

```

    --- Persistence Benchmark (VAL) ---
    AUC: 0.7862 | F1: 0.6860 | Accuracy: 0.8514
    --- Persistence Benchmark (TEST) ---
    AUC: 0.7901 | F1: 0.6949 | Accuracy: 0.8439


## 6. Modeling: Baseline out-of-sample supervised learning

This section trains a simple, interpretable classifier on the **TRAIN** split, selects a small hyperparameter setting using the **VAL** split, and reports final performance on the **TEST** split (held out, label_year > train cutoff).  

**Goal:** Predict `target_next_year_distress` one year ahead from standardized financial-ratio features.



```python
# =============================================================================
# 6.1 Setup: Features, target, and train/val/test matrices
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# Prefer standardized feature columns created earlier (fit on TRAIN only)
if "z_cols" in globals():
    MODEL_FEATS = list(z_cols)
else:
    MODEL_FEATS = [f"z_{c}" for c in feats]  # fallback

# Basic sanity checks
for df_name, df_ in [("train", train), ("val", val), ("test", test)]:
    missing_feats = [c for c in MODEL_FEATS if c not in df_.columns]
    if missing_feats:
        raise KeyError(f"{df_name}: missing feature columns: {missing_feats}")

X_train = train[MODEL_FEATS].to_numpy(dtype=float)
y_train = train[TARGET_COL].astype(int).to_numpy()

X_val = val[MODEL_FEATS].to_numpy(dtype=float)
y_val = val[TARGET_COL].astype(int).to_numpy()

X_test = test[MODEL_FEATS].to_numpy(dtype=float)
y_test = test[TARGET_COL].astype(int).to_numpy()

# Defensive: ensure model inputs are finite
def _assert_finite(name, X):
    bad = ~np.isfinite(X)
    if bad.any():
        rows, cols = np.where(bad)
        raise ValueError(f"{name}: found non-finite values at {len(rows)} cells (e.g., row={rows[0]}, col={MODEL_FEATS[cols[0]]}).")

_assert_finite("X_train", X_train)
_assert_finite("X_val", X_val)
_assert_finite("X_test", X_test)

print("Modeling matrix shapes:")
print(f"  X_train: {X_train.shape} | y_train mean={y_train.mean():.4f}")
print(f"  X_val:   {X_val.shape} | y_val mean={y_val.mean():.4f}")
print(f"  X_test:  {X_test.shape} | y_test mean={y_test.mean():.4f}")
```

    Modeling matrix shapes:
      X_train: (44783, 20) | y_train mean=0.2527
      X_val:   (6415, 20) | y_val mean=0.2469
      X_test:  (12404, 20) | y_test mean=0.2627



```python
# =============================================================================
# 6.2 Baseline model: Logistic Regression (tuned on VAL)
#   - Interpretable, strong baseline for tabular finance ratios
#   - We tune C on VAL; you can expand the grid later if needed
# =============================================================================
C_GRID = [0.01, 0.1, 1.0, 10.0]

best = {"C": None, "val_auc": -np.inf, "model": None}

for C in C_GRID:
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",  # practical default with ~10% positives
        n_jobs=None,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    if val_auc > best["val_auc"]:
        best.update({"C": C, "val_auc": val_auc, "model": clf})

print(f"Best LogisticRegression on VAL: C={best['C']} | AUC={best['val_auc']:.4f}")
logit = best["model"]

```

    Best LogisticRegression on VAL: C=1.0 | AUC=0.8197



```python
# =============================================================================
# 6.2b Statistical Inference & Economic Interpretation (Statsmodels)
#   - sklearn is great for prediction but lacks p-values and inference.
#   - We re-estimate the logit using statsmodels to audit statistical significance.
#   - We also report Marginal Effects to see economic significance.
# =============================================================================
import statsmodels.api as sm

# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
# Panel dependence: use cluster-robust SEs (firm-level) for inference in the training panel
try:
    _groups = train["firm_id"].to_numpy()
    sm_model = sm.Logit(y_train, X_train_sm).fit(
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": _groups},
    )
    _se_note = "cluster-robust SEs (firm-level)"
except Exception as e:
    # Fallback: still produce estimates, but avoid overstating inference quality
    sm_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
    _se_note = f"non-robust SEs (fallback; clustering failed: {e})"

print(f"\n=== Statsmodels Logistic Regression Summary ({_se_note}) ===")
print(sm_model.summary(xname=["const"] + MODEL_FEATS))

# Marginal Effects at the Mean (MEM)
try:
    mfx = sm_model.get_margeff(at="mean")
    print("\n=== Economic Significance: Marginal Effects at the Mean ===")
    print(mfx.summary())
except Exception as e:
    print(f"\nCould not calculate marginal effects: {e}")

# AIC / BIC / Pseudo-R2
print(f"\nModel Fit:")
print(f"  Pseudo R2 (McFadden): {sm_model.prsquared:.4f}")
print(f"  AIC: {sm_model.aic:.2f}")
print(f"  BIC: {sm_model.bic:.2f}")

```

    
    === Statsmodels Logistic Regression Summary (cluster-robust SEs (firm-level)) ===
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                44783
    Model:                          Logit   Df Residuals:                    44762
    Method:                           MLE   Df Model:                           20
    Date:                Sun, 28 Dec 2025   Pseudo R-squ.:                  0.2182
    Time:                        22:14:35   Log-Likelihood:                -19789.
    converged:                       True   LL-Null:                       -25313.
    Covariance Type:              cluster   LLR p-value:                     0.000
    ==========================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    const                     -2.0221      0.042    -48.708      0.000      -2.103      -1.941
    z_sp_debt_to_capital       1.1035      0.027     40.297      0.000       1.050       1.157
    z_sp_ffo_to_debt          -0.0768      0.019     -4.119      0.000      -0.113      -0.040
    z_sp_cfo_to_debt          -0.0601      0.039     -1.526      0.127      -0.137       0.017
    z_sp_focf_to_debt          0.0458      0.074      0.619      0.536      -0.099       0.191
    z_sp_dcf_to_debt          -0.0037      0.057     -0.064      0.949      -0.115       0.108
    z_sp_debt_to_ebitda       -0.0259      0.013     -2.012      0.044      -0.051      -0.001
    z_sp_interest_coverage     0.0268      0.030      0.890      0.374      -0.032       0.086
    z_log_at                  -0.2244      0.019    -11.932      0.000      -0.261      -0.188
    z_log_mkvalt              -0.0615      0.019     -3.201      0.001      -0.099      -0.024
    evt_div_cut                0.0081      0.120      0.068      0.946      -0.227       0.243
    evt_div_suspend            0.0041      0.154      0.027      0.979      -0.297       0.305
    evt_div_initiate           0.1322      0.044      3.024      0.002       0.047       0.218
    evt_cov_breach             0.6455      0.060     10.838      0.000       0.529       0.762
    evt_cov_collapse          -0.1616      0.071     -2.279      0.023      -0.301      -0.023
    evt_lev_spike              1.3859      0.094     14.714      0.000       1.201       1.571
    evt_liquidity_squeeze      0.5905      0.052     11.390      0.000       0.489       0.692
    evt_quick_squeeze          0.2451      0.052      4.756      0.000       0.144       0.346
    evt_ebitda_shock           0.1976      0.071      2.764      0.006       0.057       0.338
    evt_cfo_negative           0.3478      0.047      7.353      0.000       0.255       0.441
    evt_cfo_collapse           0.1739      0.067      2.609      0.009       0.043       0.305
    ==========================================================================================
    
    === Economic Significance: Marginal Effects at the Mean ===
            Logit Marginal Effects       
    =====================================
    Dep. Variable:                      y
    Method:                          dydx
    At:                              mean
    ==============================================================================
                    dy/dx    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.1853      0.005     36.648      0.000       0.175       0.195
    x2            -0.0129      0.003     -4.124      0.000      -0.019      -0.007
    x3            -0.0101      0.007     -1.527      0.127      -0.023       0.003
    x4             0.0077      0.012      0.619      0.536      -0.017       0.032
    x5            -0.0006      0.010     -0.064      0.949      -0.019       0.018
    x6            -0.0043      0.002     -2.012      0.044      -0.009      -0.000
    x7             0.0045      0.005      0.890      0.374      -0.005       0.014
    x8            -0.0377      0.003    -12.006      0.000      -0.044      -0.032
    x9            -0.0103      0.003     -3.208      0.001      -0.017      -0.004
    x10            0.0014      0.020      0.068      0.946      -0.038       0.041
    x11            0.0007      0.026      0.027      0.979      -0.050       0.051
    x12            0.0222      0.007      3.026      0.002       0.008       0.037
    x13            0.1084      0.010     10.934      0.000       0.089       0.128
    x14           -0.0271      0.012     -2.277      0.023      -0.050      -0.004
    x15            0.2327      0.016     14.505      0.000       0.201       0.264
    x16            0.0992      0.009     11.331      0.000       0.082       0.116
    x17            0.0412      0.009      4.765      0.000       0.024       0.058
    x18            0.0332      0.012      2.762      0.006       0.010       0.057
    x19            0.0584      0.008      7.352      0.000       0.043       0.074
    x20            0.0292      0.011      2.607      0.009       0.007       0.051
    ==============================================================================
    
    Model Fit:
      Pseudo R2 (McFadden): 0.2182
      AIC: 39620.55
      BIC: 39803.45



```python
# =============================================================================
# 6.2c Temporal Stability (Walk-Forward Validation)
#   - We assess whether model performance is stable across different time regimes.
#   - Using the expanding-window folds defined in diagnostics.
# =============================================================================
temporal_results = []

FOLD_FEATS = [c for c in feats if c in train_pool.columns]
if not FOLD_FEATS:
    raise KeyError("No usable feature columns found in train_pool for temporal CV. Expected 'feats' columns to exist.")

for i, (tr_idx, va_idx, tr_years, va_year) in enumerate(folds, 1):
    tr_df = train_pool.loc[tr_idx, FOLD_FEATS + [TARGET_COL]].copy()
    va_df = train_pool.loc[va_idx, FOLD_FEATS + [TARGET_COL]].copy()

    # Targets
    y_tr = pd.to_numeric(tr_df[TARGET_COL], errors="coerce").astype(int).to_numpy()
    y_va = pd.to_numeric(va_df[TARGET_COL], errors="coerce").astype(int).to_numpy()

    # Features: fold-local preprocessing (no leakage)
    X_tr_df = tr_df[FOLD_FEATS].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    X_va_df = va_df[FOLD_FEATS].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")

    # Median imputation fit on fold-train only
    fold_fill = X_tr_df.median(numeric_only=True)
    X_tr_df = X_tr_df.fillna(fold_fill)
    X_va_df = X_va_df.fillna(fold_fill)

    # Winsorize using fold-train only
    fold_bounds = {}
    for c in FOLD_FEATS:
        s = pd.to_numeric(X_tr_df[c], errors="coerce")
        fold_bounds[c] = (s.quantile(WINSOR_LOWER_Q), s.quantile(WINSOR_UPPER_Q))

    for c, (lo, hi) in fold_bounds.items():
        X_tr_df[c] = pd.to_numeric(X_tr_df[c], errors="coerce").clip(lo, hi)
        X_va_df[c] = pd.to_numeric(X_va_df[c], errors="coerce").clip(lo, hi)

    # Standardize fit on fold-train only
    fold_scaler = StandardScaler().fit(X_tr_df.to_numpy(dtype=float))
    X_tr = fold_scaler.transform(X_tr_df.to_numpy(dtype=float))
    X_va = fold_scaler.transform(X_va_df.to_numpy(dtype=float))

    # Fit model (using best C from tuning)
    fold_clf = LogisticRegression(C=best["C"], solver="lbfgs", max_iter=500, class_weight="balanced", random_state=42)
    fold_clf.fit(X_tr, y_tr)

    # Evaluate
    probs = fold_clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, probs)
    ap = average_precision_score(y_va, probs)

    temporal_results.append({
        "Fold": i,
        "Val_Year": va_year,
        "Train_End": tr_years[-1],
        "AUC": auc,
        "PR-AUC": ap
    })

temporal_df = pd.DataFrame(temporal_results)
print("\n=== Temporal Stability: Walk-Forward Results ===")
print(temporal_df.round(4))
print(f"\nAverage AUC across folds: {temporal_df['AUC'].mean():.4f}")

```

    
    === Temporal Stability: Walk-Forward Results ===
       Fold  Val_Year  Train_End     AUC  PR-AUC
    0     1      2018       2017  0.8428  0.6238
    1     2      2019       2018  0.8193  0.6360
    2     3      2020       2019  0.8045  0.5890
    3     4      2021       2020  0.8218  0.5466
    
    Average AUC across folds: 0.8221



```python
# =============================================================================
# 6.3 Evaluation: VAL and TEST (AUC, PR-AUC, Brier) + thresholding
# =============================================================================
def evaluate_split(name, y_true, y_proba, threshold=0.5):
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"AUC (ROC): {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Brier score (calibration): {brier:.4f}")
    print(f"Threshold: {threshold:.3f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Explicitly pull out Sensitivity and Specificity
    # TN FP
    # FN TP
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity (Recall of Distress): {sensitivity:.4f}")
    print(f"Specificity (Recall of Non-Distress): {specificity:.4f}")
    
    return {"auc": auc, "ap": ap, "brier": brier, "cm": cm, "sens": sensitivity, "spec": specificity}

val_proba = logit.predict_proba(X_val)[:, 1]
test_proba = logit.predict_proba(X_test)[:, 1]

# Choose an operational threshold using VAL (maximize F1 as a simple practical rule)
prec, rec, thr = precision_recall_curve(y_val, val_proba)
f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1[1:])  # skip first point (threshold undefined)
best_thr = thr[best_idx]

_ = evaluate_split("VAL (threshold from VAL-F1)", y_val, val_proba, threshold=float(best_thr))
_ = evaluate_split("TEST (same threshold)", y_test, test_proba, threshold=float(best_thr))

```

    
    === VAL (threshold from VAL-F1) ===
    AUC (ROC): 0.8197
    Average Precision (PR-AUC): 0.6079
    Brier score (calibration): 0.1656
    Threshold: 0.512
    Confusion matrix [ [TN FP] [FN TP] ]:
    [[3958  873]
     [ 490 1094]]
    
    Classification report:
                  precision    recall  f1-score   support
    
               0     0.8898    0.8193    0.8531      4831
               1     0.5562    0.6907    0.6162      1584
    
        accuracy                         0.7875      6415
       macro avg     0.7230    0.7550    0.7346      6415
    weighted avg     0.8075    0.7875    0.7946      6415
    
    Sensitivity (Recall of Distress): 0.6907
    Specificity (Recall of Non-Distress): 0.8193
    
    === TEST (same threshold) ===
    AUC (ROC): 0.8244
    Average Precision (PR-AUC): 0.6199
    Brier score (calibration): 0.1730
    Threshold: 0.512
    Confusion matrix [ [TN FP] [FN TP] ]:
    [[7209 1937]
     [ 916 2342]]
    
    Classification report:
                  precision    recall  f1-score   support
    
               0     0.8873    0.7882    0.8348      9146
               1     0.5473    0.7188    0.6215      3258
    
        accuracy                         0.7700     12404
       macro avg     0.7173    0.7535    0.7281     12404
    weighted avg     0.7980    0.7700    0.7788     12404
    
    Sensitivity (Recall of Distress): 0.7188
    Specificity (Recall of Non-Distress): 0.7882



```python
# =============================================================================
# 6.4 Diagnostic plots: ROC and Precision-Recall (VAL vs TEST)
# =============================================================================
def plot_roc(y_true, y_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.show()

def plot_pr(y_true, y_proba, title):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.show()

plot_roc(y_val, val_proba, "ROC Curve (VAL) - Logistic Regression")
plot_roc(y_test, test_proba, "ROC Curve (TEST) - Logistic Regression")

plot_pr(y_val, val_proba, "Precision-Recall Curve (VAL) - Logistic Regression")
plot_pr(y_test, test_proba, "Precision-Recall Curve (TEST) - Logistic Regression")

```


    
![png](v28_12_2_adjusted_files/v28_12_2_adjusted_17_0.png)
    



    
![png](v28_12_2_adjusted_files/v28_12_2_adjusted_17_1.png)
    



    
![png](v28_12_2_adjusted_files/v28_12_2_adjusted_17_2.png)
    



    
![png](v28_12_2_adjusted_files/v28_12_2_adjusted_17_3.png)
    



```python
# =============================================================================
# 6.5 Interpretability: coefficients as (approx.) log-odds contributions
# =============================================================================
coef = pd.Series(logit.coef_.ravel(), index=MODEL_FEATS).sort_values(ascending=False)

summary = pd.DataFrame({
    "feature": coef.index,
    "coef_log_odds": coef.values,
    "odds_ratio": np.exp(coef.values),  # change from 0->1 for binary, per 1SD for scaled continuous
}).sort_values("coef_log_odds", ascending=False)

print("Top positive (higher distress risk):")
display(summary.head(10))

print("Top negative (lower distress risk):")
display(summary.tail(10).iloc[::-1])

```

    Top positive (higher distress risk):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coef_log_odds</th>
      <th>odds_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>evt_lev_spike</td>
      <td>1.511997</td>
      <td>4.535778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>z_sp_debt_to_capital</td>
      <td>1.331778</td>
      <td>3.787770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>evt_liquidity_squeeze</td>
      <td>0.622571</td>
      <td>1.863714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>evt_cov_breach</td>
      <td>0.597312</td>
      <td>1.817227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>evt_cfo_negative</td>
      <td>0.412066</td>
      <td>1.509935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>evt_quick_squeeze</td>
      <td>0.208255</td>
      <td>1.231527</td>
    </tr>
    <tr>
      <th>6</th>
      <td>evt_cfo_collapse</td>
      <td>0.193399</td>
      <td>1.213367</td>
    </tr>
    <tr>
      <th>7</th>
      <td>evt_ebitda_shock</td>
      <td>0.160159</td>
      <td>1.173697</td>
    </tr>
    <tr>
      <th>8</th>
      <td>evt_div_initiate</td>
      <td>0.142845</td>
      <td>1.153551</td>
    </tr>
    <tr>
      <th>9</th>
      <td>z_sp_focf_to_debt</td>
      <td>0.039174</td>
      <td>1.039951</td>
    </tr>
  </tbody>
</table>
</div>


    Top negative (lower distress risk):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coef_log_odds</th>
      <th>odds_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>z_log_at</td>
      <td>-0.194051</td>
      <td>0.823616</td>
    </tr>
    <tr>
      <th>18</th>
      <td>evt_cov_collapse</td>
      <td>-0.141176</td>
      <td>0.868337</td>
    </tr>
    <tr>
      <th>17</th>
      <td>z_log_mkvalt</td>
      <td>-0.077602</td>
      <td>0.925332</td>
    </tr>
    <tr>
      <th>16</th>
      <td>z_sp_dcf_to_debt</td>
      <td>-0.058473</td>
      <td>0.943204</td>
    </tr>
    <tr>
      <th>15</th>
      <td>z_sp_ffo_to_debt</td>
      <td>-0.058362</td>
      <td>0.943308</td>
    </tr>
    <tr>
      <th>14</th>
      <td>z_sp_interest_coverage</td>
      <td>-0.056232</td>
      <td>0.945320</td>
    </tr>
    <tr>
      <th>13</th>
      <td>z_sp_debt_to_ebitda</td>
      <td>-0.034568</td>
      <td>0.966022</td>
    </tr>
    <tr>
      <th>12</th>
      <td>z_sp_cfo_to_debt</td>
      <td>-0.018550</td>
      <td>0.981621</td>
    </tr>
    <tr>
      <th>11</th>
      <td>evt_div_cut</td>
      <td>-0.010087</td>
      <td>0.989964</td>
    </tr>
    <tr>
      <th>10</th>
      <td>evt_div_suspend</td>
      <td>0.018399</td>
      <td>1.018569</td>
    </tr>
  </tbody>
</table>
</div>


### 6.6 Optional benchmark: Tree-based model (non-linear)

If you want a second point of reference without heavy tuning, a small gradient-boosted tree model often performs well on tabular data. This is optional for the seminar paper; keep Logistic Regression as the interpretability baseline.



```python
### 6.6 Optional benchmark: Cost-sensitive boosted trees (non-linear) + calibrated PDs

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor, export_text


# =============================================================================
# 0) HELPER UTILITIES
# =============================================================================
def _safe_feature_names(n_features: int):
    cand = globals().get("FEATURE_COLS", globals().get("MODEL_FEATS", None))
    if isinstance(cand, (list, tuple)) and len(cand) == n_features:
        return list(cand)
    return [f"f{i}" for i in range(n_features)]


def gmean_tpr_tnr(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    gmean = float(np.sqrt(tpr * tnr))
    return gmean, tpr, tnr, np.array([[tn, fp], [fn, tp]])


def expected_cost(y_true, proba, thr, cost_fn: float, cost_fp: float):
    y_hat = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp), np.array([[tn, fp], [fn, tp]])


def pd_decile_table(y_true, proba, n_bins=10):
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(proba).astype(float)})
    # qcut can fail with many ties; use rank to stabilize
    r = df["p"].rank(method="first")
    df["bin"] = pd.qcut(r, q=n_bins, labels=False) + 1  # 1..n_bins (1=lowest risk)
    out = (
        df.groupby("bin", as_index=False)
        .agg(n=("y", "size"), realized_rate=("y", "mean"), avg_pd=("p", "mean"),
             min_pd=("p", "min"), max_pd=("p", "max"))
        .sort_values("bin")
    )
    out["lift_vs_base"] = out["realized_rate"] / max(df["y"].mean(), 1e-12)
    return out


# =============================================================================
# 1) SPLIT VALIDATION INTO:
#    - early-stopping set (VAL_ES)
#    - calibration + threshold selection set (VAL_CAL)
#    This prevents using the same data for early stopping AND calibration/thresholding.
# =============================================================================
CAL_SIZE = 0.50  # half of VAL held for calibration + threshold selection

try:
    X_val_es, X_val_cal, y_val_es, y_val_cal = train_test_split(
        X_val, y_val,
        test_size=CAL_SIZE,
        random_state=42,
        stratify=y_val
    )
except Exception:
    # fallback if stratify fails (e.g., too few positives)
    X_val_es, X_val_cal, y_val_es, y_val_cal = train_test_split(
        X_val, y_val,
        test_size=CAL_SIZE,
        random_state=42
    )

# =============================================================================
# 2) COST-SENSITIVE WEIGHTS (weighted cross-entropy spirit: α_FN > α_FP)
#    We implement via sample_weight to avoid double-counting with scale_pos_weight.
# =============================================================================
pos = int(np.sum(y_train))
neg = int(len(y_train) - pos)
imbalance_ratio = neg / max(pos, 1)

# Choose explicit costs (finance logic: FN typically more costly than FP).
# Default: match imbalance ratio -> roughly "balanced" effective loss.
COST_FP = 1.0
COST_FN = float(imbalance_ratio)

w_train = np.where(np.asarray(y_train).astype(int) == 1, COST_FN, COST_FP)
w_val_es = np.where(np.asarray(y_val_es).astype(int) == 1, COST_FN, COST_FP)

# =============================================================================
# 3) XGBOOST SPECIFICATION (strong tabular baseline; modest regularization)
#    Key change: early stopping explicitly on PR-AUC (aucpr), not logloss.
# =============================================================================
base_params = dict(
    objective="binary:logistic",
    booster="gbtree",
    tree_method="hist",
    n_estimators=5000,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    gamma=0.0,
    max_delta_step=1,
    random_state=42,
    n_jobs=-1,
    # IMPORTANT: do NOT set scale_pos_weight if using sample_weight for costs
    scale_pos_weight=1.0,
)

# Early stop on PR-AUC explicitly (your learning curve peaked in aucpr before logloss)
try:
    es = xgb.callback.EarlyStopping(
        rounds=200,
        metric_name="aucpr",
        data_name="validation_0",
        maximize=True,
        save_best=True
    )
    xgb_clf = XGBClassifier(**base_params, eval_metric=["aucpr", "auc", "logloss"], callbacks=[es])
    xgb_clf.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val_es, y_val_es)],
        sample_weight_eval_set=[w_val_es],
        verbose=200
    )
except TypeError:
    # fallback for older API variants
    xgb_clf = XGBClassifier(**base_params, eval_metric=["aucpr", "auc", "logloss"], early_stopping_rounds=200)
    xgb_clf.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val_es, y_val_es)],
        sample_weight_eval_set=[w_val_es],
        verbose=200
    )

# =============================================================================
# 4) CALIBRATION (manual isotonic; avoids sklearn cv='prefit' deprecation path)
#    Calibrate ONLY on VAL_CAL (not used for early stopping).
# =============================================================================
raw_cal = xgb_clf.predict_proba(X_val_cal)[:, 1]
raw_test = xgb_clf.predict_proba(X_test)[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, np.asarray(y_val_cal).astype(int))

val_cal_proba = iso.predict(raw_cal)
test_proba = iso.predict(raw_test)

print("XGBoost (cost-sensitive + early stop on VAL_ES aucpr + isotonic calib on VAL_CAL):")
print(
    f"  VAL_CAL AUC={roc_auc_score(y_val_cal, val_cal_proba):.4f} | "
    f"PR-AUC={average_precision_score(y_val_cal, val_cal_proba):.4f} | "
    f"Brier={brier_score_loss(y_val_cal, val_cal_proba):.4f}"
)
print(
    f"  TEST    AUC={roc_auc_score(y_test, test_proba):.4f} | "
    f"PR-AUC={average_precision_score(y_test, test_proba):.4f} | "
    f"Brier={brier_score_loss(y_test, test_proba):.4f}"
)

# =============================================================================
# 5) THRESHOLDS: (i) cost-based, (ii) G-mean, (iii) capacity (top X%)
#    Thresholds chosen on VAL_CAL; then reported on TEST.
# =============================================================================
grid = np.linspace(0.01, 0.99, 99)

# (i) Cost-based threshold: minimize expected misclassification cost
best_cost = {"thr": 0.5, "cost": np.inf, "cm": None}
for t in grid:
    c, cm = expected_cost(y_val_cal, val_cal_proba, t, cost_fn=COST_FN, cost_fp=COST_FP)
    if c < best_cost["cost"]:
        best_cost.update({"thr": float(t), "cost": float(c), "cm": cm})

# (ii) G-mean threshold (imbalance-robust diagnostic)
best_g = {"thr": 0.5, "gmean": -1, "tpr": None, "tnr": None, "cm": None}
for t in grid:
    g, tpr, tnr, cm = gmean_tpr_tnr(y_val_cal, val_cal_proba, t)
    if g > best_g["gmean"]:
        best_g.update({"thr": float(t), "gmean": g, "tpr": tpr, "tnr": tnr, "cm": cm})

# (iii) Capacity rule: flag top X% by PD (common in finance screening / surveillance)
TOP_PCT = 0.10
thr_top = float(np.quantile(val_cal_proba, 1 - TOP_PCT))


def report_threshold(name, thr):
    g_val, tpr_val, tnr_val, cm_val = gmean_tpr_tnr(y_val_cal, val_cal_proba, thr)
    g_t, tpr_t, tnr_t, cm_t = gmean_tpr_tnr(y_test, test_proba, thr)

    # precision / flag rate
    tn, fp, fn, tp = cm_t.ravel()
    prec = tp / max(tp + fp, 1)
    flag = (tp + fp) / max((tp + fp + tn + fn), 1)

    print(f"\n{name}: threshold t={thr:.3f}")
    print(f"  VAL_CAL  G-mean={g_val:.3f} | TPR={tpr_val:.3f} | TNR={tnr_val:.3f} | CM:\n{cm_val}")
    print(
        f"  TEST     G-mean={g_t:.3f} | TPR={tpr_t:.3f} | TNR={tnr_t:.3f} | Precision={prec:.3f} | FlagRate={flag:.3f} | CM:\n{cm_t}")


report_threshold("Cost-based (min expected cost; COST_FN vs COST_FP)", best_cost["thr"])
report_threshold("G-mean (diagnostic)", best_g["thr"])
report_threshold(f"Top-{int(TOP_PCT * 100)}% capacity rule", thr_top)

# =============================================================================
# 6) FINANCE-STYLE PD SORTS (deciles): monotonicity check + lift
# =============================================================================
print("\nPD Deciles on VAL_CAL (1=lowest risk, 10=highest risk):")
print(pd_decile_table(y_val_cal, val_cal_proba, n_bins=10).to_string(index=False))

print("\nPD Deciles on TEST (1=lowest risk, 10=highest risk):")
print(pd_decile_table(y_test, test_proba, n_bins=10).to_string(index=False))

# =============================================================================
# 7) INTERPRETABILITY: “single-tree approximation” of the calibrated PD surface
#    (practical analogue to merging rules into an interpretable tree)
# =============================================================================
feature_names = _safe_feature_names(X_train.shape[1])

sur = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=max(50, int(0.01 * len(y_train))),
    random_state=42
)

# Fit surrogate on TRAIN predictions (do not use TEST); calibrated PD mapping included
raw_train = xgb_clf.predict_proba(X_train)[:, 1]
train_pd = iso.predict(raw_train)
sur.fit(X_train, train_pd)

print("\nSurrogate tree (depth<=4) approximating calibrated PDs:")
print(export_text(sur, feature_names=feature_names))

# =============================================================================
# 8) FEATURE IMPORTANCE: prefer SHAP; fallback to gain if SHAP unavailable
# =============================================================================
try:
    import shap  # optional

    explainer = shap.TreeExplainer(xgb_clf)
    idx = np.random.RandomState(42).choice(len(X_train), size=min(2000, len(X_train)), replace=False)
    shap_values = explainer.shap_values(X_train[idx])

    imp = np.mean(np.abs(shap_values), axis=0)
    top = np.argsort(-imp)[:20]
    print("\nTop-20 features by mean(|SHAP|):")
    for j in top:
        print(f"{feature_names[j]}: {imp[j]:.6f}")
except Exception:
    booster = xgb_clf.get_booster()
    score = booster.get_score(importance_type="gain")  # {"f0": gain, ...}
    imp = np.zeros(X_train.shape[1], dtype=float)
    for k, v in score.items():
        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
            j = int(k[1:])
            if 0 <= j < imp.size:
                imp[j] = float(v)

    top = np.argsort(-imp)[:20]
    print("\nTop-20 features by XGBoost gain (fallback; SHAP unavailable):")
    for j in top:
        print(f"{feature_names[j]}: {imp[j]:.6f}")

```

    [0]	validation_0-aucpr:0.71990	validation_0-auc:0.77188	validation_0-logloss:0.68889
    [200]	validation_0-aucpr:0.87663	validation_0-auc:0.86972	validation_0-logloss:0.44660
    [400]	validation_0-aucpr:0.87963	validation_0-auc:0.87111	validation_0-logloss:0.44157
    [600]	validation_0-aucpr:0.88052	validation_0-auc:0.87175	validation_0-logloss:0.44014
    [800]	validation_0-aucpr:0.88098	validation_0-auc:0.87206	validation_0-logloss:0.43991
    [1000]	validation_0-aucpr:0.88175	validation_0-auc:0.87269	validation_0-logloss:0.43933
    [1200]	validation_0-aucpr:0.88193	validation_0-auc:0.87265	validation_0-logloss:0.43959
    [1400]	validation_0-aucpr:0.88215	validation_0-auc:0.87263	validation_0-logloss:0.43991
    [1600]	validation_0-aucpr:0.88258	validation_0-auc:0.87292	validation_0-logloss:0.44000
    [1800]	validation_0-aucpr:0.88254	validation_0-auc:0.87297	validation_0-logloss:0.44037
    [1877]	validation_0-aucpr:0.88240	validation_0-auc:0.87277	validation_0-logloss:0.44070
    XGBoost (cost-sensitive + early stop on VAL_ES aucpr + isotonic calib on VAL_CAL):
      VAL_CAL AUC=0.8787 | PR-AUC=0.7292 | Brier=0.1074
      TEST    AUC=0.8809 | PR-AUC=0.7390 | Brier=0.1098
    
    Cost-based (min expected cost; COST_FN vs COST_FP): threshold t=0.210
      VAL_CAL  G-mean=0.804 | TPR=0.758 | TNR=0.853 | CM:
    [[2062  354]
     [ 192  600]]
      TEST     G-mean=0.810 | TPR=0.800 | TNR=0.821 | Precision=0.614 | FlagRate=0.342 | CM:
    [[7510 1636]
     [ 653 2605]]
    
    G-mean (diagnostic): threshold t=0.210
      VAL_CAL  G-mean=0.804 | TPR=0.758 | TNR=0.853 | CM:
    [[2062  354]
     [ 192  600]]
      TEST     G-mean=0.810 | TPR=0.800 | TNR=0.821 | Precision=0.614 | FlagRate=0.342 | CM:
    [[7510 1636]
     [ 653 2605]]
    
    Top-10% capacity rule: threshold t=0.759
      VAL_CAL  G-mean=0.606 | TPR=0.375 | TNR=0.979 | CM:
    [[2365   51]
     [ 495  297]]
      TEST     G-mean=0.619 | TPR=0.393 | TNR=0.976 | Precision=0.856 | FlagRate=0.121 | CM:
    [[8930  216]
     [1979 1279]]
    
    PD Deciles on VAL_CAL (1=lowest risk, 10=highest risk):
     bin   n  realized_rate   avg_pd   min_pd   max_pd  lift_vs_base
       1 321       0.015576 0.014982 0.000000 0.026012      0.063092
       2 321       0.024922 0.026765 0.026012 0.028926      0.100947
       3 321       0.046729 0.047120 0.028926 0.078365      0.189276
       4 320       0.075000 0.078365 0.078365 0.078365      0.303788
       5 321       0.093458 0.091740 0.078365 0.155172      0.378552
       6 321       0.168224 0.162100 0.155172 0.164286      0.681393
       7 320       0.168750 0.172232 0.164286 0.200000      0.683523
       8 321       0.352025 0.351936 0.200000 0.457143      1.425879
       9 321       0.663551 0.661406 0.457143 0.758621      2.687718
      10 321       0.859813 0.861424 0.758621 1.000000      3.482677
    
    PD Deciles on TEST (1=lowest risk, 10=highest risk):
     bin    n  realized_rate   avg_pd   min_pd   max_pd  lift_vs_base
       1 1241       0.019339 0.015660 0.000000 0.026012      0.073629
       2 1240       0.037097 0.027272 0.026012 0.028926      0.141236
       3 1240       0.048387 0.057197 0.028926 0.078365      0.184221
       4 1241       0.079774 0.078365 0.078365 0.078365      0.303720
       5 1240       0.097581 0.119318 0.078365 0.164286      0.371513
       6 1240       0.148387 0.164520 0.164286 0.171779      0.564946
       7 1241       0.212732 0.233233 0.171779 0.309278      0.809921
       8 1240       0.395968 0.433648 0.309278 0.600000      1.507546
       9 1240       0.721774 0.710090 0.600000 0.775000      2.747970
      10 1241       0.865431 0.866160 0.775000 1.000000      3.294907
    
    Surrogate tree (depth<=4) approximating calibrated PDs:
    |--- z_sp_debt_to_capital <= -0.21
    |   |--- z_sp_interest_coverage <= -0.12
    |   |   |--- z_sp_cfo_to_debt <= -0.03
    |   |   |   |--- evt_liquidity_squeeze <= 0.50
    |   |   |   |   |--- value: [0.17]
    |   |   |   |--- evt_liquidity_squeeze >  0.50
    |   |   |   |   |--- value: [0.26]
    |   |   |--- z_sp_cfo_to_debt >  -0.03
    |   |   |   |--- z_log_at <= -0.68
    |   |   |   |   |--- value: [0.15]
    |   |   |   |--- z_log_at >  -0.68
    |   |   |   |   |--- value: [0.09]
    |   |--- z_sp_interest_coverage >  -0.12
    |   |   |--- z_sp_interest_coverage <= 0.64
    |   |   |   |--- z_sp_dcf_to_debt <= 0.12
    |   |   |   |   |--- value: [0.15]
    |   |   |   |--- z_sp_dcf_to_debt >  0.12
    |   |   |   |   |--- value: [0.06]
    |   |   |--- z_sp_interest_coverage >  0.64
    |   |   |   |--- evt_liquidity_squeeze <= 0.50
    |   |   |   |   |--- value: [0.02]
    |   |   |   |--- evt_liquidity_squeeze >  0.50
    |   |   |   |   |--- value: [0.06]
    |--- z_sp_debt_to_capital >  -0.21
    |   |--- z_sp_ffo_to_debt <= 0.06
    |   |   |--- z_sp_focf_to_debt <= 0.08
    |   |   |   |--- z_sp_debt_to_capital <= -0.21
    |   |   |   |   |--- value: [0.83]
    |   |   |   |--- z_sp_debt_to_capital >  -0.21
    |   |   |   |   |--- value: [0.59]
    |   |   |--- z_sp_focf_to_debt >  0.08
    |   |   |   |--- z_sp_debt_to_capital <= 0.38
    |   |   |   |   |--- value: [0.17]
    |   |   |   |--- z_sp_debt_to_capital >  0.38
    |   |   |   |   |--- value: [0.58]
    |   |--- z_sp_ffo_to_debt >  0.06
    |   |   |--- z_sp_debt_to_capital <= 1.38
    |   |   |   |--- z_sp_debt_to_capital <= 0.36
    |   |   |   |   |--- value: [0.08]
    |   |   |   |--- z_sp_debt_to_capital >  0.36
    |   |   |   |   |--- value: [0.20]
    |   |   |--- z_sp_debt_to_capital >  1.38
    |   |   |   |--- value: [0.75]
    
    
    Top-20 features by XGBoost gain (fallback; SHAP unavailable):
    z_sp_debt_to_capital: 109.433311
    evt_liquidity_squeeze: 82.294708
    evt_cov_breach: 59.479637
    z_sp_ffo_to_debt: 52.799309
    z_sp_cfo_to_debt: 49.851017
    z_sp_debt_to_ebitda: 30.006271
    evt_lev_spike: 29.498652
    z_sp_interest_coverage: 21.789652
    evt_quick_squeeze: 21.209650
    evt_cfo_negative: 17.018761
    z_sp_focf_to_debt: 16.517914
    z_sp_dcf_to_debt: 13.130123
    z_log_at: 12.953803
    z_log_mkvalt: 9.860869
    evt_div_initiate: 8.223381
    evt_cov_collapse: 7.804729
    evt_cfo_collapse: 7.746669
    evt_div_cut: 6.921174
    evt_ebitda_shock: 6.825545
    evt_div_suspend: 5.840136



```python
# =============================================================================
# 6.7 Export: out-of-sample predictions (VAL and TEST) for reporting / appendix
# =============================================================================
ID_COLS = [c for c in ["gvkey", "fyear", "label_year"] if c in train.columns]

pred_val = val[ID_COLS + [TARGET_COL]].copy()
pred_val["p_distress_logit"] = val_proba

pred_test = test[ID_COLS + [TARGET_COL]].copy()
pred_test["p_distress_logit"] = test_proba

pred_oos = pd.concat(
    [pred_val.assign(split="VAL"), pred_test.assign(split="TEST")],
    ignore_index=True
)

out_csv = "oos_predictions_logit.csv"
pred_oos.to_csv(out_csv, index=False)
print(f"Saved: {out_csv} | rows={len(pred_oos):,} | cols={pred_oos.shape[1]}")

```

    Saved: oos_predictions_logit.csv | rows=18,819 | cols=6

