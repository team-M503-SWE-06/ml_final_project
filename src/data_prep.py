"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Keep ALL data wrangling here so the rest of the code just calls these
functions. Every cleaning decision is appended to CLEANING_LOG, which
run_all prints so the report can quote it verbatim.

Leakage policy
  * Target-derived columns (LEAKAGE_COLUMNS) are dropped in clean().
  * Anything *fitted* (category vocabularies, imputation medians,
    standardisation mean/std, the tier threshold) uses TRAIN rows only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"          # Kaggle: sehriyarmemmedli/binaaz-sale-project v1
SPLIT_PATH = "data/split_indices.npz"      # frozen split (row positions in the cleaned frame)

# Columns that LEAK the target (derived from price). Verified on the data:
#   unit_price  = price / area   (corr(unit_price * area, price) = 1.0000)
#   total_price = price          (identical in 99.99 % of rows)
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
]

# Identifiers, free text, and scrape/listing metadata. None of them is a
# property attribute known at pricing time, and the identifiers would let a
# model memorise individual listings or agents.
ID_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url", "estate_rel_url_x", "estate_rel_url_y", "img_url",
    "owner_name", "shop_name", "shop_title", "address",
]
TEXT_COLUMNS: list[str] = ["description", "attributes"]   # attributes = "3 otaqlı, 100 m²" (redundant)
META_COLUMNS: list[str] = [
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "day_y", "hour_x", "hour_y",
    "city_when", "updated", "views", "vip", "featured",
    "currency_x", "currency_y",     # consumed by the currency conversion
]
# English-named raw columns that duplicate the Azerbaijani detail columns
# (Təmirli <-> Təmir, 'Çıxarış var' <-> Çıxarış, 'İpoteka var' <-> İpoteka).
REDUNDANT_COLUMNS: list[str] = ["repair", "bill_of_sale", "mortgage"]

# Azerbaijani -> English names for the columns we keep.
COLUMN_MAP: dict[str, str] = {
    "Sahə": "area",
    "Otaq sayı": "rooms",
    "Mərtəbə": "floor_raw",
    "Torpaq sahəsi": "land_area",
    "Kateqoriya": "category",
    "Binanın növü": "building_type",
    "Təmir": "repair",
    "Çıxarış": "bill_of_sale",
    "İpoteka": "mortgage",
    "extra_info": "region",
    "products_label": "listing_label",
    "owner_title": "owner_type",
}
CATEGORY_MAP: dict[str, str] = {
    "Yeni tikili": "new_build",
    "Köhnə tikili": "old_build",
    "Həyət evi/Bağ evi": "house",
    "Torpaq": "land",
    "Obyekt": "commercial",
    "Ofis": "office",
    "Qaraj": "garage",
}

USD_TO_AZN = 1.70            # AZN is pegged at 1.70 per USD
SOT_TO_M2 = 100.0            # 1 sot = 100 m²
AZ_BBOX = (38.3, 41.95, 44.7, 50.7)   # lat_min, lat_max, lng_min, lng_max of Azerbaijan
MIN_PRICE_AZN = 1_000        # below this a sale price is not credible (min in data: 11 AZN)
MIN_BUILDING_AREA = 10.0     # m²; smaller "buildings" are data-entry errors
# Outlier limits are FIXED domain bounds, not data quantiles, so no statistic
# from val/test rows decides which listings exist.
PRICE_LIMITS_AZN = (10_000, 5_000_000)
AREA_LIMITS_M2: dict[str, tuple[float, float]] = {
    "new_build": (20, 500),
    "old_build": (15, 300),
    "house": (30, 1_500),
    "land": (100, 200_000),        # 1 sot .. 20 ha
    "commercial": (10, 10_000),
    "office": (10, 10_000),
    "garage": (10, 2_000),
}
MAX_MISSING_FRAC = 0.95      # drop columns that are almost entirely empty
TOP_K_LOCATIONS = 30         # most frequent locations (train) kept, rest -> "other"
MIN_LEVEL_COUNT = 30         # rarer category/district levels (train) -> "other"

NUMERIC_FEATURES: list[str] = [
    "area", "rooms", "floor", "total_floors", "land_area", "lat", "lng",
]
BINARY_FEATURES: list[str] = [
    "repair", "bill_of_sale", "mortgage", "is_agent", "in_complex", "is_baku",
]
CATEGORICAL_FEATURES: list[str] = ["category", "district", "location"]

CLEANING_LOG: list[str] = []


def _log(msg: str) -> None:
    CLEANING_LOG.append(msg)


# --- Loading ----------------------------------------------------------------
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


# --- Parsing helpers --------------------------------------------------------
def _to_number(s: pd.Series) -> pd.Series:
    """'2 500,5 AZN/m²' -> 2500.5 ; strips spaces, units and currency text."""
    s = s.astype("string").str.replace(r"\s", "", regex=True).str.replace(",", ".", regex=False)
    num = pd.to_numeric(s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")
    return num.astype("float64")


def _area_m2(s: pd.Series) -> pd.Series:
    """Area strings come as '85 m²' or '6 sot' (land plots). Return m²."""
    value = _to_number(s)
    is_sot = s.astype("string").str.contains("sot", na=False)
    return value.where(~is_sot, value * SOT_TO_M2)


def _yes_no(s: pd.Series) -> pd.Series:
    """'var' (there is) -> 1, 'yoxdur' (there is not) -> 0, else NaN."""
    s = s.astype("string").str.strip().str.lower()
    return s.map({"var": 1.0, "yoxdur": 0.0}).astype(float)


def _price_azn(df: pd.DataFrame) -> pd.Series:
    """Numeric price in AZN; rows quoted in USD are converted at USD_TO_AZN."""
    raw = df["price"]
    price = raw.astype(float) if pd.api.types.is_numeric_dtype(raw) else _to_number(raw)
    if "currency_x" in df:
        cur = df["currency_x"].astype("string").str.upper().str.strip()
        is_usd = cur.isin(["USD", "$"])
        price = price.where(~is_usd, price * USD_TO_AZN)
        counts = {k: int(v) for k, v in cur.value_counts().items()}
        _log(f"currency: {counts}; {int(is_usd.sum())} USD rows converted at {USD_TO_AZN} AZN/USD")
    return price


# --- Cleaning ---------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame (every decision is recorded in CLEANING_LOG):
      - fix dtypes (prices/areas/floors to numeric; strip currency/units),
      - drop duplicates (exact rows, then re-scrapes of the same listing),
      - drop impossible values and price/area outliers,
      - drop LEAKAGE_COLUMNS, identifiers, free text and scrape metadata.
    Missing feature values stay NaN; they are imputed later from TRAIN only.
    """
    CLEANING_LOG.clear()
    df = df.copy()
    _log(f"raw: {len(df):,} rows x {df.shape[1]} columns")

    # 1. duplicates ----------------------------------------------------------
    n = len(df)
    df = df.drop_duplicates()
    _log(f"exact duplicate rows dropped: {n - len(df):,}")

    # the same listing (/items/<id>) is scraped on several days; keep the latest
    n = len(df)
    df = (df.sort_values("datetime_scrape_x", kind="mergesort")
            .drop_duplicates("estate_rel_url_x", keep="last")
            .sort_index())
    _log(f"repeat scrapes of the same listing (estate_rel_url) dropped, latest kept: {n - len(df):,}")

    # 2. types ----------------------------------------------------------------
    df["price"] = _price_azn(df)
    df = df.drop(columns=[c for c in REDUNDANT_COLUMNS if c in df])
    _log(f"redundant raw columns dropped (same info as the Azerbaijani detail columns): "
         f"{REDUNDANT_COLUMNS}")
    df = df.rename(columns=COLUMN_MAP)
    df["category"] = df["category"].map(CATEGORY_MAP).fillna(df["category"])
    df["area"] = _area_m2(df["area"])
    df["land_area"] = _area_m2(df["land_area"])
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    floors = df["floor_raw"].astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    df["floor"] = pd.to_numeric(floors[0], errors="coerce").astype("float64")
    df["total_floors"] = pd.to_numeric(floors[1], errors="coerce").astype("float64")
    _log("parsed: area '85 m²' / '6 sot' -> m² (1 sot = 100 m²); land_area 'sot' -> m²; "
         "floor '3 / 9' -> floor=3, total_floors=9")

    for col in ("repair", "bill_of_sale"):
        df[col] = _yes_no(df[col])
    # İpoteka only ever says 'var'; an empty cell means the option is not offered
    df["mortgage"] = _yes_no(df["mortgage"]).fillna(0.0)
    df["is_agent"] = (df["owner_type"].astype("string").str.contains("agent", na=False)).astype(float)
    df["in_complex"] = (df["listing_label"] == "Kompleks").astype(float)
    df["is_baku"] = (df["city"].astype("string").str.strip().str.lower() == "bakı").astype(float)
    _log("binary: repair / bill_of_sale ('var'=1, 'yoxdur'=0, else NaN); mortgage (empty -> 0); "
         "is_agent (owner_title), in_complex (products_label='Kompleks'), is_baku (city)")

    # district = the 'r.' (rayon) token of extra_info, e.g. 'Xəzər r.* Mərdəkan q.' -> 'Xəzər r.'
    df["district"] = (df["region"].astype("string")
                      .str.extract(r"([^*]+? r\.)", expand=False).str.strip().astype(object))
    _log(f"district extracted from extra_info: {df['district'].nunique()} levels, "
         f"{df['district'].isna().mean():.1%} missing")

    # 3. impossible values -----------------------------------------------------
    n = len(df)
    df = df[df["price"].notna() & (df["price"] >= MIN_PRICE_AZN)]
    _log(f"price missing or < {MIN_PRICE_AZN:,} AZN dropped: {n - len(df):,}")

    n = len(df)
    is_land = df["category"] == "land"
    bad_area = df["area"].isna() | (df["area"] <= 0) | (~is_land & (df["area"] < MIN_BUILDING_AREA))
    df = df[~bad_area]
    _log(f"area missing/<=0, or building < {MIN_BUILDING_AREA:.0f} m² dropped: {n - len(df):,}")

    n = len(df)
    bad = (df["rooms"] <= 0) | (df["total_floors"] < 1) | (df["floor"] > df["total_floors"])
    df = df[~bad.fillna(False)]
    _log(f"rooms <= 0, total_floors < 1, or floor > total_floors dropped: {n - len(df):,}")

    lat0, lat1, lng0, lng1 = AZ_BBOX
    off_map = ~(df["lat"].between(lat0, lat1) & df["lng"].between(lng0, lng1))
    df.loc[off_map, ["lat", "lng"]] = np.nan
    _log(f"lat/lng outside Azerbaijan bbox {AZ_BBOX} set to NaN: {int(off_map.sum()):,}")

    # 4. outliers --------------------------------------------------------------
    # Prices are heavy-tailed (max 600M AZN); the extreme tails are mostly
    # mis-typed prices. Drop rows outside fixed, documented bounds; area bounds
    # are per category because land plots and flats live on different scales.
    p_lo, p_hi = PRICE_LIMITS_AZN
    n = len(df)
    df = df[df["price"].between(p_lo, p_hi)]
    _log(f"price outside fixed limits [{p_lo:,}, {p_hi:,}] AZN dropped: {n - len(df):,}")

    n = len(df)
    a_lo = df["category"].map({k: v[0] for k, v in AREA_LIMITS_M2.items()}).fillna(0.0)
    a_hi = df["category"].map({k: v[1] for k, v in AREA_LIMITS_M2.items()}).fillna(np.inf)
    df = df[(df["area"] >= a_lo) & (df["area"] <= a_hi)]
    _log(f"area outside fixed per-category limits (m²) {AREA_LIMITS_M2} dropped: {n - len(df):,}")

    # 5. columns ---------------------------------------------------------------
    drop = [c for c in LEAKAGE_COLUMNS if c in df]
    _log(f"LEAKAGE columns dropped (target-derived): {drop}")
    ids = [c for c in ID_COLUMNS if c in df]
    _log(f"identifier columns dropped: {ids}")
    other = [c for c in TEXT_COLUMNS + META_COLUMNS if c in df]
    _log(f"free-text / scrape-metadata / redundant columns dropped: {other}")
    helpers = ["floor_raw", "region", "listing_label", "owner_type", "city"]
    sparse = [c for c in df.columns
              if c not in drop + ids + other + helpers and df[c].isna().mean() > MAX_MISSING_FRAC]
    _log(f"columns > {MAX_MISSING_FRAC:.0%} missing dropped: "
         f"{[(c, round(float(df[c].isna().mean()), 3)) for c in sparse]}")
    df = df.drop(columns=drop + ids + other + helpers + sparse)

    keep = ["price"] + NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    df = df[keep].reset_index(drop=True)

    miss = {c: f"{df[c].isna().mean():.1%}" for c in keep if df[c].isna().any()}
    _log(f"remaining missing values (imputed later with TRAIN medians + *_missing flags): {miss}")
    _log(f"clean: {len(df):,} rows x {df.shape[1]} columns")
    return df


# --- Features ---------------------------------------------------------------
def _vocabulary(s: pd.Series, top_k: int | None, min_count: int) -> list[str]:
    counts = s.dropna().value_counts()
    counts = counts[counts >= min_count]
    if top_k is not None:
        counts = counts.head(top_k)
    return sorted(counts.index)


def make_features(df: pd.DataFrame, fit_mask=None):
    """
    Turn the cleaned frame into a numeric design matrix.
      - numeric columns kept as-is (NaN preserved) + a 0/1 `<col>_missing`
        indicator for every numeric column that has missing values,
      - binary flags (NaN preserved -> imputed later),
      - one-hot encoding for category, district and location (nominal, so
        no artificial order). Levels are learned on `fit_mask` rows (train)
        only; rare / unseen levels go to "<col>=other",
      - y = log(price)  (prices are heavy-tailed).
    Return (X, y, feature_names).
    """
    fit = np.ones(len(df), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    fit_df = df[fit]

    cols, names = [], []
    for c in NUMERIC_FEATURES:
        cols.append(df[c].to_numpy(dtype=float))
        names.append(c)
    for c in NUMERIC_FEATURES:
        if fit_df[c].isna().any():
            cols.append(df[c].isna().to_numpy(dtype=float))
            names.append(f"{c}_missing")
    for c in BINARY_FEATURES:
        cols.append(df[c].to_numpy(dtype=float))
        names.append(c)
    for c in CATEGORICAL_FEATURES:
        top_k = TOP_K_LOCATIONS if c == "location" else None
        vocab = _vocabulary(fit_df[c], top_k, MIN_LEVEL_COUNT)
        values = df[c].astype("string").fillna("missing")
        for level in vocab:
            cols.append((values == level).to_numpy(dtype=float))
            names.append(f"{c}={level}")
        cols.append((~values.isin(vocab)).to_numpy(dtype=float))
        names.append(f"{c}=other")

    X = np.column_stack(cols)
    y = np.log(df["price"].to_numpy(dtype=float))
    return X, y, names


# --- Split ------------------------------------------------------------------
def split_indices(y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Stratified train/val/test row indices, NumPy only.

    The final tier threshold must come from train, which does not exist yet,
    so we stratify on a *provisional* median bin of y (log-price is monotone
    in price, so this is the same bin as on price). The real threshold is
    re-derived from the training rows in make_tier_label.
    """
    y = np.asarray(y, dtype=float)
    strata = (y > np.median(y)).astype(int)
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def freeze_split(idx_tr, idx_val, idx_te, path: str = SPLIT_PATH) -> None:
    """Save the split; if one is already saved, assert it is reproduced exactly."""
    if os.path.exists(path):
        saved = np.load(path)
        same = all(np.array_equal(saved[k], v)
                   for k, v in (("train", idx_tr), ("val", idx_val), ("test", idx_te)))
        if not same:
            raise RuntimeError(f"split differs from the frozen split in {path}; "
                               "cleaning or SEED changed — delete the file to re-freeze")
        return
    np.savez(path, train=idx_tr, val=idx_val, test=idx_te)


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic stratified split into train / validation / test (70/15/15).
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    idx_tr, idx_val, idx_te = split_indices(y, val_size, test_size, seed)
    return X[idx_tr], y[idx_tr], X[idx_val], y[idx_val], X[idx_te], y[idx_te]


# --- Fitted transforms (TRAIN statistics only) ------------------------------
def impute_median(X_tr, *others):
    """Fill NaNs with TRAIN column medians. Return arrays in the given order."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for X in (X_tr, *others):
        X = X.copy()
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = med[cols]
        out.append(X)
    return tuple(out)


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  Call it on train first, then pass that
    threshold for val/test (no peeking at val/test).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return tuple((X - mean) / std for X in (X_tr, *others))
