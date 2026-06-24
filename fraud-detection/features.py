"""
features.py
Feature engineering + handling missing values + encoding.

Dua jalur:
- prepare_for_tree(): biarkan NaN apa adanya, kategori -> dtype 'category'
  (LightGBM menangani NaN & kategori secara native).
- prepare_for_dl():  impute + standardize numerik, kategori -> index integer
  untuk embedding layer.
"""
import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------
# 1. Feature engineering row-wise (aman, tidak menyebabkan leakage)
# ----------------------------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- TransactionAmt ---
    if "TransactionAmt" in df:
        amt = df["TransactionAmt"].astype("float32")
        df["TransactionAmt_log"] = np.log1p(amt)
        # bagian desimal (sen) sering jadi sinyal kuat untuk fraud
        df["TransactionAmt_decimal"] = ((amt - np.floor(amt)) * 1000).astype("float32")

    # --- Fitur waktu dari TransactionDT (detik sejak titik referensi) ---
    if config.TIME_COL in df:
        dt = df[config.TIME_COL].astype("float64")
        df["hour"] = np.floor(dt / 3600) % 24
        df["dayofweek"] = np.floor(dt / (3600 * 24)) % 7
        df["hour"] = df["hour"].astype("float32")
        df["dayofweek"] = df["dayofweek"].astype("float32")

    # --- Jumlah missing per baris (proxy kelengkapan data) ---
    df["n_missing"] = df.isna().sum(axis=1).astype("float32")

    # --- Email domain disederhanakan (ambil provider utama) ---
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df:
            df[col + "_bin"] = (
                df[col].astype("object").str.split(".").str[0].fillna("missing")
            )

    return df


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Kolom kategori = semua kolom bertipe object (string)."""
    drop = {config.ID_COL, config.TARGET}
    return [c for c in df.columns
            if (df[c].dtype == object) and c not in drop]


# ----------------------------------------------------------------------
# 2. Frequency encoding (fit di train, apply ke valid/test -> tanpa leakage)
# ----------------------------------------------------------------------
class FrequencyEncoder:
    """Ganti nilai kategori dengan frekuensi kemunculannya di train set."""
    def __init__(self, columns):
        self.columns = columns
        self.maps_ = {}

    def fit(self, df):
        for col in self.columns:
            self.maps_[col] = df[col].astype("object").value_counts(normalize=True)
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.columns:
            df[col + "_freq"] = (
                df[col].astype("object").map(self.maps_[col]).fillna(0).astype("float32")
            )
        return df


# ----------------------------------------------------------------------
# 3. Jalur TREE (LightGBM / XGBoost)
# ----------------------------------------------------------------------
def prepare_for_tree(df: pd.DataFrame):
    """
    Return: X (DataFrame), y (Series), cat_cols (list)
    Kategori dikonversi ke dtype 'category' supaya LightGBM menangani native.
    NaN dibiarkan (LightGBM punya default direction untuk missing).
    """
    df = add_basic_features(df)
    cat_cols = get_categorical_columns(df)

    for c in cat_cols:
        df[c] = df[c].astype("category")

    y = df[config.TARGET].astype("int8") if config.TARGET in df else None
    drop = [c for c in [config.ID_COL, config.TARGET] if c in df]
    X = df.drop(columns=drop)
    cat_cols = [c for c in cat_cols if c in X.columns]
    return X, y, cat_cols


# ----------------------------------------------------------------------
# 4. Jalur DEEP LEARNING (MLP + entity embeddings)
# ----------------------------------------------------------------------
class DLPreprocessor:
    """
    Fit di train: simpan median numerik, mean/std, dan vocab kategori.
    Transform: numerik -> impute(median)+standardize; kategori -> index int.
    Index 0 dicadangkan untuk 'unknown/rare/missing'.
    """
    def __init__(self, max_vocab=config.DL_MAX_VOCAB):
        self.max_vocab = max_vocab
        self.num_cols = None
        self.cat_cols = None
        self.medians_ = {}
        self.means_ = {}
        self.stds_ = {}
        self.vocab_ = {}          # col -> {kategori: index}
        self.cat_dims_ = {}       # col -> jumlah kategori (termasuk slot unknown)

    def fit(self, df):
        df = add_basic_features(df)
        self.cat_cols = get_categorical_columns(df)
        drop = {config.ID_COL, config.TARGET, *self.cat_cols}
        self.num_cols = [c for c in df.columns
                         if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

        # numerik
        for c in self.num_cols:
            med = df[c].median()
            self.medians_[c] = 0.0 if pd.isna(med) else float(med)
            filled = df[c].fillna(self.medians_[c]).astype("float64")
            self.means_[c] = float(filled.mean())
            std = float(filled.std())
            self.stds_[c] = std if std > 1e-6 else 1.0

        # kategori: ambil top-(max_vocab-1) kategori paling sering
        for c in self.cat_cols:
            top = (df[c].astype("object").fillna("missing")
                   .value_counts().head(self.max_vocab - 1).index.tolist())
            self.vocab_[c] = {cat: i + 1 for i, cat in enumerate(top)}  # 0 = unknown
            self.cat_dims_[c] = len(top) + 1
        return self

    def transform(self, df):
        df = add_basic_features(df)

        X_num = np.zeros((len(df), len(self.num_cols)), dtype=np.float32)
        for j, c in enumerate(self.num_cols):
            col = df[c].fillna(self.medians_[c]).astype("float64")
            X_num[:, j] = ((col - self.means_[c]) / self.stds_[c]).astype("float32")

        X_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            vals = df[c].astype("object").fillna("missing")
            X_cat[:, j] = vals.map(self.vocab_[c]).fillna(0).astype("int64").values

        y = df[config.TARGET].astype("float32").values if config.TARGET in df else None
        return X_num, X_cat, y

    @property
    def embedding_specs(self):
        """List (num_categories, embedding_dim) untuk tiap kolom kategori."""
        specs = []
        for c in self.cat_cols:
            n = self.cat_dims_[c]
            dim = int(min(50, round(1.6 * n ** 0.56)))  # rule-of-thumb fastai
            specs.append((n, max(dim, 1)))
        return specs
