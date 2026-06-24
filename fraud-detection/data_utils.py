"""
data_utils.py
Loading, merging tabel transaction + identity, dan reduksi memori.
"""
import numpy as np
import pandas as pd

import config


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast tipe numerik untuk menghemat RAM (dataset ini ~1.5 GB di memori)."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or str(col_type).startswith("category"):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            # float: pakai float32 (cukup presisi, hemat 50%)
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"[mem] {start:7.1f} MB -> {end:7.1f} MB "
              f"({100 * (start - end) / start:4.1f}% lebih kecil)")
    return df


def _normalize_identity_columns(df_id: pd.DataFrame) -> pd.DataFrame:
    """Beberapa rilis dataset memakai 'id-01', versi lain 'id_01'. Samakan ke underscore."""
    df_id = df_id.rename(columns=lambda c: c.replace("id-", "id_"))
    return df_id


def load_data(transaction_path=None, identity_path=None,
              nrows=None, reduce_mem=True) -> pd.DataFrame:
    """
    Load train_transaction.csv dan (kalau ada) train_identity.csv,
    lalu LEFT JOIN di TransactionID.

    nrows: batasi jumlah baris (berguna untuk eksperimen cepat).
    """
    transaction_path = transaction_path or config.TRANSACTION_CSV
    identity_path = identity_path or config.IDENTITY_CSV

    print(f"[load] membaca {transaction_path} ...")
    df = pd.read_csv(transaction_path, nrows=nrows)
    print(f"[load] transaction: {df.shape}")

    try:
        df_id = pd.read_csv(identity_path)
        df_id = _normalize_identity_columns(df_id)
        print(f"[load] identity   : {df_id.shape}")
        df = df.merge(df_id, on=config.ID_COL, how="left")
        print(f"[load] merged     : {df.shape}")
    except FileNotFoundError:
        print(f"[load] identity tidak ditemukan ({identity_path}) -> lanjut tanpa identity.")

    if reduce_mem:
        df = reduce_mem_usage(df)
    return df
