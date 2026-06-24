"""
generate_synthetic.py
Buat data SINTETIK kecil yang meniru skema IEEE-CIS (train_transaction.csv +
train_identity.csv) supaya kamu bisa MENGUJI pipeline tanpa download dataset 600 MB.

Jalankan:  python generate_synthetic.py --rows 20000
Lalu:      python train_lgbm.py --nrows 20000 --trials 5

CATATAN: ini hanya untuk uji jalannya kode. Hasil metrik TIDAK bermakna —
untuk hasil nyata pakai dataset asli dari Kaggle.
"""
import argparse
import numpy as np
import pandas as pd

import config


def generate(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.035            # ~3.5% fraud, mirip dataset asli

    df = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1) + 2987000,
        "isFraud": fraud.astype(int),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 180, n)),
        "TransactionAmt": np.round(rng.lognormal(4, 1, n), 2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
        "card1": rng.integers(1000, 18000, n),
        "card2": rng.integers(100, 600, n).astype(float),
        "card3": rng.choice([150.0, 185.0], n),
        "card4": rng.choice(["visa", "mastercard", "amex", "discover", None], n),
        "card5": rng.integers(100, 240, n).astype(float),
        "card6": rng.choice(["debit", "credit", None], n),
        "addr1": rng.choice(list(range(100, 540)) + [np.nan], n),
        "addr2": rng.choice([87.0, 60.0, np.nan], n),
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
    })

    # C1-C14, D1-D15, M1-M9, beberapa V (anonim)
    for i in range(1, 15):
        df[f"C{i}"] = rng.integers(0, 50, n).astype(float)
    for i in range(1, 16):
        col = rng.integers(0, 600, n).astype(float)
        col[rng.random(n) < 0.4] = np.nan          # banyak missing seperti aslinya
        df[f"D{i}"] = col
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice(["T", "F", None], n)
    for i in range(1, 50):
        col = rng.normal(0, 1, n)
        col[rng.random(n) < 0.5] = np.nan
        df[f"V{i}"] = col

    # buat fitur sedikit informatif supaya AUC > 0.5 (uji sanity)
    df.loc[df["isFraud"] == 1, "TransactionAmt"] *= 1.4
    df.loc[df["isFraud"] == 1, "C1"] += 8

    # identity (hanya ~25% transaksi punya)
    has_id = rng.random(n) < 0.25
    id_ids = df.loc[has_id, "TransactionID"].values
    df_id = pd.DataFrame({"TransactionID": id_ids})
    for i in range(1, 12):
        df_id[f"id_{i:02d}"] = rng.normal(0, 1, len(id_ids))
    for i in range(12, 39):
        df_id[f"id_{i:02d}"] = rng.choice(["A", "B", "C", None], len(id_ids))
    df_id["DeviceType"] = rng.choice(["mobile", "desktop", None], len(id_ids))
    df_id["DeviceInfo"] = rng.choice(
        ["Windows", "iOS", "Android", "MacOS", None], len(id_ids))

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.TRANSACTION_CSV, index=False)
    df_id.to_csv(config.IDENTITY_CSV, index=False)
    print(f"[ok] {config.TRANSACTION_CSV}  ({df.shape})")
    print(f"[ok] {config.IDENTITY_CSV}  ({df_id.shape})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()
    generate(args.rows)
