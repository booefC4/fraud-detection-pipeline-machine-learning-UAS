# Fraud Detection — IEEE-CIS (End-to-End ML/DL Pipeline)

Pipeline lengkap untuk memprediksi probabilitas sebuah transaksi online itu **fraud**
(`isFraud`), memakai dataset IEEE-CIS Fraud Detection. Mencakup preprocessing,
handling missing values & class imbalance, feature engineering, dua jenis model
(LightGBM & Deep Learning), hyperparameter tuning dengan **Optuna**, evaluasi dengan
metrik yang sesuai untuk data imbalanced, dan tracking dengan **MLflow**.

---

## 1. Struktur Proyek

```
fraud-detection/
├── config.py             # pusat konfigurasi (path, seed, hyperparam, MLflow)
├── data_utils.py         # load + merge transaction&identity + reduksi memori
├── features.py           # feature engineering, missing values, encoding
├── evaluate.py           # metrik (ROC-AUC, PR-AUC, dll) + plot
├── train_lgbm.py         # >> pipeline ML: LightGBM + Optuna + MLflow
├── train_dl.py           # >> pipeline DL: MLP + embeddings + Optuna + MLflow
├── generate_synthetic.py # data sintetik utk uji pipeline tanpa download
├── requirements.txt
└── data/                 # taruh train_transaction.csv & train_identity.csv di sini
```

## 2. Setup (Python 3.11.9)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

> Catatan torch: kalau pakai GPU, install torch sesuai panduan resmi
> https://pytorch.org/get-started/locally/ (versi CPU sudah cukup untuk dataset ini).

## 3. Dataset

Download dari Kaggle **IEEE-CIS Fraud Detection** → letakkan di folder `data/`:
- `data/train_transaction.csv`
- `data/train_identity.csv`

(Pipeline tetap jalan walau `train_identity.csv` tidak ada — identity-nya di-skip.)

## 4. Cara Menjalankan

**Uji cepat dulu tanpa download** (data palsu, hanya untuk memastikan kode jalan):
```bash
python generate_synthetic.py --rows 20000
python train_lgbm.py --nrows 20000 --trials 5
```

**Run sesungguhnya** (pakai dataset asli di `data/`):
```bash
python train_lgbm.py              # LightGBM (model utama, paling kuat di tabular)
python train_dl.py                # Deep Learning (MLP + entity embeddings)
```

Opsi berguna:
```bash
python train_lgbm.py --nrows 100000   # subset, lebih cepat
python train_lgbm.py --trials 50      # tuning lebih dalam
python train_dl.py --trials 15
```

## 5. Lihat Hasil di MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# buka http://127.0.0.1:5000
```
Di sana ada params, metrik (`test_roc_auc`, `test_pr_auc`, ...), dan artefak
(ROC/PR curve, confusion matrix, feature importance, model).

---

## 6. Bagaimana Tiap Syarat Tugas Dipenuhi

| Syarat tugas | Diimplementasikan di |
|---|---|
| Pakai tabel transaction **dan** identity | `data_utils.load_data()` — LEFT JOIN di `TransactionID` |
| Data cleaning & preprocessing | `data_utils.reduce_mem_usage`, `features.add_basic_features` |
| Handle missing values | tree: NaN dibiarkan (native LightGBM); DL: impute median + flag `n_missing` |
| Handle class imbalance | LightGBM `scale_pos_weight`; DL `pos_weight` pada BCEWithLogitsLoss |
| Feature engineering / selection | `features.py` (log amount, desimal sen, jam/hari, freq-encoding, embeddings) |
| Model ML / DL | `train_lgbm.py` (LightGBM) + `train_dl.py` (MLP) |
| Hyperparameter tuning (**Optuna**) | objective Optuna di kedua script (maksimalkan ROC-AUC valid) |
| Evaluasi metrik yang sesuai | `evaluate.py`: **ROC-AUC** (metrik resmi), **PR-AUC**, precision/recall/F1, confusion matrix |
| **MLflow** tracking | `mlflow.start_run`, log params/metrics/artifacts ke backend SQLite |

## 7. Catatan Metodologi

- **Kenapa ROC-AUC + PR-AUC?** Data sangat imbalanced (~3.5% fraud). Accuracy menyesatkan
  (tebak "semua legit" sudah 96%+). ROC-AUC adalah metrik resmi kompetisi; PR-AUC lebih
  sensitif terhadap performa di kelas minoritas.
- **Mencegah leakage:** transform (impute, standardize, frequency-encoding) di-*fit* hanya
  pada train split, lalu diterapkan ke valid/test.
- **Split temporal:** set `SPLIT_STRATEGY = "time"` di `config.py` untuk train pada transaksi
  paling awal & test pada paling akhir — lebih realistis untuk fraud detection di produksi.
- **LightGBM vs DL:** untuk data tabular seperti ini, gradient boosting (LightGBM) biasanya
  mengungguli neural network. MLP disertakan untuk memenuhi syarat "deep learning" dan sebagai
  pembanding.

## 8. Roadmap Lanjutan (opsional, untuk nilai lebih)
- Cross-validation (StratifiedKFold) di dalam objective Optuna, bukan single fold.
- Feature selection berbasis importance / null-importance.
- Ensemble LightGBM + DL (rata-rata probabilitas).
- Kalibrasi probabilitas (Platt / isotonic) bila output butuh probabilitas terkalibrasi.
