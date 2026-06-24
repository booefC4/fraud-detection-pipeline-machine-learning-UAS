"""
config.py
Pusat konfigurasi pipeline. Ubah path & hyperparameter di sini saja.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MLRUNS_DIR = ROOT / "mlruns"

TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
IDENTITY_CSV = DATA_DIR / "train_identity.csv"   # optional; auto-skip kalau tidak ada

OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Kolom inti
# ----------------------------------------------------------------------
ID_COL = "TransactionID"
TARGET = "isFraud"
TIME_COL = "TransactionDT"

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------
# Split & training
# ----------------------------------------------------------------------
TEST_SIZE = 0.2          # ukuran test set (hold-out final)
VALID_SIZE = 0.2         # ukuran validation (di dalam train, untuk early stopping/Optuna)
SPLIT_STRATEGY = "stratified"   # "stratified" atau "time"

# ----------------------------------------------------------------------
# Optuna
# ----------------------------------------------------------------------
N_TRIALS = 30            # jumlah trial tuning; naikkan kalau punya waktu/compute
OPTUNA_TIMEOUT = None    # detik; None = tanpa batas waktu

# ----------------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------------
MLRUNS_DIR.mkdir(exist_ok=True)
# SQLite backend: kompatibel dengan MLflow 2.x & 3.x (file-store sudah deprecated di 3.x)
MLFLOW_TRACKING_URI = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_LGBM = "fraud_lightgbm"
EXPERIMENT_DL = "fraud_deeplearning"

# ----------------------------------------------------------------------
# Deep Learning
# ----------------------------------------------------------------------
DL_EPOCHS = 15
DL_BATCH_SIZE = 2048
DL_MAX_VOCAB = 200       # cap kardinalitas kategori untuk embedding
DL_EARLY_STOP_PATIENCE = 4
