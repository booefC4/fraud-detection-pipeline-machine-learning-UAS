"""
train_dl.py
Pipeline Deep Learning: MLP + Entity Embeddings (PyTorch) + Optuna + MLflow.

Kenapa embeddings? Banyak fitur kategori berkardinalitas tinggi (card1, DeviceInfo,
P_emaildomain, dll). Embedding layer mempelajari representasi padat tiap kategori,
lebih baik daripada one-hot yang boros & sparse.

Imbalance ditangani lewat pos_weight pada BCEWithLogitsLoss.

Jalankan:  python train_dl.py
           python train_dl.py --nrows 50000 --trials 10
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
import data_utils
import features
import evaluate

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, n_numeric, embedding_specs, hidden=(256, 128), dropout=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, dim) for n_cat, dim in embedding_specs])
        emb_dim = sum(dim for _, dim in embedding_specs)
        in_dim = n_numeric + emb_dim

        layers, prev = [], in_dim
        self.bn_num = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        parts = []
        if self.bn_num is not None:
            parts.append(self.bn_num(x_num))
        if embs:
            parts.append(torch.cat(embs, dim=1))
        x = torch.cat(parts, dim=1)
        return self.mlp(x).squeeze(1)


# ----------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------
def make_loader(Xn, Xc, y, batch_size, shuffle, drop_last=False):
    tensors = [torch.tensor(Xn), torch.tensor(Xc)]
    if y is not None:
        tensors.append(torch.tensor(y))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=drop_last)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        xn, xc = batch[0].to(DEVICE), batch[1].to(DEVICE)
        out.append(torch.sigmoid(model(xn, xc)).cpu().numpy())
    return np.concatenate(out)


def train_model(model, train_loader, valid_loader, y_valid, pos_weight,
                lr, weight_decay, epochs, patience):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, bad = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(DEVICE), xc.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xn, xc), yb)
            loss.backward()
            opt.step()

        val_prob = predict_proba(model, valid_loader)
        auc = roc_auc_score(y_valid, val_prob)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {
                k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(nrows=None, n_trials=None):
    n_trials = n_trials or 10

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_DL)

    # Load + FE
    df = data_utils.load_data(nrows=nrows)
    y_all = df[config.TARGET].astype("int8").values

    # split index dulu (stratified) lalu fit preprocessor HANYA di train (no leakage)
    idx = np.arange(len(df))
    idx_tmp, idx_te = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=y_all, random_state=config.SEED)
    idx_tr, idx_va = train_test_split(
        idx_tmp, test_size=config.VALID_SIZE,
        stratify=y_all[idx_tmp], random_state=config.SEED)

    pre = features.DLPreprocessor().fit(df.iloc[idx_tr])
    Xn_tr, Xc_tr, y_tr = pre.transform(df.iloc[idx_tr])
    Xn_va, Xc_va, y_va = pre.transform(df.iloc[idx_va])
    Xn_te, Xc_te, y_te = pre.transform(df.iloc[idx_te])
    print(f"[fe] numerik={len(pre.num_cols)} kategori={len(pre.cat_cols)} "
          f"| fraud rate={y_tr.mean():.4f}")

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"[imbalance] pos_weight={pos_weight:.2f}")

    train_loader = make_loader(Xn_tr, Xc_tr, y_tr, config.DL_BATCH_SIZE, True,
                               drop_last=True)
    valid_loader = make_loader(Xn_va, Xc_va, None, config.DL_BATCH_SIZE, False)
    test_loader = make_loader(Xn_te, Xc_te, None, config.DL_BATCH_SIZE, False)

    with mlflow.start_run(run_name="dl_mlp_optuna"):
        mlflow.log_params({
            "model": "mlp_embeddings", "n_trials": n_trials,
            "n_numeric": len(pre.num_cols), "n_categorical": len(pre.cat_cols),
            "pos_weight": round(pos_weight, 3), "epochs": config.DL_EPOCHS,
            "batch_size": config.DL_BATCH_SIZE,
        })

        # Optuna
        def objective(trial):
            hidden_choice = trial.suggest_categorical(
                "hidden", ["256,128", "512,256", "256,128,64", "128,64"])
            hidden = tuple(int(h) for h in hidden_choice.split(","))
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            model = TabularMLP(len(pre.num_cols), pre.embedding_specs,
                               hidden=hidden, dropout=dropout)
            auc = train_model(model, train_loader, valid_loader, y_va, pos_weight,
                              lr=lr, weight_decay=wd,
                              epochs=config.DL_EPOCHS,
                              patience=config.DL_EARLY_STOP_PATIENCE)
            return auc

        print(f"[optuna] menjalankan {n_trials} trial ...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.SEED))
        study.optimize(objective, n_trials=n_trials)
        print(f"[optuna] best valid AUC = {study.best_value:.5f}")
        mlflow.log_metric("valid_auc_best", study.best_value)
        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})

        # Train final dengan best params
        bp = study.best_params
        hidden = tuple(int(h) for h in bp["hidden"].split(","))
        final = TabularMLP(len(pre.num_cols), pre.embedding_specs,
                           hidden=hidden, dropout=bp["dropout"])
        train_model(final, train_loader, valid_loader, y_va, pos_weight,
                    lr=bp["lr"], weight_decay=bp["weight_decay"],
                    epochs=config.DL_EPOCHS, patience=config.DL_EARLY_STOP_PATIENCE)

        # Evaluasi test
        test_prob = predict_proba(final, test_loader)
        thr = evaluate.best_f1_threshold(y_va, predict_proba(final, valid_loader))
        metrics = evaluate.compute_metrics(y_te, test_prob, threshold=thr)
        evaluate.pretty_print(metrics, "DEEP LEARNING (MLP) — TEST SET")

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        prefix = (config.OUTPUT_DIR / "dl").as_posix()
        for p in evaluate.save_plots(y_te, test_prob, prefix, threshold=thr):
            mlflow.log_artifact(p)

        model_path = config.OUTPUT_DIR / "dl_model.pt"
        torch.save(final.state_dict(), model_path)
        mlflow.log_artifact(model_path.as_posix())
        print(f"\n[done] artefak di {config.OUTPUT_DIR} | MLflow: {config.MLRUNS_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    main(nrows=args.nrows, n_trials=args.trials)
