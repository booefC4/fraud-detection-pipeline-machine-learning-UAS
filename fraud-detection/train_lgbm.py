"""
train_lgbm.py
Pipeline utama (Machine Learning): LightGBM + Optuna + MLflow.

Alur:
  1. Load & merge data
  2. Feature engineering (jalur tree)
  3. Split: train / valid / test (stratified atau time-based)
  4. Optuna: cari hyperparameter terbaik (maksimalkan ROC-AUC valid)
  5. Train model final dengan best params + handle imbalance (scale_pos_weight)
  6. Evaluasi di test set + simpan plot
  7. Log semuanya ke MLflow

Jalankan:  python train_lgbm.py
           python train_lgbm.py --nrows 50000     # subset cepat untuk uji coba
"""
import argparse
import warnings

import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split

import config
import data_utils
import features
import evaluate

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------
# Split helpers
# ----------------------------------------------------------------------
def split_data(X, y, time_values=None):
    """Pisahkan jadi train / valid / test."""
    if config.SPLIT_STRATEGY == "time" and time_values is not None:
        # train pada transaksi paling awal, test pada paling akhir (mirip produksi)
        order = np.argsort(time_values.values)
        X, y = X.iloc[order], y.iloc[order]
        n = len(X)
        n_test = int(n * config.TEST_SIZE)
        n_valid = int(n * config.VALID_SIZE)
        X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]
        X_valid, y_valid = X.iloc[-(n_test + n_valid):-n_test], y.iloc[-(n_test + n_valid):-n_test]
        X_train, y_train = X.iloc[:-(n_test + n_valid)], y.iloc[:-(n_test + n_valid)]
    else:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_tmp, y_tmp, test_size=config.VALID_SIZE, stratify=y_tmp, random_state=config.SEED)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def scale_pos_weight(y):
    """Rasio negatif:positif untuk mengkompensasi class imbalance."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


# ----------------------------------------------------------------------
# Optuna objective
# ----------------------------------------------------------------------
def make_objective(X_train, y_train, X_valid, y_valid, cat_cols, spw):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols,
                         reference=dtrain, free_raw_data=False)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": config.SEED,
            "feature_pre_filter": False,  # izinkan min_child_samples berubah antar-trial
            "scale_pos_weight": spw,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)],
        )
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_valid, preds)
        trial.set_user_attr("best_iteration", model.best_iteration)
        return auc

    return objective


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(nrows=None, n_trials=None):
    n_trials = n_trials or config.N_TRIALS

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_LGBM)

    # 1-2. Load + FE
    df = data_utils.load_data(nrows=nrows)
    time_values = df[config.TIME_COL].copy() if config.TIME_COL in df else None
    X, y, cat_cols = features.prepare_for_tree(df)
    print(f"[fe] fitur: {X.shape[1]} | kategori: {len(cat_cols)} | "
          f"fraud rate: {y.mean():.4f}")

    # 3. Split
    if time_values is not None:
        time_values = time_values.loc[X.index]
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, time_values)
    spw = scale_pos_weight(y_tr)
    print(f"[split] train={len(X_tr)} valid={len(X_va)} test={len(X_te)} "
          f"| scale_pos_weight={spw:.2f}")

    with mlflow.start_run(run_name="lgbm_optuna"):
        mlflow.log_params({
            "model": "lightgbm", "n_trials": n_trials,
            "split_strategy": config.SPLIT_STRATEGY,
            "scale_pos_weight": round(spw, 3),
            "n_features": X.shape[1], "n_categorical": len(cat_cols),
            "fraud_rate": round(float(y.mean()), 5),
        })

        # 4. Optuna
        print(f"[optuna] menjalankan {n_trials} trial ...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.SEED))
        objective = make_objective(X_tr, y_tr, X_va, y_va, cat_cols, spw)
        study.optimize(objective, n_trials=n_trials, timeout=config.OPTUNA_TIMEOUT)

        best = study.best_params
        best_iter = study.best_trial.user_attrs.get("best_iteration", 1000)
        print(f"[optuna] best valid AUC = {study.best_value:.5f}")
        mlflow.log_metric("valid_auc_best", study.best_value)
        mlflow.log_params({f"best__{k}": v for k, v in best.items()})

        # 5. Train final: gabung train+valid, pakai best params
        final_params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "verbosity": -1, "seed": config.SEED, "scale_pos_weight": spw, **best,
        }
        import pandas as pd
        X_trval = pd.concat([X_tr, X_va]); y_trval = pd.concat([y_tr, y_va])
        dtrval = lgb.Dataset(X_trval, label=y_trval, categorical_feature=cat_cols)
        final_model = lgb.train(
            final_params, dtrval,
            num_boost_round=int(best_iter * 1.1) + 50,
            callbacks=[lgb.log_evaluation(0)],
        )

        # 6. Evaluasi di test
        test_prob = final_model.predict(X_te)
        thr = evaluate.best_f1_threshold(y_va, final_model.predict(X_va))
        metrics = evaluate.compute_metrics(y_te, test_prob, threshold=thr)
        evaluate.pretty_print(metrics, "LIGHTGBM — TEST SET")

        # 7. Logging
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        prefix = (config.OUTPUT_DIR / "lgbm").as_posix()
        for p in evaluate.save_plots(y_te, test_prob, prefix, threshold=thr):
            mlflow.log_artifact(p)

        # feature importance
        imp = sorted(zip(X.columns, final_model.feature_importance(importance_type="gain")),
                     key=lambda t: -t[1])[:30]
        imp_path = config.OUTPUT_DIR / "lgbm_feature_importance.txt"
        imp_path.write_text("\n".join(f"{n}\t{v:.1f}" for n, v in imp))
        mlflow.log_artifact(imp_path.as_posix())

        model_path = config.OUTPUT_DIR / "lgbm_model.txt"
        final_model.save_model(model_path.as_posix())
        mlflow.log_artifact(model_path.as_posix())
        mlflow.lightgbm.log_model(final_model, name="model")

        print(f"\n[done] artefak di {config.OUTPUT_DIR} | MLflow: {config.MLRUNS_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                    help="batasi jumlah baris (uji coba cepat)")
    ap.add_argument("--trials", type=int, default=None, help="jumlah trial Optuna")
    args = ap.parse_args()
    main(nrows=args.nrows, n_trials=args.trials)
