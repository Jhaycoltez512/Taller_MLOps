"""Entrena una configuración, elige su umbral sin tocar validación y registra todo en MLflow.

1. Predicciones out-of-fold sobre train (StratifiedGroupKFold con los mismos grupos que el
   split) -> umbral según la regla de params.yaml (train.threshold).
2. Ajuste del Pipeline completo con todo train.
3. Métricas en validación con el umbral elegido y con el umbral de referencia (0.5).
4. MLflow: hipótesis, cambio, parámetros, semilla, commit Git, versión de datos DVC, métricas,
   modelo completo (skops) con su SHA-256, configuración y predicciones de validación.

El test no se lee aquí.

Uso: python -m src.models.train --config configs/experiments/exp01_logreg_base.yaml
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.data.schema import ATTACK_CAT, CATEGORICAL, PREPARED, TARGET, feature_columns, sha256_file
from src.models.io import save_model
from src.models.metrics import binary_metrics, choose_threshold
from src.models.pipeline import build_pipeline

EXPERIMENT = "defiende-el-servidor"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODELS_DIR = Path("models")
DECISION_RULE = "ataque (1) si score_attack >= threshold; score_attack = predict_proba[:, 1]"


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return commit, dirty


def data_version():
    """Referencias DVC (md5 de directorio) de los datos originales y preparados."""
    original = yaml.safe_load(open("data/original.dvc"))["outs"][0]["md5"]
    lock = yaml.safe_load(open("dvc.lock"))
    prepared = next(o["md5"] for o in lock["stages"]["prepare"]["outs"] if o["path"] == "data/prepared")
    return {
        "data.original_dvc_md5": original,
        "data.prepared_dvc_md5": prepared,
        "data.train_sha256": sha256_file(PREPARED / "train.parquet"),
        "data.val_sha256": sha256_file(PREPARED / "val.parquet"),
    }


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_metrics(prefix, m):
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in m.items() if isinstance(v, (int, float))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="permite cambios sin commit (queda marcado)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed, th = params["seed"], params["threshold"]
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("Hay cambios sin commit: el run no quedaría ligado a un commit. Haga commit primero.")
    data_ref = data_version()

    train = pd.read_parquet(PREPARED / "train.parquet")
    val = pd.read_parquet(PREPARED / "val.parquet")
    features = feature_columns(train)
    X, y = train[features], train[TARGET]

    # 1. Umbral con predicciones out-of-fold sobre train (validación no participa).
    cv = StratifiedGroupKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)
    folds = list(cv.split(X, train[ATTACK_CAT], groups=train["group_id"]))
    t0 = time.perf_counter()
    oof = cross_val_predict(build_pipeline(cfg, features, seed), X, y, cv=folds, method="predict_proba")[:, 1]
    oof_seconds = time.perf_counter() - t0
    threshold, curve, feasible = choose_threshold(y, oof, th["rule"], th["max_fpr"], th["grid_step"])

    # 2. Solución final ajustada con todo train.
    model = build_pipeline(cfg, features, seed)
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - t0

    # 3. Validación.
    score = model.predict_proba(val[features])[:, 1]
    val_op = binary_metrics(val[TARGET], score, threshold)
    val_ref = binary_metrics(val[TARGET], score, params["reference_threshold"])
    oof_op = binary_metrics(y, oof, threshold)

    out = MODELS_DIR / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    model_sha = save_model(model, out / "model.skops")
    config = {
        "id": cfg["id"],
        "family": cfg["family"],
        "features": features,
        "categorical": CATEGORICAL,
        "excluded": ["id", "attack_cat", "label"],
        "positive_class": "1 = ataque",
        "decision_rule": DECISION_RULE,
        "threshold": threshold,
        "threshold_rule": {**th, "fitted_on": "predicciones out-of-fold de train", "feasible": feasible},
        "seed": seed,
        "git_commit": commit,
        **data_ref,
        "model_file": "model.skops",
        "model_sha256": model_sha,
        "experiment_config": cfg,
    }
    (out / "model_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(
        {
            "row_id": val["row_id"],
            "y_true": val[TARGET],
            "score_attack": score,
            "y_pred": (score >= threshold).astype(int),
        }
    ).to_csv(out / "val_predictions.csv", index=False)
    pd.DataFrame(curve).drop(columns="undefined").to_csv(out / "oof_threshold_curve.csv", index=False)
    (out / "val_metrics.json").write_text(
        json.dumps({"operating": val_op, "reference_0.5": val_ref, "oof_operating": oof_op}, indent=2) + "\n"
    )

    # 4. MLflow.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=cfg["id"]) as run:
        mlflow.set_tags(
            {
                "config_id": cfg["id"],
                "family": cfg["family"],
                "hypothesis": cfg["hypothesis"],
                "change": cfg["change"],
                "mlflow.note.content": f"Hipótesis: {cfg['hypothesis']}\n\nCambio: {cfg['change']}",
                "git_commit": commit,
                "git_dirty": str(dirty),
                "config_file": args.config,
                "model_sha256": model_sha,
                "decision_rule": DECISION_RULE,
                "threshold_feasible": str(feasible),
                **data_ref,
            }
        )
        mlflow.log_params(flatten({"pre": cfg["preprocessing"], "model": cfg["model"]}))
        mlflow.log_params(
            {
                "seed": seed,
                "cv_folds": params["cv_folds"],
                "threshold_rule": th["rule"],
                "threshold_max_fpr": th["max_fpr"],
                "threshold": threshold,
                "n_features": len(features),
                "n_train": len(train),
                "n_val": len(val),
            }
        )
        log_metrics("val", val_op)
        log_metrics("val_at_0.5", val_ref)
        log_metrics("oof", oof_op)
        mlflow.log_metrics(
            {
                "val_roc_auc": roc_auc_score(val[TARGET], score),
                "val_pr_auc": average_precision_score(val[TARGET], score),
                "fit_seconds": fit_seconds,
                "oof_seconds": oof_seconds,
                "model_megabytes": (out / "model.skops").stat().st_size / 1e6,
            }
        )
        mlflow.log_artifact(args.config, "config")
        mlflow.log_artifacts(str(out), "model")
        run_id = run.info.run_id
    (out / "run.json").write_text(json.dumps({"run_id": run_id, "experiment": EXPERIMENT}, indent=2) + "\n")

    print(
        f"{cfg['id']}: run {run_id} | umbral {threshold:.3f} | val recall {val_op['recall_attack']:.4f} "
        f"precision {val_op['precision_attack']:.4f} FPR {val_op['fpr']:.4f} macro-F1 {val_op['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
