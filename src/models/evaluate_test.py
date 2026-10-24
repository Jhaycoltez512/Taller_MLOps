"""Evaluación final y única de la versión congelada (v2) sobre el archivo de test completo.

Condiciones (el script se niega a correr si no se cumplen):
  - evidencias/seleccion.json está congelado y en un commit, y no hay cambios sin commit.
  - evidencias/metricas-test.json no existe todavía (el test se evalúa una sola vez).
El modelo se descarga de Hugging Face por el SHA congelado en una caché vacía y se comprueba
que su SHA-256 y su umbral coinciden con la selección. Nada de esto modifica modelo ni umbral.

Salida:
  evidencias/predicciones-test.csv  row_id, y_true, score_attack, y_pred (una fila por registro)
  evidencias/metricas-test.json     versión, hashes, filas, umbral, matriz de confusión y métricas

Uso: python -m src.models.evaluate_test
"""
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.schema import (
    ATTACK_CAT,
    EXPECTED_SHA256,
    FILES,
    ORIGINAL,
    PREPARED,
    TARGET,
    sha256_file,
    sha256_lines,
)
from src.models.metrics import binary_metrics
from src.models.predict import download_solution, load_solution, predict

SELECTION = Path("evidencias/seleccion.json")
PREDICTIONS = Path("evidencias/predicciones-test.csv")
METRICS = Path("evidencias/metricas-test.json")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    if METRICS.exists():
        raise SystemExit(f"{METRICS} ya existe: el test se evalúa una sola vez.")
    if git("status", "--porcelain"):
        raise SystemExit("Hay cambios sin commit: la evaluación debe quedar ligada a un commit.")
    sel = json.loads(SELECTION.read_text())
    if sel.get("status") != "congelada":
        raise SystemExit("La selección no está congelada.")
    freeze_commit = git("log", "-1", "--format=%H", "--", str(SELECTION))

    with tempfile.TemporaryDirectory() as cache:
        path = download_solution(sel["hf_repo_id"], sel["hf_revision"], cache_dir=cache)
        model, config = load_solution(path)
        model_sha = sha256_file(path / config["model_file"])
    assert model_sha == sel["model_sha256"], "el modelo descargado no es el congelado"
    assert config["threshold"] == sel["threshold"], "el umbral no es el congelado"

    original = ORIGINAL / FILES["test"]
    original_sha = sha256_file(original)
    assert original_sha == EXPECTED_SHA256[FILES["test"]]
    test = pd.read_parquet(PREPARED / "test.parquet")
    n_original = len(pd.read_csv(original, usecols=["id"]))
    assert len(test) == n_original, "el test preparado no cubre todo el archivo publicado"

    pred = predict(model, config, test)
    pred.insert(1, "y_true", test[TARGET].to_numpy())
    pred.to_csv(PREDICTIONS, index=False, float_format="%.17g")

    y, score = test[TARGET], pred["score_attack"]
    m = binary_metrics(y, score, config["threshold"])
    by_cat = pred.assign(attack_cat=test[ATTACK_CAT].to_numpy()).groupby("attack_cat")["y_pred"].agg(["size", "mean"])
    train_groups = set(pd.read_parquet(PREPARED / "train.parquet", columns=["group_id"])["group_id"])
    seen = test["group_id"].isin(train_groups).to_numpy()
    val = sel["candidates"][[c["config_id"] for c in sel["candidates"]].index(sel["config_id"])]

    record = {
        "version": sel["version"],
        "config_id": sel["config_id"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_git_commit": git("rev-parse", "HEAD"),
        "selection_frozen_in_commit": freeze_commit,
        "model": {
            "hf_repo_id": sel["hf_repo_id"],
            "hf_revision": sel["hf_revision"],
            "sha256": model_sha,
            "mlflow_run_id": sel["mlflow_run_id"],
            "loaded_from": "Hugging Face por SHA, caché temporal vacía",
        },
        "dataset": {
            "file": str(original),
            "sha256": original_sha,
            "prepared_file": str(PREPARED / "test.parquet"),
            "prepared_sha256": sha256_file(PREPARED / "test.parquet"),
            "row_ids_sha256": sha256_lines(test["row_id"]),
            "rows_in_published_file": n_original,
            "rows_evaluated": len(test),
            "dvc_prepared_md5": sel["dvc_prepared_md5"],
        },
        "positive_class": "1 = ataque",
        "threshold": config["threshold"],
        "decision_rule": "y_pred = 1 si score_attack >= threshold; score_attack = predict_proba[:, 1]",
        "zero_division_policy": "una razón con denominador 0 vale 0.0 y se lista en metrics.undefined",
        "confusion_matrix": {
            "labels": ["0 normal", "1 ataque"],
            "matrix_rows_true_cols_pred": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        },
        "metrics": {
            k: m[k]
            for k in ("recall_attack", "precision_attack", "fpr", "macro_f1", "f1_attack", "f1_normal", "accuracy", "undefined")
        },
        "threshold_free_metrics": {
            "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score),
        },
        "validation_reference": {
            k: val[k] for k in ("val_recall_attack", "val_precision_attack", "val_fpr", "val_macro_f1")
        },
        "detection_rate_by_attack_cat": {
            cat: {"rows": int(r["size"]), "predicted_attack_rate": float(r["mean"])} for cat, r in by_cat.iterrows()
        },
        "overlap_with_train": {
            "note": "filas de test cuyo vector de entradas aparece en train (ver auditoria.json); se reportan por separado para limitar conclusiones",
            "rows_seen_in_train": int(seen.sum()),
            "metrics_seen_in_train": binary_metrics(y[seen], score[seen], config["threshold"]),
            "metrics_not_seen_in_train": binary_metrics(y[~seen], score[~seen], config["threshold"]),
        },
        "outputs": {"predictions_file": str(PREDICTIONS)},
    }
    METRICS.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(
        f"test {len(test)} filas | recall {m['recall_attack']:.4f} precision {m['precision_attack']:.4f} "
        f"FPR {m['fpr']:.4f} macro-F1 {m['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
