"""Exporta los runs de MLflow a evidencias consultables sin el servidor local.

Salida:
  evidencias/experimentos.csv                  una fila por configuración (último run de cada una)
  evidencias/errores_por_categoria.csv         tasa de detección en validación por attack_cat
  evidencias/mlflow/runs.json                  info, parámetros, métricas y tags de esos runs
  evidencias/mlflow/<run_id>/                  artefactos relevantes: configuración, umbral, métricas,
                                               curva de umbral, predicciones de validación y modelo si
                                               pesa menos de --max-model-mb (si no, solo su SHA-256)

Uso: python -m src.models.export_mlflow
"""
import argparse
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.data.schema import ATTACK_CAT, PREPARED
from src.models.train import EXPERIMENT, TRACKING_URI

OUT = Path("evidencias")
COLUMNS = {
    "tags.config_id": "config_id",
    "run_id": "run_id",
    "tags.family": "family",
    "tags.hypothesis": "hypothesis",
    "tags.change": "change",
    "tags.config_file": "config_file",
    "params.seed": "seed",
    "tags.git_commit": "git_commit",
    "tags.data.prepared_dvc_md5": "dvc_prepared_md5",
    "tags.data.original_dvc_md5": "dvc_original_md5",
    "params.threshold": "threshold",
    "metrics.val_recall_attack": "val_recall_attack",
    "metrics.val_precision_attack": "val_precision_attack",
    "metrics.val_fpr": "val_fpr",
    "metrics.val_macro_f1": "val_macro_f1",
    "metrics.val_accuracy": "val_accuracy",
    "metrics.val_tp": "val_tp",
    "metrics.val_fp": "val_fp",
    "metrics.val_tn": "val_tn",
    "metrics.val_fn": "val_fn",
    "metrics.val_at_0.5_recall_attack": "val@0.5_recall_attack",
    "metrics.val_at_0.5_precision_attack": "val@0.5_precision_attack",
    "metrics.val_at_0.5_fpr": "val@0.5_fpr",
    "metrics.val_at_0.5_macro_f1": "val@0.5_macro_f1",
    "metrics.val_roc_auc": "val_roc_auc",
    "metrics.val_pr_auc": "val_pr_auc",
    "metrics.fit_seconds": "fit_seconds",
    "metrics.model_megabytes": "model_megabytes",
    "tags.model_sha256": "model_sha256",
    "start_time": "start_time",
}


def write_error_table(export, runs):
    """Fracción de filas de validación marcadas como ataque, por attack_cat y configuración.

    Para Normal es la tasa de falsos positivos; para las demás categorías, el recall.
    """
    val = pd.read_parquet(PREPARED / "val.parquet", columns=["row_id", ATTACK_CAT])
    table = {}
    for run_id, cid in zip(runs["run_id"], runs["tags.config_id"]):
        pred = pd.read_csv(export / run_id / "model" / "val_predictions.csv").merge(val, on="row_id")
        table[cid] = pred.groupby(ATTACK_CAT)["y_pred"].mean()
    out = pd.DataFrame(table)
    out.insert(0, "n_val", val[ATTACK_CAT].value_counts())
    out.sort_values("n_val", ascending=False).round(4).to_csv(OUT / "errores_por_categoria.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-mb", type=float, default=20)
    ap.add_argument("--conclusions", default="evidencias/conclusiones_experimentos.yaml")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time ASC"])
    runs = runs[runs["status"] == "FINISHED"].drop_duplicates("tags.config_id", keep="last")
    runs = runs.sort_values("tags.config_id")

    table = runs[list(COLUMNS)].rename(columns=COLUMNS)
    conclusions = {}
    if Path(args.conclusions).exists():
        conclusions = yaml.safe_load(open(args.conclusions)) or {}
    table["validation_conclusion"] = table["config_id"].map(conclusions).fillna("")
    table["start_time"] = table["start_time"].astype(str)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "experimentos.csv", index=False)

    export = OUT / "mlflow"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True)
    client = mlflow.MlflowClient()
    records = []
    for run_id in runs["run_id"]:
        run = client.get_run(run_id)
        records.append(
            {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "experiment": EXPERIMENT,
                "status": run.info.status,
                "start_time_ms": run.info.start_time,
                "end_time_ms": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.source")},
            }
        )
        local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=str(export / run_id)))
        model = local / "model" / "model.skops"
        if model.exists() and model.stat().st_size / 1e6 > args.max_model_mb:
            note = {
                "file": "model/model.skops",
                "omitted": f"pesa {model.stat().st_size / 1e6:.1f} MB (> {args.max_model_mb} MB)",
                "sha256": run.data.tags["model_sha256"],
                "where": "artefacto del run en el MLflow local (mlruns/); v1/v2 están en Hugging Face",
            }
            model.unlink()
            (local / "model" / "model.skops.omitted.json").write_text(json.dumps(note, indent=2) + "\n")
    write_error_table(export, runs)
    (export / "runs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(records)} runs -> {OUT / 'experimentos.csv'} y {export}/")


if __name__ == "__main__":
    main()
