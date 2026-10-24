"""Selección y congelación de v2 usando solo validación -> evidencias/seleccion.json.

Paso 1 (antes de publicar v2):  python -m src.models.select
    Aplica la regla de configs/seleccion.yaml sobre evidencias/experimentos.csv.
Paso 2 (después de publicar v2): python -m src.models.select --freeze
    Añade la revisión de Hugging Face, verifica el SHA-256 del modelo y marca la decisión como
    congelada con fecha. El commit que contiene este archivo (tag v2) es anterior a cualquier
    evaluación en test.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

OUT = Path("evidencias/seleccion.json")


def choose(table, rule):
    ok = table[table["val_fpr"] <= rule["max_val_fpr"]]
    if ok.empty:
        raise SystemExit("ninguna configuración cumple la FPR máxima de validación")
    best = ok[rule["metric"]].max()
    tied = ok[ok[rule["metric"]] >= best - rule["tie_tolerance"]]
    return tied.sort_values(rule["tie_break"]).iloc[0], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    sel_cfg = yaml.safe_load(open("configs/seleccion.yaml"))

    if args.freeze:
        sel = json.loads(OUT.read_text())
        v2 = json.loads(Path("evidencias/versiones/v2.json").read_text())
        assert v2["config_id"] == sel["config_id"] and v2["model_sha256"] == sel["model_sha256"]
        sel.update(
            status="congelada",
            frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            freeze_evidence="commit Git que añade este archivo congelado (tag v2), anterior a "
            "evidencias/metricas-test.json; el test no se había leído al tomar la decisión",
            hf_repo_id=v2["hf_repo_id"],
            hf_revision=v2["hf_revision"],
            hf_tag=v2["hf_tag"],
            hf_revision_url=v2["hf_revision_url"],
        )
        OUT.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")
        print(f"v2 congelada: {sel['config_id']} @ {sel['hf_revision']}")
        return

    table = pd.read_csv("evidencias/experimentos.csv")
    row, eligible = choose(table, sel_cfg["rule"])
    v1 = table.set_index("config_id").loc[sel_cfg["v1"]]
    cols = ["config_id", "run_id", "threshold", "val_recall_attack", "val_precision_attack", "val_fpr",
            "val_macro_f1", "model_megabytes"]
    sel = {
        "version": "v2",
        "status": "elegida (pendiente de publicar y congelar)",
        "decided_with": "solo validación (evidencias/experimentos.csv); test no leído",
        "config_id": row["config_id"],
        "mlflow_run_id": row["run_id"],
        "model_sha256": row["model_sha256"],
        "threshold": float(row["threshold"]),
        "threshold_rule": "máximo recall de ataque con FPR <= 5 % en predicciones out-of-fold de train",
        "decision_rule": "ataque (1) si score_attack >= threshold",
        "git_training_commit": row["git_commit"],
        "config_file": row["config_file"],
        "dvc_original_md5": row["dvc_original_md5"],
        "dvc_prepared_md5": row["dvc_prepared_md5"],
        "selection_rule": sel_cfg["rule"],
        "reason": (
            f"Mayor recall de validación ({row['val_recall_attack']:.4f}) entre las configuraciones con "
            f"FPR de validación <= {sel_cfg['rule']['max_val_fpr']}; FPR {row['val_fpr']:.4f}, "
            f"macro-F1 {row['val_macro_f1']:.4f}."
        ),
        "candidates": table[cols].assign(eligible=table["config_id"].isin(eligible["config_id"])).to_dict("records"),
        "v1": {"config_id": sel_cfg["v1"], "mlflow_run_id": v1["run_id"], "threshold": float(v1["threshold"])},
        "change_vs_v1": sel_cfg["change_vs_v1"],
        "tradeoffs": sel_cfg["tradeoffs"],
    }
    OUT.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")
    print(f"v2 = {row['config_id']} ({sel['reason']})")


if __name__ == "__main__":
    main()
