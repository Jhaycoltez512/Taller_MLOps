"""Prueba de recuperación: muestra fija de validación, salidas esperadas y comparación.

  python -m src.models.recovery sample
      (en el entorno original) Elige 30 filas fijas de validación (3 por attack_cat, semilla 42)
      y guarda, para v1 y v2, las salidas que produjeron sus runs originales de MLflow
      (evidencias/mlflow/<run_id>/model/val_predictions.csv).
  python -m src.models.recovery verify-data
      (en el entorno nuevo) Comprueba que los datos recuperados con DVC tienen las huellas de
      evidencias/auditoria.json y que la muestra coincide con las filas recuperadas.
  python -m src.models.recovery check --outdir <dir>
      (en el entorno nuevo) Compara las predicciones recuperadas de v1/v2 con las esperadas.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.schema import ATTACK_CAT, EXPECTED_SHA256, PREPARED, sha256_file
from src.models.predict import read_table

DIR = Path("evidencias/recuperacion")
SAMPLE = DIR / "entrada.csv"
PER_CATEGORY = 3
SEED = 42
TOLERANCE = 1e-9  # diferencia absoluta máxima aceptada en score_attack


def versions():
    return {v: json.loads(Path(f"evidencias/versiones/{v}.json").read_text()) for v in ("v1", "v2")}


def sample():
    val = pd.read_parquet(PREPARED / "val.parquet")
    rows = val.groupby(ATTACK_CAT).sample(n=PER_CATEGORY, random_state=SEED).sort_values("row_id")
    features = [c for c in val.columns if c not in ("group_id",)]
    rows[features].to_csv(SAMPLE, index=False, float_format="%.17g")
    meta = {"rows": len(rows), "seed": SEED, "per_attack_cat": PER_CATEGORY, "file": str(SAMPLE),
            "sha256": sha256_file(SAMPLE), "expected": {}}
    for v, rec in versions().items():
        saved = pd.read_csv(f"evidencias/mlflow/{rec['mlflow_run_id']}/model/val_predictions.csv")
        exp = rows[["row_id"]].merge(saved, on="row_id", how="left")
        assert exp["score_attack"].notna().all()
        path = DIR / f"esperado_{v}.csv"
        exp[["row_id", "score_attack", "y_pred"]].to_csv(path, index=False, float_format="%.17g")
        meta["expected"][v] = {
            "file": str(path),
            "sha256": sha256_file(path),
            "source": f"predicciones de validación del run original {rec['mlflow_run_id']}",
            "hf_revision": rec["hf_revision"],
            "model_sha256": rec["model_sha256"],
            "threshold": rec["threshold"],
        }
    (DIR / "muestra.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} filas -> {SAMPLE}")


def verify_data():
    audit = json.loads(Path("evidencias/auditoria.json").read_text())
    result = {}
    for name, f in audit["files"].items():
        sha = sha256_file(f["file"])
        result[f["file"]] = {"sha256": sha, "matches_audit": sha == f["sha256"],
                             "matches_annex": sha == EXPECTED_SHA256[Path(f["file"]).name]}
    for name, part in audit["splits"]["parts"].items():
        sha = sha256_file(part["file"])
        result[part["file"]] = {"sha256": sha, "matches_audit": sha == part["sha256_file"]}
    sample = read_table(SAMPLE)
    val = pd.read_parquet(PREPARED / "val.parquet").set_index("row_id").loc[sample["row_id"]]
    cols = [c for c in sample.columns if c != "row_id"]
    same = all(
        (sample[c].to_numpy() == val[c].to_numpy()).all()
        if pd.api.types.is_numeric_dtype(sample[c])
        else (sample[c].astype(str).to_numpy() == val[c].astype(str).to_numpy()).all()
        for c in cols
    )
    result["sample_matches_recovered_val"] = bool(same)
    ok = all(v["matches_audit"] for k, v in result.items() if isinstance(v, dict)) and same
    print(json.dumps(result, indent=2))
    print("DATOS OK" if ok else "DATOS NO COINCIDEN")
    sys.exit(0 if ok else 1)


def check(outdir):
    outdir = Path(outdir)
    report = {"tolerance_abs_score": TOLERANCE, "versions": {}}
    for v, rec in versions().items():
        exp = read_table(DIR / f"esperado_{v}.csv")
        got = read_table(outdir / f"recuperado_{v}.csv")
        same_rows = exp["row_id"].tolist() == got["row_id"].tolist()
        diff = (exp["score_attack"] - got["score_attack"]).abs().max()
        labels = bool((exp["y_pred"] == got["y_pred"]).all())
        report["versions"][v] = {
            "hf_revision": rec["hf_revision"],
            "rows": len(got),
            "same_row_ids": same_rows,
            "labels_match": labels,
            "max_abs_score_diff": float(diff),
            "within_tolerance": bool(diff <= TOLERANCE),
            "predicted_attacks": int(got["y_pred"].sum()),
            "recovered_file_sha256": sha256_file(outdir / f"recuperado_{v}.csv"),
            "passed": bool(same_rows and labels and diff <= TOLERANCE),
        }
    report["passed"] = all(r["passed"] for r in report["versions"].values())
    (outdir / "comparacion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("RECUPERACIÓN OK" if report["passed"] else "RECUPERACIÓN FALLIDA")
    sys.exit(0 if report["passed"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sample", "verify-data", "check"])
    ap.add_argument("--outdir")
    args = ap.parse_args()
    if args.command == "sample":
        sample()
    elif args.command == "verify-data":
        verify_data()
    else:
        check(args.outdir)


if __name__ == "__main__":
    main()
