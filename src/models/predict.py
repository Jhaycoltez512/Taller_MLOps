"""Carga una solución completa (modelo + configuración + umbral) y predice.

Desde Hugging Face, por revisión inmutable (SHA completo de commit):
    python -m src.models.predict --repo-id <usuario>/<repo> --revision <sha> \
        --input data/prepared/val.parquet --output preds.csv

Desde una carpeta local con model.skops y model_config.json:
    python -m src.models.predict --model-dir models/exp04_hist_gradient_boosting --input ... --output ...

Salida: row_id, score_attack (probabilidad de ataque) y y_pred (1 si score_attack >= umbral).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.schema import sha256_file
from src.models.io import load_model

SOLUTION_FILES = ["model.skops", "model_config.json"]


def download_solution(repo_id, revision, cache_dir=None):
    """Descarga los archivos de la solución en una revisión concreta y devuelve la carpeta."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=["*"])
    return Path(path)


def load_solution(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "model_config.json").read_text())
    sha = sha256_file(model_dir / config["model_file"])
    if sha != config["model_sha256"]:
        raise ValueError(f"SHA-256 del modelo {sha} no coincide con model_config.json ({config['model_sha256']})")
    return load_model(model_dir / config["model_file"]), config


def predict(model, config, df):
    score = model.predict_proba(df[config["features"]])[:, 1]
    return pd.DataFrame(
        {
            "row_id": df["row_id"].to_numpy() if "row_id" in df else range(len(df)),
            "score_attack": score,
            "y_pred": (score >= config["threshold"]).astype(int),
        }
    )


def read_table(path):
    path = Path(path)
    # round_trip: los floats del CSV se leen bit a bit igual que se escribieron.
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir")
    src.add_argument("--repo-id")
    ap.add_argument("--revision", help="SHA completo del commit en Hugging Face (obligatorio con --repo-id)")
    ap.add_argument("--cache-dir", help="caché de Hugging Face (use una vacía para probar la recuperación)")
    ap.add_argument("--input", required=True, help=".parquet o .csv con las 42 entradas (y row_id opcional)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.repo_id:
        if not args.revision:
            ap.error("--revision es obligatorio con --repo-id")
        model_dir = download_solution(args.repo_id, args.revision, args.cache_dir)
    else:
        model_dir = Path(args.model_dir)
    model, config = load_solution(model_dir)
    out = predict(model, config, read_table(args.input))
    out.to_csv(args.output, index=False)
    print(
        f"{len(out)} filas | {config.get('version', config['id'])} | umbral {config['threshold']} | "
        f"modelo sha256 {config['model_sha256']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
