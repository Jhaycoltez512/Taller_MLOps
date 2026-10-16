"""Preprocesamiento de los crudos de UNSW-NB15 -> data/processed/unsw_nb15.parquet.

Pasos (justificados en reports/eda_raw/README.md):
  1. Limpieza común (src/data/clean.py): puertos, `ct_ftp_cmd`, `attack_cat`, sesión.
  2. Nulos FTP/HTTP -> 0 (en la sesión 2 significan "no aplica"; sin indicador de nulo,
     que filtraría la sesión) e `is_ftp_login` acotada a {0, 1}.
  3. Se descartan IPs, puertos y timestamps como features (identifican el escenario).
  4. Nombres de columna en minúscula y alineados con los splits oficiales.
  5. Deduplicación por (features, label) conservando el registro más antiguo, y
     eliminación de vectores con etiquetas contradictorias.

Solo hace transformaciones deterministas y sin estado: todo lo que se ajusta a los datos
(codificación de categóricas, escalado, agrupar valores raros) va en el pipeline del modelo
y se ajusta solo con train.

Salida: data/processed/unsw_nb15.parquet y métricas en reports/data/preprocess.json.
Uso: python -m src.data.preprocess
"""
import json
from pathlib import Path

import yaml

from src.data.clean import CATEGORICAL, ID_COLS, clean_raw, load_raw

RAW = Path("data/raw")
OUT = Path("data/processed/unsw_nb15.parquet")
METRICS = Path("reports/data/preprocess.json")

FTP_HTTP = ["ct_ftp_cmd", "is_ftp_login", "ct_flw_http_mthd"]
# Nombres de los crudos -> nombres de los splits oficiales (el resto solo pasa a minúscula).
RENAME = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
    "Label": "label",
}
TARGETS = ["attack_cat", "label"]
META = ["session", "stime"]


def feature_columns(df):
    return [c for c in df.columns if c not in TARGETS + META]


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    df, _ = load_raw(RAW)
    df = clean_raw(df)
    n_raw = len(df)

    df[FTP_HTTP] = df[FTP_HTTP].fillna(0)
    df["is_ftp_login"] = df["is_ftp_login"].clip(upper=1)
    for c in FTP_HTTP:
        df[c] = df[c].astype("int64")

    stime = df["Stime"]
    df = df.drop(columns=ID_COLS + ["file", "ts"])
    df["stime"] = stime
    df = df.rename(columns=lambda c: RENAME.get(c, c.lower()))
    # Orden estable por tiempo: al deduplicar se conserva el registro más antiguo.
    df = df.sort_values("stime", kind="stable").reset_index(drop=True)

    feats = feature_columns(df)
    stats = {"rows_raw": n_raw}
    if params["dedup"]:
        dup = df.duplicated(subset=feats + ["label"])
        stats["dup_removed"] = int(dup.sum())
        stats["dup_removed_by_session"] = {
            str(k): int(v) for k, v in df.loc[dup, "session"].value_counts().sort_index().items()
        }
        df = df[~dup]
    if params["drop_conflicting"]:
        conflict = df.groupby(feats, dropna=False)["label"].transform("nunique") > 1
        stats["conflicting_removed"] = int(conflict.sum())
        df = df[~conflict]
    df = df.reset_index(drop=True)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df["attack_cat"] = df["attack_cat"].astype("category")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    stats.update(
        rows=len(df),
        features=len(feats),
        attack_rate=round(float(df["label"].mean()), 5),
        rows_by_session={str(k): int(v) for k, v in df["session"].value_counts().sort_index().items()},
        attack_rate_by_session={
            str(k): round(float(v), 5) for k, v in df.groupby("session")["label"].mean().items()
        },
        attack_cat=df["attack_cat"].value_counts().to_dict(),
        feature_names=feats,
    )
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"{n_raw:,} -> {len(df):,} filas, {len(feats)} features -> {OUT}")


if __name__ == "__main__":
    main()
