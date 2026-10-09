"""Particiones de entrenamiento/validación/prueba a partir de data/processed/unsw_nb15.parquet.

Dos esquemas (ver params.yaml y reports/eda_raw/README.md §9):
  - temporal: sesión 1 -> train/val (estratificado), sesión 2 -> test. Evaluación principal:
    mide generalización a datos posteriores con otra proporción de ataques.
  - holdout:  ambas sesiones, train/val/test estratificado. Referencia sin drift temporal.

Salida: data/splits/{temporal,holdout}/{train,val,test}.parquet y reports/data/splits.json.
Uso: python -m src.data.split
"""
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

IN = Path("data/processed/unsw_nb15.parquet")
OUT = Path("data/splits")
METRICS = Path("reports/data/splits.json")


def temporal(df, p, seed, strat):
    dev = df[df["session"] == p["train_session"]]
    test = df[df["session"] == p["test_session"]]
    train, val = train_test_split(dev, test_size=p["val_size"], random_state=seed, stratify=dev[strat])
    return {"train": train, "val": val, "test": test}


def holdout(df, p, seed, strat):
    rest, test = train_test_split(df, test_size=p["test_size"], random_state=seed, stratify=df[strat])
    val_frac = p["val_size"] / (1 - p["test_size"])
    train, val = train_test_split(rest, test_size=val_frac, random_state=seed, stratify=rest[strat])
    return {"train": train, "val": val, "test": test}


def summary(parts):
    return {
        name: {
            "rows": len(d),
            "attack_rate": round(float(d["label"].mean()), 5),
            "attack_cat": d["attack_cat"].value_counts().to_dict(),
        }
        for name, d in parts.items()
    }


def main():
    params = yaml.safe_load(open("params.yaml"))["split"]
    seed, strat = params["seed"], params["stratify"]
    df = pd.read_parquet(IN)

    metrics = {}
    for scheme, fn in (("temporal", temporal), ("holdout", holdout)):
        parts = fn(df, params[scheme], seed, strat)
        out = OUT / scheme
        out.mkdir(parents=True, exist_ok=True)
        for name, d in parts.items():
            d.sort_values("stime", kind="stable").to_parquet(out / f"{name}.parquet", index=False)
        metrics[scheme] = summary(parts)
        print(scheme, {k: len(v) for k, v in parts.items()})

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
