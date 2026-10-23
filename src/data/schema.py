"""Definiciones comunes de los splits oficiales de UNSW-NB15 usados en el taller."""
import hashlib
from pathlib import Path

import pandas as pd

ORIGINAL = Path("data/original")
PREPARED = Path("data/prepared")
FILES = {
    "train": "UNSW_NB15_training-set.csv",
    "test": "UNSW_NB15_testing-set.csv",
}
# Huellas publicadas en el anexo del enunciado (SHA-256 de los CSV sin comprimir).
EXPECTED_SHA256 = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}

ID = "id"
TARGET = "label"  # 0 = normal, 1 = ataque (clase positiva)
ATTACK_CAT = "attack_cat"
EXCLUDED = {
    ID: "identifica la fila; solo para auditoría y para relacionar predicciones",
    ATTACK_CAT: "revela la respuesta (Normal <=> label 0)",
    TARGET: "variable objetivo",
}
CATEGORICAL = ["proto", "service", "state"]
BINARY_CLIP = ["is_ftp_login"]  # declarada binaria, pero trae valores 2 y 4


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDED and c not in ("row_id", "group_id", "split")]


def read_original(name):
    """Lee un CSV oficial con tipos explícitos: categóricas como texto, resto numérico."""
    return pd.read_csv(ORIGINAL / FILES[name], dtype={c: "string" for c in CATEGORICAL + [ATTACK_CAT]})


def row_ids(name, df):
    """Identificador inequívoco archivo/fila: `id` coincide con el número de fila (1..N) del CSV."""
    return name + ":" + df[ID].astype(str)


def group_ids(df, feats):
    """Huella estable (SHA-1) del vector de entradas: filas con las mismas entradas comparten grupo."""
    joined = df[feats].astype(str).agg("|".join, axis=1)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(values):
    """Huella de una lista ordenada de identificadores (independiente del formato de archivo)."""
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
