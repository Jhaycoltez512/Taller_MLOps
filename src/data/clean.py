"""Carga y limpieza determinista de los CSV crudos de UNSW-NB15 (UNSW-NB15_{1..4}.csv).

Lo usan el EDA (src/eda/raw_eda.py) y el preprocesamiento (src/data/preprocess.py),
para que ambos vean exactamente los mismos datos. Las anomalías que se corrigen aquí
están documentadas en reports/eda_raw/README.md (§2).
"""
from pathlib import Path

import pandas as pd

RAW_FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
CATEGORICAL = ["proto", "state", "service"]
ID_COLS = ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"]
TARGETS = ["attack_cat", "Label"]
# Normaliza las variantes de nombre de categoría que aparecen en los crudos.
CAT_FIX = {"Backdoors": "Backdoor"}
# La captura tiene dos sesiones separadas por ~26 días; se cortan por un hueco mayor a esto.
SESSION_GAP = "6h"


def parse_port(s):
    """Puertos: enteros, hex ('0x000b') o '-'. Devuelve Int64 con NA para '-'."""
    s = s.astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    hexmask = s.str.startswith("0x")
    out[hexmask] = s[hexmask].map(lambda x: int(x, 16))
    return out.astype("Int64")


def load_raw(raw):
    """Lee los cuatro crudos con los nombres de NUSW-NB15_features.csv. Devuelve (df, features)."""
    raw = Path(raw)
    feats = pd.read_csv(raw / "NUSW-NB15_features.csv", encoding="latin-1")
    feats.columns = feats.columns.str.strip()
    names = [n.strip().replace(" ", "") for n in feats["Name"]]
    frames = []
    for f in RAW_FILES:
        d = pd.read_csv(raw / f, header=None, names=names, encoding="utf-8-sig", low_memory=False)
        d["file"] = f
        frames.append(d)
    return pd.concat(frames, ignore_index=True), feats


def clean_raw(df):
    """Corrige tipos y etiquetas, y añade `ts` (UTC) y `session` (1 o 2). No elimina filas."""
    df = df.copy()
    df["sport"] = parse_port(df["sport"])
    df["dsport"] = parse_port(df["dsport"])
    df["ct_ftp_cmd"] = pd.to_numeric(df["ct_ftp_cmd"].astype(str).str.strip(), errors="coerce")
    df["attack_cat"] = df["attack_cat"].str.strip().replace(CAT_FIX).fillna("Normal")
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["Stime"], unit="s", utc=True)
    order = df["ts"].sort_values()
    session = (order.diff() > pd.Timedelta(SESSION_GAP)).cumsum() + 1
    df["session"] = session.reindex(df.index)
    return df
