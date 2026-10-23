"""Datos preparados y particiones fijas a partir de los splits oficiales de UNSW-NB15.

Entrada: data/original/UNSW_NB15_{training,testing}-set.csv (versión DVC data/original.dvc).
Salida:  data/prepared/{train,val,test}.parquet y data/prepared/split_assignment.csv.

Política (parámetros en params.yaml, sección `data`):
  - Limpieza sin estado, igual para todas las particiones: categóricas sin espacios e
    `is_ftp_login` acotada a {0, 1}. Nada se ajusta aquí: codificación, escalado e imputación
    van dentro del Pipeline del modelo y se ajustan solo con train.
  - Duplicados por entradas: se conservan (reflejan la distribución publicada) y se agrupan.
  - Grupos de entradas idénticas con etiquetas contradictorias: se excluyen de train/val.
  - Train/val salen solo del archivo de entrenamiento, con StratifiedGroupKFold: ningún grupo
    de entradas idénticas queda repartido entre ambos.
  - Test: el archivo de test completo, sin eliminar filas. No se usa para decidir nada.

Uso: python -m src.data.prepare
"""
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.data.schema import (
    ATTACK_CAT,
    BINARY_CLIP,
    CATEGORICAL,
    PREPARED,
    TARGET,
    feature_columns,
    group_ids,
    read_original,
    row_ids,
)


def clean(df):
    """Reglas deterministas y sin estado; se aplican igual a train, val y test."""
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].str.strip()
    df[BINARY_CLIP] = df[BINARY_CLIP].clip(upper=1)
    return df


def contradictory_groups(df):
    nunique = df.groupby("group_id")[TARGET].nunique()
    return set(nunique[nunique > 1].index)


def main():
    p = yaml.safe_load(open("params.yaml"))["data"]
    train_full = clean(read_original("train"))
    test = clean(read_original("test"))
    feats = feature_columns(train_full)

    for name, df in (("train", train_full), ("test", test)):
        df.insert(0, "row_id", row_ids(name, df))
        df.insert(1, "group_id", group_ids(df, feats))

    excluded = train_full.iloc[0:0]
    if p["drop_contradictory"]:
        bad = train_full["group_id"].isin(contradictory_groups(train_full))
        excluded, train_full = train_full[bad], train_full[~bad]

    sgkf = StratifiedGroupKFold(n_splits=p["n_folds"], shuffle=True, random_state=p["seed"])
    folds = list(sgkf.split(train_full, train_full[p["stratify"]], groups=train_full["group_id"]))
    train_idx, val_idx = folds[p["val_fold"]]
    parts = {
        "train": train_full.iloc[train_idx],
        "val": train_full.iloc[val_idx],
        "test": test,
    }
    assert not set(parts["train"]["group_id"]) & set(parts["val"]["group_id"]), "grupo repartido"

    PREPARED.mkdir(parents=True, exist_ok=True)
    cols = ["row_id", "group_id"] + feats + [ATTACK_CAT, TARGET]
    for name, df in parts.items():
        df[cols].reset_index(drop=True).to_parquet(PREPARED / f"{name}.parquet", index=False)
    # Asignación de todas las filas originales, incluidas las excluidas por contradicción.
    assignment = pd.concat(
        [pd.DataFrame({"row_id": d["row_id"], "split": n}) for n, d in parts.items()]
        + [pd.DataFrame({"row_id": excluded["row_id"], "split": "excluded_contradictory"})],
        ignore_index=True,
    )
    assignment.to_csv(PREPARED / "split_assignment.csv", index=False)
    print({n: len(d) for n, d in parts.items()})


if __name__ == "__main__":
    main()
