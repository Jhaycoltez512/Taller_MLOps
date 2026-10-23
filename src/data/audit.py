"""Auditoría de los datos originales y de las particiones -> evidencias/auditoria.json.

Registra, para cada archivo oficial: fuente, fecha de descarga, SHA-256 (y su coincidencia
con el anexo), filas, columnas, tipos, clases, faltantes, duplicados por entradas y grupos
contradictorios. Para las particiones: política de limpieza, variables usadas/excluidas,
semilla, tamaños, huellas y cruces de entradas entre particiones.

Del test solo se auditan esquema, clases, duplicados y cruces; sus etiquetas no se usan
para ninguna decisión.

Uso: python -m src.data.audit
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.schema import (
    ATTACK_CAT,
    BINARY_CLIP,
    CATEGORICAL,
    EXCLUDED,
    EXPECTED_SHA256,
    FILES,
    ID,
    ORIGINAL,
    PREPARED,
    TARGET,
    feature_columns,
    group_ids,
    read_original,
    sha256_file,
    sha256_lines,
)

OUT = Path("evidencias/auditoria.json")


def audit_file(name, manifest):
    fname = FILES[name]
    path = ORIGINAL / fname
    df = read_original(name)
    feats = feature_columns(df)
    gid = group_ids(df, feats)
    groups = df.assign(group_id=gid).groupby("group_id")[TARGET].agg(["nunique", "size"])
    contradictory = groups[groups["nunique"] > 1]
    sha = sha256_file(path)
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "sha256_expected_annex": EXPECTED_SHA256[fname],
        "sha256_matches_annex": sha == EXPECTED_SHA256[fname],
        "sha256_matches_manifest": sha == manifest["files"][fname]["sha256"],
        "rows": len(df),
        "columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "id_is_row_number": bool((df[ID].to_numpy() == range(1, len(df) + 1)).all()),
        "label_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "attack_rate": round(float(df[TARGET].mean()), 6),
        "attack_cat_counts": df[ATTACK_CAT].value_counts().to_dict(),
        "attack_cat_consistent_with_label": bool(((df[ATTACK_CAT] == "Normal") == (df[TARGET] == 0)).all()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "anomalies": {f"{c}_values_gt_1": int((df[c] > 1).sum()) for c in BINARY_CLIP},
        "duplicates_by_features": {
            "definition": "filas cuyo vector de las 42 entradas (sin id, attack_cat ni label) ya apareció antes",
            "duplicate_rows": int(gid.duplicated().sum()),
            "duplicate_rate": round(float(gid.duplicated().mean()), 6),
            "unique_input_vectors": int(len(groups)),
            "groups_with_more_than_one_row": int((groups["size"] > 1).sum()),
            "largest_group_size": int(groups["size"].max()),
        },
        "contradictory_groups": {
            "definition": "grupos de entradas idénticas con label 0 y 1 a la vez",
            "groups": int(len(contradictory)),
            "rows": int(contradictory["size"].sum()),
        },
    }


def main():
    params = yaml.safe_load(open("params.yaml"))["data"]
    manifest = json.loads((ORIGINAL / "MANIFEST.json").read_text())
    parts = {n: pd.read_parquet(PREPARED / f"{n}.parquet") for n in ("train", "val", "test")}
    assignment = pd.read_csv(PREPARED / "split_assignment.csv")
    feats = feature_columns(parts["train"])
    groups = {n: set(d["group_id"]) for n, d in parts.items()}

    def cross(a, b):
        shared = groups[a] & groups[b]
        return {
            "shared_input_vectors": len(shared),
            f"rows_in_{b}_with_vector_in_{a}": int(parts[b]["group_id"].isin(shared).sum()),
            f"share_of_{b}_rows": round(float(parts[b]["group_id"].isin(shared).mean()), 6),
        }

    unseen = {c: sorted(set(parts["test"][c]) - set(parts["train"][c])) for c in CATEGORICAL}
    audit = {
        "source": {
            "official_page": manifest["source"],
            "share_link": manifest["share_link"],
            "downloaded_at": manifest["downloaded_at"],
            "dvc_tracking_file": "data/original.dvc",
        },
        "files": {n: audit_file(n, manifest) for n in FILES},
        "cleaning_policy": {
            "stateless_rules_applied_to_all_splits": [
                f"categóricas {CATEGORICAL}: espacios recortados",
                f"{BINARY_CLIP}: acotada a {{0, 1}} (variable declarada binaria)",
            ],
            "duplicates": "se conservan y se agrupan por vector de entradas (group_id = SHA-1 de las 42 entradas "
            "ya limpias); "
            "ningún grupo se reparte entre train y val",
            "contradictory_groups": "se excluyen de train/val (etiqueta ambigua); el test se conserva completo"
            if params["drop_contradictory"]
            else "se conservan",
            "fitted_transformations": "codificación, imputación, escalado y selección de variables se ajustan "
            "dentro del Pipeline del modelo y solo con train",
            "test_policy": "archivo de test completo, sin eliminar filas; solo se usa en la evaluación final "
            "de la versión congelada",
        },
        "features": {
            "used": feats,
            "n_used": len(feats),
            "categorical": CATEGORICAL,
            "numeric": [c for c in feats if c not in CATEGORICAL],
            "excluded": EXCLUDED,
            "test_categorical_values_unseen_in_train": unseen,
        },
        "splits": {
            "method": "StratifiedGroupKFold sobre el archivo de entrenamiento; un fold como validación",
            "seed": params["seed"],
            "n_folds": params["n_folds"],
            "val_fold": params["val_fold"],
            "stratify": params["stratify"],
            "group": "group_id",
            "assignment_file": str(PREPARED / "split_assignment.csv"),
            "assignment_counts": assignment["split"].value_counts().to_dict(),
            "parts": {
                n: {
                    "file": str(PREPARED / f"{n}.parquet"),
                    "rows": len(d),
                    "label_counts": {str(k): int(v) for k, v in d[TARGET].value_counts().sort_index().items()},
                    "attack_rate": round(float(d[TARGET].mean()), 6),
                    "attack_cat_counts": d[ATTACK_CAT].value_counts().to_dict(),
                    "unique_input_vectors": len(groups[n]),
                    "sha256_file": sha256_file(PREPARED / f"{n}.parquet"),
                    "sha256_row_ids": sha256_lines(d["row_id"]),
                }
                for n, d in parts.items()
            },
        },
        "crosses": {
            "train_val": cross("train", "val"),
            "train_test": cross("train", "test"),
            "val_test": cross("val", "test"),
            "note": "Los cruces con test se reportan para limitar las conclusiones; no se eliminan filas de test.",
        },
    }
    assert audit["crosses"]["train_val"]["shared_input_vectors"] == 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(f"auditoría -> {OUT}")


if __name__ == "__main__":
    main()
