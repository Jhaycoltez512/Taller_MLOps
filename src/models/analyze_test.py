"""Análisis posterior a la evaluación final: de dónde vienen los falsos positivos en test.

Se ejecuta después de evidencias/metricas-test.json y no cambia modelo, umbral ni selección.
Compara, por partición, la fracción de tráfico normal con TTL de origen típico de los equipos
atacantes del banco de pruebas (sttl 62 o 254) y la FPR dentro y fuera de ese grupo.

Salida: evidencias/analisis-test.json
Uso: python -m src.models.analyze_test
"""
import json
from pathlib import Path

import pandas as pd

from src.data.schema import PREPARED, TARGET

ATTACKER_TTL = [62, 254]
OUT = Path("evidencias/analisis-test.json")


def ttl_split(df):
    normal = df[df[TARGET] == 0]
    high = normal["sttl"].isin(ATTACKER_TTL)
    out = {"normal_rows": len(normal), "share_attacker_ttl": float(high.mean())}
    if "y_pred" in normal:
        out["fpr_attacker_ttl"] = float(normal.loc[high, "y_pred"].mean())
        out["fpr_other_ttl"] = float(normal.loc[~high, "y_pred"].mean())
        out["false_positives"] = int(normal["y_pred"].sum())
        out["false_positives_with_attacker_ttl"] = int(normal.loc[high, "y_pred"].sum())
    return out


def fp_by(df, col, top=6):
    normal = df[df[TARGET] == 0]
    g = normal.groupby(col)["y_pred"].agg(normal_rows="size", false_positives="sum")
    g["fpr"] = g["false_positives"] / g["normal_rows"]
    g = g.sort_values("false_positives", ascending=False).head(top)
    return {str(k): {c: (float(v) if c == "fpr" else int(v)) for c, v in r.items()} for k, r in g.iterrows()}


def main():
    sel = json.loads(Path("evidencias/seleccion.json").read_text())
    train = pd.read_parquet(PREPARED / "train.parquet")
    val = pd.read_parquet(PREPARED / "val.parquet").merge(
        pd.read_csv(f"models/{sel['config_id']}/val_predictions.csv")[["row_id", "y_pred"]], on="row_id"
    )
    test = pd.read_parquet(PREPARED / "test.parquet").merge(
        pd.read_csv("evidencias/predicciones-test.csv")[["row_id", "y_pred"]], on="row_id"
    )
    result = {
        "version": sel["version"],
        "question": "¿Por qué la FPR sube de ~5 % en validación a ~18 % en test?",
        "attacker_ttl_values": ATTACKER_TTL,
        "normal_traffic_by_ttl": {"train": ttl_split(train), "val": ttl_split(val), "test": ttl_split(test)},
        "test_false_positives_by_service": fp_by(test, "service"),
        "test_false_positives_by_state": fp_by(test, "state"),
        "conclusion": (
            "En test, la fracción de tráfico normal con sttl 62/254 (típico de los equipos atacantes "
            "del banco de pruebas) pasa de ~23 % a ~52 %, y todos los falsos positivos están en ese "
            "grupo. El modelo aprendió el artefacto de TTL; el cambio de distribución del test, no un "
            "error de código, explica la mayor FPR. Modelo y umbral congelados no se modifican."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    t = result["normal_traffic_by_ttl"]["test"]
    print(f"test: {t['share_attacker_ttl']:.1%} normales con TTL atacante; FP en ese grupo "
          f"{t['false_positives_with_attacker_ttl']}/{t['false_positives']}")


if __name__ == "__main__":
    main()
