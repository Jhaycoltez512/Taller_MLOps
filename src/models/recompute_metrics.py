"""Recalcula las métricas de test desde predicciones-test.csv con scikit-learn, de forma
independiente de src/models/metrics.py, y las compara con metricas-test.json.

Uso: python -m src.models.recompute_metrics
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PREDICTIONS = Path("evidencias/predicciones-test.csv")
METRICS = Path("evidencias/metricas-test.json")
TOL = 1e-12


def main():
    pred = pd.read_csv(PREDICTIONS)
    saved = json.loads(METRICS.read_text())
    th = saved["threshold"]
    y, score, y_pred = pred["y_true"], pred["score_attack"], pred["y_pred"]

    checks = {
        "rows_match": len(pred) == saved["dataset"]["rows_evaluated"],
        "row_ids_unique": pred["row_id"].is_unique,
        "y_pred_follows_threshold": bool((y_pred == (score >= th).astype(int)).all()),
    }
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    recomputed = {
        "recall_attack": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "precision_attack": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
    }
    checks["confusion_matrix_match"] = [[int(tn), int(fp)], [int(fn), int(tp)]] == saved["confusion_matrix"][
        "matrix_rows_true_cols_pred"
    ]
    diffs = {k: abs(v - saved["metrics"][k]) for k, v in recomputed.items()}
    checks["metrics_match"] = all(d <= TOL for d in diffs.values())

    for k, v in recomputed.items():
        print(f"{k:18s} recalculado {v:.6f} | guardado {saved['metrics'][k]:.6f} | dif {diffs[k]:.1e}")
    print(json.dumps(checks))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
