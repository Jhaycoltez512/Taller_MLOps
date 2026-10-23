"""Métricas binarias con ataque (label = 1) como clase positiva.

Regla de decisión: ataque si score_attack >= umbral.

Denominadores cero: si una razón tiene denominador 0 (p. ej. precision sin predicciones
positivas), la métrica vale 0.0 y se marca en `undefined` con su nombre. F1 de una clase vale
0.0 si su precision + recall es 0. Así ninguna métrica es NaN y el caso queda registrado.
"""
import numpy as np


def _ratio(num, den, name, undefined):
    if den == 0:
        undefined.append(name)
        return 0.0
    return float(num / den)


def _f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def confusion(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def binary_metrics(y_true, score, threshold):
    tp, fp, tn, fn = confusion(y_true, score, threshold)
    undefined = []
    recall = _ratio(tp, tp + fn, "recall_attack", undefined)
    precision = _ratio(tp, tp + fp, "precision_attack", undefined)
    fpr = _ratio(fp, fp + tn, "fpr", undefined)
    # Clase normal (0) para el macro-F1.
    recall_n = _ratio(tn, tn + fp, "recall_normal", undefined)
    precision_n = _ratio(tn, tn + fn, "precision_normal", undefined)
    f1_attack, f1_normal = _f1(precision, recall), _f1(precision_n, recall_n)
    return {
        "threshold": float(threshold),
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": fpr,
        "f1_attack": f1_attack,
        "f1_normal": f1_normal,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": tp + fp + tn + fn,
        "undefined": undefined,
    }


def choose_threshold(y_true, score, rule, max_fpr, grid_step):
    """Máximo recall con FPR <= max_fpr sobre una rejilla; empate -> umbral más alto (menos alertas)."""
    if rule != "max_recall_at_fpr":
        raise ValueError(f"regla desconocida: {rule}")
    grid = np.round(np.arange(grid_step, 1.0, grid_step), 6)
    rows = [binary_metrics(y_true, score, t) for t in grid]
    feasible = [r for r in rows if r["fpr"] <= max_fpr]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall_attack"], r["threshold"]))
        return best["threshold"], rows, True
    best = min(rows, key=lambda r: (r["fpr"], -r["threshold"]))
    return best["threshold"], rows, False
