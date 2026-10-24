"""Figuras del informe, generadas solo a partir de evidencias/.

Uso: python informe/figuras.py   ->  informe/figuras/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("informe/figuras")
V1, V2 = "exp01_logreg_base", "exp04_hist_gradient_boosting"
BLUE, RED, GREY = "#3f66a6", "#b8444a", "#8a94a0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def categories():
    t = pd.read_csv("evidencias/errores_por_categoria.csv", index_col=0)
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = range(len(t))
    ax.barh([i + 0.2 for i in y], t[V1] * 100, height=0.38, color=GREY, label="v1 · regresión logística")
    ax.barh([i - 0.2 for i in y], t[V2] * 100, height=0.38, color=BLUE, label="v2 · HistGradientBoosting")
    ax.set_yticks(list(y), [f"{c} (n={int(n):,})".replace(",", ".") for c, n in zip(t.index, t["n_val"])])
    ax.set_xlabel("% de filas de validación marcadas como ataque (Normal = FPR)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "categorias.png", dpi=200)


def ttl_shift():
    a = json.loads(Path("evidencias/analisis-test.json").read_text())["normal_traffic_by_ttl"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.3))
    parts = ["train", "val", "test"]
    ax1.bar(parts, [a[p]["share_attacker_ttl"] * 100 for p in parts], color=[GREY, GREY, RED])
    ax1.set_title("Tráfico normal con sttl 62/254 (%)")
    for i, p in enumerate(parts):
        ax1.text(i, a[p]["share_attacker_ttl"] * 100 + 1, f"{a[p]['share_attacker_ttl'] * 100:.1f}", ha="center")
    ax1.set_ylim(0, 65)
    labels = ["val\nsttl 62/254", "val\nresto", "test\nsttl 62/254", "test\nresto"]
    vals = [a["val"]["fpr_attacker_ttl"], a["val"]["fpr_other_ttl"], a["test"]["fpr_attacker_ttl"], a["test"]["fpr_other_ttl"]]
    ax2.bar(labels, [v * 100 for v in vals], color=[GREY, GREY, RED, RED])
    ax2.set_title("FPR de v2 por grupo (%)")
    for i, v in enumerate(vals):
        ax2.text(i, v * 100 + 1, f"{v * 100:.1f}", ha="center")
    ax2.set_ylim(0, 45)
    fig.tight_layout()
    fig.savefig(OUT / "ttl.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    categories()
    ttl_shift()
    print(f"figuras -> {OUT}")
