"""Publica una versión (v1/v2) de la solución en Hugging Face Hub y prueba su recuperación.

Cada revisión contiene: model.skops (Pipeline completo), model_config.json (variables, umbral,
regla de decisión, referencias), val_metrics.json, requirements.txt con las versiones fijadas
para inferencia y README.md (model card).

Pasos:
  1. Verifica que el SHA-256 del modelo coincide con model_config.json y con el run de MLflow.
  2. Arma la revisión en models/hf/<version>/ y la sube como un commit; crea el tag <version>.
  3. Descarga la revisión por su SHA completo en una caché vacía, compara los SHA-256 de los
     archivos, carga el modelo y comprueba que reproduce las predicciones de validación.
  4. Escribe evidencias/versiones/<version>.json.

Requiere HF_TOKEN con permiso de escritura.
Uso: python -m src.models.publish_hf --version v1 --config-id exp01_logreg_base
     # solo verificar y registrar una revisión ya publicada:
     python -m src.models.publish_hf --version v1 --config-id exp01_logreg_base --revision <sha>
"""
import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
import yaml
from huggingface_hub import HfApi

from src.data.schema import PREPARED, sha256_file
from src.models.predict import download_solution, load_solution, predict
from src.models.train import TRACKING_URI

GITHUB = "https://github.com/Jhaycoltez512/Taller_MLOps"
DAGSHUB_DVC = "https://dagshub.com/Jhaycoltez512/Taller_MLOps.dvc"
INFERENCE_DEPS = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "skops", "huggingface_hub"]
VERSIONS = Path("evidencias/versiones")


def pinned_requirements():
    lock = {}
    for line in Path("requirements.lock.txt").read_text().splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            lock[name.lower().replace("_", "-")] = f"{name}=={ver}"
    return "\n".join(lock[d.lower().replace("_", "-")] for d in INFERENCE_DEPS) + "\n"


def pct(x):
    return f"{100 * x:.2f} %"


def metrics_table(val):
    rows = [("Umbral de operación", val["operating"]), ("Umbral 0,5 (referencia)", val["reference_0.5"])]
    lines = [
        "| Umbral | Recall ataque | Precision ataque | FPR | Macro-F1 | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name}: {m['threshold']:.3f} | {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | "
            f"{pct(m['fpr'])} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |"
        )
    return "\n".join(lines)


def category_table(config_id):
    t = pd.read_csv("evidencias/errores_por_categoria.csv", index_col=0)
    lines = ["| attack_cat | Filas en validación | Marcadas como ataque |", "|---|---|---|"]
    for cat, row in t.iterrows():
        lines.append(f"| {cat} | {int(row['n_val']):,} | {pct(row[config_id])} |")
    return "\n".join(lines)


def model_card(version, cfg, run_id, val, repo_id, changes, trusted):
    op = val["operating"]
    exp = cfg["experiment_config"]
    params = ", ".join(f"`{k}={v}`" for k, v in exp["model"]["params"].items())
    pre = exp["preprocessing"]
    return f"""---
library_name: sklearn
license: other
tags:
  - tabular-classification
  - intrusion-detection
  - unsw-nb15
  - skops
---

# Defiende el servidor · {version} · `{cfg['id']}`

Clasificador binario de conexiones de red (**0 = normal, 1 = ataque**) entrenado con los
splits oficiales de [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
Taller evaluable de MLOps (Universidad Externado de Colombia, 2026-II): el objetivo es
versionar y recuperar código, datos, experimentos y modelos, no maximizar una métrica.

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Versión | `{version}` (tag `{version}` en este repositorio de modelos y en el de código) |
| Código | [{GITHUB}]({GITHUB}), commit de entrenamiento `{cfg['git_commit']}` |
| Configuración | `{exp_path(cfg['id'])}` |
| Datos (DVC) | original `{cfg['data.original_dvc_md5']}` · preparados `{cfg['data.prepared_dvc_md5']}` · remote `{DAGSHUB_DVC}` |
| Particiones | train SHA-256 `{cfg['data.train_sha256']}` · val SHA-256 `{cfg['data.val_sha256']}` |
| Experimento (MLflow) | experimento `defiende-el-servidor`, run `{run_id}` |
| Modelo | `model.skops`, SHA-256 `{cfg['model_sha256']}` |

## Propósito y uso previsto

Apoyar a un equipo de seguridad a priorizar conexiones sospechosas en un entorno académico.
No es un sistema de detección listo para producción: ver Limitaciones.

## Datos

- Entrenamiento y validación salen solo de `UNSW_NB15_training-set.csv` (175.341 filas):
  train 139.520 y validación 34.881, con `StratifiedGroupKFold` (semilla {cfg['seed']}). Ningún
  grupo de entradas idénticas queda repartido entre ambos; los 229 grupos con etiquetas
  contradictorias (940 filas) se excluyen.
- `UNSW_NB15_testing-set.csv` (82.332 filas) se reserva para la evaluación final.
- Entradas: {len(cfg['features'])} variables (categóricas: {', '.join(cfg['categorical'])}). Excluidas:
  `id`, `attack_cat` (revela la respuesta) y `label`.

## Modelo

- Familia: `{cfg['family']}` · estimador `{exp['model']['type']}` con {params}.
- Preprocesamiento dentro del Pipeline, ajustado solo con train: categóricas `{pre['categorical']}`,
  numéricas `{pre['numeric']}`.
- Hipótesis del experimento: {exp['hypothesis']}

## Umbral

`threshold = {cfg['threshold']}` (en `model_config.json`). Regla de decisión: **ataque si
`score_attack >= threshold`**, con `score_attack = predict_proba[:, 1]`. Se eligió con predicciones
out-of-fold sobre train (5 folds agrupados), maximizando el recall de ataque con FPR ≤
{pct(cfg['threshold_rule']['max_fpr'])}. Validación y test no intervinieron.

## Métricas de validación

Ataque como clase positiva; {op['n']:,} filas de validación.

{metrics_table(val)}

Matriz de confusión con el umbral de operación: TP {op['tp']:,} · FP {op['fp']:,} · TN {op['tn']:,} · FN {op['fn']:,}.
Denominador cero: la métrica vale 0,0 y se marca en `undefined` (aquí: {op['undefined'] or 'ninguno'}).

Detección por categoría (para `Normal` es la tasa de falsos positivos):

{category_table(cfg['id'])}
{changes}
## Evaluación final en test

{test_section(version)}

## Limitaciones

- UNSW-NB15 es un dataset sintético de 2015 (banco de pruebas IXIA): no demuestra eficacia
  sobre una red actual.
- Los `Fuzzers` son la principal fuente de ataques no detectados.
- El umbral fija la FPR en ~5 % sobre la distribución de train; si cambia la proporción de
  ataques o el tráfico, la FPR y la precision cambian.
- El 8,1 % de las filas de test comparte vector de entradas con train: las métricas de test
  pueden ser algo optimistas para esas filas.

## Uso

```bash
pip install -r requirements.txt
```

```python
import json, pandas as pd, skops.io as sio
from huggingface_hub import snapshot_download

path = snapshot_download("{repo_id}", revision="{version}")  # o el SHA completo de la revisión
config = json.load(open(f"{{path}}/model_config.json"))
model_file = f"{{path}}/model.skops"
# Tipos de scikit-learn que este archivo necesita y que skops no confía por defecto:
trusted = {trusted!r}
assert set(sio.get_untrusted_types(file=model_file)) <= set(trusted)
model = sio.load(model_file, trusted=trusted)

df = pd.read_csv("conexiones.csv")  # columnas de config["features"]
score = model.predict_proba(df[config["features"]])[:, 1]
y_pred = (score >= config["threshold"]).astype(int)
```

Desde el repositorio de código: `python -m src.models.predict --repo-id {repo_id} --revision <SHA> --input <archivo> --output preds.csv`.

## Referencias

- Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
  detection systems. MilCIS 2015.
- Código, datos y experimentos: [{GITHUB}]({GITHUB}).
"""


def test_section(version):
    if version == "v1":
        return "No aplica: según el enunciado, v1 se compara con v2 solo en validación."
    path = Path("evidencias/metricas-test.json")
    if not path.exists():
        return "Pendiente: la versión elegida se evalúa una sola vez en test después de congelar la selección."
    t = json.loads(path.read_text())
    a = json.loads(Path("evidencias/analisis-test.json").read_text())["normal_traffic_by_ttl"]
    m, cm, v = t["metrics"], t["confusion_matrix"], t["validation_reference"]
    return f"""Evaluación única de la versión congelada, el {t['evaluated_at'][:10]}.

- Revisión evaluada: `{t['model']['hf_revision']}` (tag `{version}`), modelo SHA-256 `{t['model']['sha256']}`.
  Esta sección se añadió después en un commit que solo cambia este README: el modelo no cambia.
- Datos: `UNSW_NB15_testing-set.csv` completo, {t['dataset']['rows_evaluated']:,} filas, SHA-256 `{t['dataset']['sha256']}`.
- Umbral {t['threshold']} (congelado); ataque si `score_attack >= {t['threshold']}`.

| Métrica | Validación | Test |
|---|---|---|
| Recall ataque | {v['val_recall_attack']:.4f} | {m['recall_attack']:.4f} |
| Precision ataque | {v['val_precision_attack']:.4f} | {m['precision_attack']:.4f} |
| FPR | {pct(v['val_fpr'])} | {pct(m['fpr'])} |
| Macro-F1 | {v['val_macro_f1']:.4f} | {m['macro_f1']:.4f} |

Matriz de confusión en test: TP {cm['tp']:,} · FP {cm['fp']:,} · TN {cm['tn']:,} · FN {cm['fn']:,}.
ROC-AUC {t['threshold_free_metrics']['roc_auc']:.4f} · PR-AUC {t['threshold_free_metrics']['pr_auc']:.4f}.

**El resultado empeora en FPR y se conserva.** En test, el {pct(a['test']['share_attacker_ttl'])} del
tráfico normal tiene `sttl` 62 o 254, valores típicos de los equipos atacantes del banco de
pruebas (en train y validación: {pct(a['train']['share_attacker_ttl'])} y {pct(a['val']['share_attacker_ttl'])}).
Todos los falsos positivos de test están en ese grupo (FPR {pct(a['test']['fpr_attacker_ttl'])}; en el
resto, {pct(a['test']['fpr_other_ttl'])}). El modelo aprendió ese artefacto de TTL y el test tiene otra
distribución de tráfico normal. Predicciones por fila y métricas recalculables en el repositorio
de código (`evidencias/predicciones-test.csv`, `evidencias/metricas-test.json`)."""


def update_card(version, config_id, repo_id):
    """Sube solo el README.md regenerado; el tag sigue en la revisión evaluada."""
    record_path = VERSIONS / f"{version}.json"
    record = json.loads(record_path.read_text())
    src = Path("models") / config_id
    cfg = json.loads((src / "model_config.json").read_text())
    val = json.loads((src / "val_metrics.json").read_text())
    trusted = sio.get_untrusted_types(file=src / "model.skops")
    card = model_card(version, cfg, record["mlflow_run_id"], val, repo_id, changes_section(version, cfg, val), trusted)
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(card)
        readme_sha = sha256_file(readme)
        api = HfApi()
        commit = api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"{version}: model card con el resultado de test (modelo sin cambios)",
        )
    tree = {f.path: f for f in api.list_repo_tree(repo_id, revision=commit.oid)}
    assert tree["model.skops"].lfs.sha256 == record["model_sha256"], "el modelo cambió"
    assert api.model_info(repo_id, revision=version).sha == record["hf_revision"], "el tag se movió"
    record.setdefault("card_updates", []).append(
        {
            "hf_revision": commit.oid,
            "changed_files": ["README.md"],
            "readme_sha256": readme_sha,
            "model_sha256_at_revision": record["model_sha256"],
            "reason": "añade el resultado de la evaluación final en test",
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"{version}: model card actualizado en {commit.oid}; modelo sin cambios; tag en {record['hf_revision']}")


def exp_path(config_id):
    return f"configs/experiments/{config_id}.yaml"


def changes_section(version, cfg, val):
    if version == "v1":
        return ""
    v1 = json.loads((VERSIONS / "v1.json").read_text())
    a, b = v1["val_metrics_operating"], val["operating"]
    sel = json.loads(Path("evidencias/seleccion.json").read_text())
    return f"""
## Cambios respecto a v1

v1 = `{v1['config_id']}` (revisión `{v1['hf_revision']}`). Cambio: {sel['change_vs_v1']}

| Validación | v1 | {version} | Diferencia |
|---|---|---|---|
| Recall ataque | {a['recall_attack']:.4f} | {b['recall_attack']:.4f} | {100 * (b['recall_attack'] - a['recall_attack']):+.2f} pp |
| Precision ataque | {a['precision_attack']:.4f} | {b['precision_attack']:.4f} | {100 * (b['precision_attack'] - a['precision_attack']):+.2f} pp |
| FPR | {pct(a['fpr'])} | {pct(b['fpr'])} | {100 * (b['fpr'] - a['fpr']):+.2f} pp |
| Macro-F1 | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {100 * (b['macro_f1'] - a['macro_f1']):+.2f} pp |
| Falsos negativos | {a['fn']:,} | {b['fn']:,} | {b['fn'] - a['fn']:+,} |
| Falsos positivos | {a['fp']:,} | {b['fp']:,} | {b['fp'] - a['fp']:+,} |

Qué empeora y qué se acepta: {sel['tradeoffs']}
"""


def download_check(repo_id, revision, files, src):
    """Descarga la revisión por SHA en una caché vacía; compara archivos y predicciones de validación."""
    with tempfile.TemporaryDirectory() as cache:
        path = download_solution(repo_id, revision, cache_dir=cache)
        remote = {name: sha256_file(path / name) for name in files}
        assert files == remote, f"archivos descargados distintos: {files} vs {remote}"
        model, config = load_solution(path)
        sample = pd.read_parquet(PREPARED / "val.parquet").head(200)
        local = pd.read_csv(src / "val_predictions.csv").head(200)
        got = predict(model, config, sample)
    max_diff = float(np.abs(got["score_attack"].to_numpy() - local["score_attack"].to_numpy()).max())
    assert (got["row_id"].to_numpy() == local["row_id"].to_numpy()).all()
    assert (got["y_pred"].to_numpy() == local["y_pred"].to_numpy()).all() and max_diff < 1e-9
    return {
        "cache": "temporal vacía",
        "files_match": True,
        "rows_compared": len(got),
        "max_abs_score_diff": max_diff,
        "labels_match": True,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--revision", help="no subir: verificar y registrar esta revisión ya publicada")
    ap.add_argument("--skip-download-check", action="store_true", help="registrar sin probar la descarga")
    ap.add_argument("--update-card", action="store_true", help="solo actualizar el model card (README.md)")
    args = ap.parse_args()
    repo_id = yaml.safe_load(open("params.yaml"))["hub"]["repo_id"]
    if args.update_card:
        update_card(args.version, args.config_id, repo_id)
        return

    src = Path("models") / args.config_id
    cfg = json.loads((src / "model_config.json").read_text())
    val = json.loads((src / "val_metrics.json").read_text())
    run_id = json.loads((src / "run.json").read_text())["run_id"]

    # 1. El artefacto que se publica es el mismo que registró MLflow.
    model_sha = sha256_file(src / "model.skops")
    mlflow.set_tracking_uri(TRACKING_URI)
    logged = mlflow.MlflowClient().get_run(run_id).data.tags["model_sha256"]
    assert model_sha == cfg["model_sha256"] == logged, "el modelo no coincide con el run de MLflow"

    # 2. Revisión (contenido determinista: se puede regenerar para verificar).
    stage = Path("models/hf") / args.version
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(src / "model.skops", stage / "model.skops")
    cfg_out = {**cfg, "version": args.version, "mlflow_run_id": run_id, "mlflow_experiment": "defiende-el-servidor"}
    (stage / "model_config.json").write_text(json.dumps(cfg_out, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(src / "val_metrics.json", stage / "val_metrics.json")
    (stage / "requirements.txt").write_text(pinned_requirements())
    changes = changes_section(args.version, cfg, val)
    trusted = sio.get_untrusted_types(file=stage / "model.skops")
    (stage / "README.md").write_text(model_card(args.version, cfg, run_id, val, repo_id, changes, trusted))
    files = {f.name: sha256_file(f) for f in sorted(stage.iterdir())}

    api = HfApi()
    if args.revision:
        revision = args.revision
        assert api.model_info(repo_id, revision=args.version).sha == revision, "el tag no apunta a esa revisión"
    else:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=stage,
            commit_message=f"{args.version}: {args.config_id} (run {run_id}, umbral {cfg['threshold']})",
            delete_patterns=["*.skops", "*.json", "*.txt", "*.md"],
        )
        revision = commit.oid
        api.create_tag(repo_id, tag=args.version, revision=revision, tag_message=f"{args.version}: {args.config_id}")
    # El modelo de esa revisión en el Hub (hash LFS) es el mismo archivo local.
    tree = {f.path: f for f in api.list_repo_tree(repo_id, revision=revision)}
    assert tree["model.skops"].lfs.sha256 == model_sha, "el modelo en el Hub no coincide"
    commit_date = next(c.created_at for c in api.list_repo_commits(repo_id) if c.commit_id == revision)

    # 3. Recuperación por SHA en una caché vacía.
    check = None if args.skip_download_check else download_check(repo_id, revision, files, src)

    record = {
        "version": args.version,
        "config_id": args.config_id,
        "hf_repo_id": repo_id,
        "hf_url": f"https://huggingface.co/{repo_id}",
        "hf_revision": revision,
        "hf_tag": args.version,
        "hf_revision_url": f"https://huggingface.co/{repo_id}/tree/{revision}",
        "hf_commit_date": commit_date.isoformat(),
        "files_sha256": files,
        "model_sha256": model_sha,
        "hub_lfs_sha256_matches": True,
        "threshold": cfg["threshold"],
        "mlflow_run_id": run_id,
        "git_training_commit": cfg["git_commit"],
        "config_file": exp_path(args.config_id),
        "dvc_original_md5": cfg["data.original_dvc_md5"],
        "dvc_prepared_md5": cfg["data.prepared_dvc_md5"],
        "val_metrics_operating": val["operating"],
        "download_check": check or "pendiente: descarga por SHA no verificada todavía",
    }
    VERSIONS.mkdir(parents=True, exist_ok=True)
    (VERSIONS / f"{args.version}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.version} -> {record['hf_revision_url']} | descarga: {'ok' if check else 'pendiente'}")


if __name__ == "__main__":
    main()
