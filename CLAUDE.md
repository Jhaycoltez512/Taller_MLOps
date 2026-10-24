# Taller MLOps — UNSW-NB15

## Autoría de commits

Todos los commits de este repo deben tener como autor y committer a
`Mateo Hoyos <132465005+Jhaycoltez512@users.noreply.github.com>`, también en sesiones
de Claude Code en la nube o en la app. El hook `SessionStart` (`.claude/hooks/git-identity.sh`)
lo configura; antes de hacer commit, verifica con `git var GIT_AUTHOR_IDENT` y no uses
`--author` con otra identidad.

## Proyecto

Taller evaluable "Defiende el servidor" (MLOps 2026-II): clasificación binaria normal/ataque
con los splits oficiales de UNSW-NB15, versionando código (Git), datos (DVC + DagsHub),
experimentos (MLflow) y modelos (Hugging Face Hub).

- Entorno: `.venv` (Python 3.13); dependencias exactas en `requirements.lock.txt`. DVC solo se
  instala dentro del entorno virtual.
- Datos: `data/original.dvc` (los dos CSV oficiales) y `data/prepared` (pipeline `dvc.yaml`);
  remote DVC en DagsHub, credenciales solo en `.dvc/config.local`. Nunca se suben datos a git.
- Train/val salen solo del archivo de entrenamiento (`StratifiedGroupKFold`, grupos por vector
  de entradas); el test se reserva para una única evaluación de la versión congelada.
- Experimentos: `python -m src.models.train --config configs/experiments/<id>.yaml` con el árbol
  limpio (el run registra el commit); MLflow local en `mlflow.db`, exportado a `evidencias/`.
- Modelos: `yeico512/unsw-nb15-defiende-el-servidor`; v1 y v2 se identifican por SHA completo.
  No reescribir commits citados en MLflow, Hugging Face o `evidencias/` (p. ej. `fecc329`).
- Tokens (`DAGSHUB_TOKEN`, `HF_TOKEN`) solo como variables de entorno; nunca en archivos.
