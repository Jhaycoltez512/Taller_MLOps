# Taller MLOps — UNSW-NB15

## Autoría de commits

Todos los commits de este repo deben tener como autor y committer a
`Mateo Hoyos <132465005+Jhaycoltez512@users.noreply.github.com>`, también en sesiones
de Claude Code en la nube o en la app. El hook `SessionStart` (`.claude/hooks/git-identity.sh`)
lo configura; antes de hacer commit, verifica con `git var GIT_AUTHOR_IDENT` y no uses
`--author` con otra identidad.

## Proyecto

- Entorno: `.venv` (Python 3.14); dependencias exactas en `requirements.lock.txt`.
- Datos y salidas del pipeline versionados con DVC (`dvc.yaml`, `params.yaml`, `dvc.lock`);
  nunca se suben a git.
- Split principal: temporal por sesión de captura (ver `reports/eda_raw/README.md`).
