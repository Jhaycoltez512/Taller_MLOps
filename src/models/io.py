"""Guardado y carga de la solución completa en formato skops (sin pickle arbitrario)."""
import skops.io as sio

from src.data.schema import sha256_file

# Tipos internos de scikit-learn que skops no confía por defecto y que estos pipelines
# necesitan. La carga falla si el archivo pide cualquier otro tipo.
TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
    "sklearn.utils.validation.check_array",
]


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    return sha256_file(path)


def load_model(path):
    untrusted = sio.get_untrusted_types(file=path)
    unexpected = sorted(set(untrusted) - set(TRUSTED_TYPES))
    if unexpected:
        raise ValueError(f"{path} pide tipos no permitidos: {unexpected}")
    return sio.load(path, trusted=untrusted)
