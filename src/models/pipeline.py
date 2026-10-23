"""Construye la solución completa (preprocesamiento + estimador) a partir de una configuración.

Todo lo que se ajusta a los datos (codificadores, escalado) vive dentro del Pipeline, así que
se ajusta solo con los datos que recibe `fit` (train) y se guarda/carga como una sola pieza.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.schema import CATEGORICAL

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def numeric_transformer(kind):
    if kind == "standard":
        return StandardScaler()
    if kind == "log1p_standard":
        # Todas las numéricas son >= 0 (verificado en la auditoría), así que log1p está definido.
        return make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    if kind == "passthrough":
        return "passthrough"
    raise ValueError(f"numeric desconocido: {kind}")


def categorical_transformer(pre):
    if pre["categorical"] == "onehot":
        # Valores con menos de `min_frequency` apariciones en train se agrupan; los no vistos se ignoran.
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=pre["onehot_min_frequency"])
    if pre["categorical"] == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"categorical desconocido: {pre['categorical']}")


def build_pipeline(cfg, features, seed):
    pre = cfg["preprocessing"]
    numeric = [c for c in features if c not in CATEGORICAL]
    columns = ColumnTransformer(
        [
            ("cat", categorical_transformer(pre), CATEGORICAL),
            ("num", numeric_transformer(pre["numeric"]), numeric),
        ],
        verbose_feature_names_out=False,
    )
    params = dict(cfg["model"]["params"])
    if cfg["model"]["type"] == "hist_gradient_boosting":
        # Las categóricas ordinales son las primeras columnas de la salida del ColumnTransformer.
        params["categorical_features"] = list(range(len(CATEGORICAL)))
    model = MODELS[cfg["model"]["type"]](random_state=seed, **params)
    return Pipeline([("preprocess", columns), ("model", model)])
