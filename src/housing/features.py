"""Feature engineering.

The raw dataset already provides per-household averages (``AveRooms``,
``AveBedrms``, ``AveOccup``). The engineered features below encode *ratios* and
*nonlinearities* that a linear model in particular cannot discover on its own:

    bedrooms_per_room   - low values flag bigger, pricier homes
    rooms_per_person    - space per occupant; a wealth proxy
    population_density  - crowding within a block group
    log_median_income   - income's effect on price is strongly diminishing,
                          so the log is a better linear predictor

Everything is implemented as a single function wrapped in a
``FunctionTransformer`` so it composes cleanly inside an sklearn ``Pipeline``
and is applied identically to train, validation and test data — no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# A tiny constant to keep ratios finite if a denominator is ever zero.
_EPS = 1e-6


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``X`` with extra engineered columns appended.

    Pure and stateless: the output depends only on the input row, so applying it
    per-split cannot leak information between train and test.
    """
    X = X.copy()
    X["bedrooms_per_room"] = X["AveBedrms"] / (X["AveRooms"] + _EPS)
    X["rooms_per_person"] = X["AveRooms"] / (X["AveOccup"] + _EPS)
    X["population_density"] = X["Population"] / (X["AveOccup"] + _EPS)
    X["log_median_income"] = np.log1p(X["MedInc"])
    return X


# The four columns appended by add_engineered_features, in order.
_ENGINEERED_COLUMNS = [
    "bedrooms_per_room",
    "rooms_per_person",
    "population_density",
    "log_median_income",
]


def _feature_names_out(transformer, input_features):
    """``feature_names_out`` callback for the FunctionTransformer.

    Defined at module level (not as a closure) so the fitted pipeline remains
    picklable with joblib — a local/nested function cannot be serialised.
    """
    return np.asarray(list(input_features) + _ENGINEERED_COLUMNS, dtype=object)


def make_feature_transformer() -> FunctionTransformer:
    """An sklearn transformer that applies :func:`add_engineered_features`.

    ``feature_names_out`` is wired up so downstream tools (and feature-importance
    reporting) can recover human-readable names.
    """
    return FunctionTransformer(
        add_engineered_features,
        validate=False,
        feature_names_out=_feature_names_out,
    )
