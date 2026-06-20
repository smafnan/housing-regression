"""Model definitions and cross-validation.

Three models, in deliberate order of ambition:

    baseline : DummyRegressor predicting the mean. This is the bar every real
               model must clear. If a model can't beat "always guess the
               average", it has learned nothing.
    linear   : engineered features -> scaling -> LinearRegression. Interpretable,
               fast, and a strong sanity check.
    gbr      : engineered features -> HistGradientBoostingRegressor. Captures
               nonlinearities and interactions the linear model can't.

All three are full ``Pipeline`` objects so that feature engineering (and scaling)
happen *inside* cross-validation — fit only on each training fold — which is what
keeps the CV estimate honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import RANDOM_STATE
from .features import make_feature_transformer


def build_models() -> dict[str, Pipeline]:
    """Construct the three candidate pipelines, keyed by name."""
    baseline = Pipeline([
        ("model", DummyRegressor(strategy="mean")),
    ])

    linear = Pipeline([
        ("features", make_feature_transformer()),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])

    gbr = Pipeline([
        ("features", make_feature_transformer()),
        # Tree ensembles don't need scaling; skip it for speed and clarity.
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=400,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])

    return {"baseline": baseline, "linear": linear, "gbr": gbr}


def cross_validate_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """K-fold CV for every model; returns a tidy DataFrame of RMSE per model.

    We report the mean and standard deviation of RMSE across folds. The std is
    not decoration: a model that is better on average but wildly variable across
    folds may be the riskier choice in production.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        # scikit-learn maximises score, so RMSE is exposed as a negative number.
        neg_rmse = cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error"
        )
        rmse = -neg_rmse
        rows.append({
            "model": name,
            "cv_rmse_mean": rmse.mean(),
            "cv_rmse_std": rmse.std(),
        })
    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
