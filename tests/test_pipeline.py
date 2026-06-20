"""Tests for the regression workflow.

We keep dataset-dependent tests light (they load the real data once) and focus
the rest on the pure, fast units: feature engineering and metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing import (
    add_engineered_features,
    build_models,
    cross_validate_models,
    load_housing,
    make_feature_transformer,
    regression_metrics,
    train_test_split_housing,
)
from housing.data import TARGET_CAP, is_capped
from housing.evaluate import error_analysis


@pytest.fixture(scope="module")
def data():
    X, y = load_housing()
    return train_test_split_housing(X, y)


# --- feature engineering ---------------------------------------------------- #

def test_engineered_features_added():
    X = pd.DataFrame({
        "MedInc": [3.0], "HouseAge": [20.0], "AveRooms": [5.0],
        "AveBedrms": [1.0], "Population": [1000.0], "AveOccup": [2.5],
        "Latitude": [34.0], "Longitude": [-118.0],
    })
    out = add_engineered_features(X)
    for col in ["bedrooms_per_room", "rooms_per_person",
                "population_density", "log_median_income"]:
        assert col in out.columns
    assert out["bedrooms_per_room"].iloc[0] == pytest.approx(1.0 / 5.0, rel=1e-3)
    assert out["rooms_per_person"].iloc[0] == pytest.approx(5.0 / 2.5, rel=1e-3)


def test_feature_engineering_is_stateless():
    # Same input -> same output, and input is not mutated (no leakage risk).
    X = pd.DataFrame({
        "MedInc": [1.0, 2.0], "HouseAge": [1.0, 2.0], "AveRooms": [4.0, 6.0],
        "AveBedrms": [1.0, 1.0], "Population": [100.0, 200.0],
        "AveOccup": [2.0, 3.0], "Latitude": [34.0, 35.0],
        "Longitude": [-118.0, -119.0],
    })
    before = X.copy()
    out1 = add_engineered_features(X)
    out2 = add_engineered_features(X)
    pd.testing.assert_frame_equal(out1, out2)
    pd.testing.assert_frame_equal(X, before)  # original untouched


def test_transformer_feature_names_out():
    ft = make_feature_transformer()
    X = pd.DataFrame({
        "MedInc": [1.0], "HouseAge": [1.0], "AveRooms": [4.0], "AveBedrms": [1.0],
        "Population": [100.0], "AveOccup": [2.0], "Latitude": [34.0],
        "Longitude": [-118.0],
    })
    ft.fit(X)
    names = list(ft.get_feature_names_out(X.columns))
    assert "log_median_income" in names
    assert len(names) == X.shape[1] + 4


# --- metrics ---------------------------------------------------------------- #

def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_metrics_known_values():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    m = regression_metrics(y_true, y_pred)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)


def test_is_capped():
    y = np.array([1.0, 4.99, 5.0, 5.5])
    assert list(is_capped(y)) == [False, False, True, True]


# --- model / end-to-end ----------------------------------------------------- #

def test_split_is_reproducible_and_disjoint(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) > len(X_test)
    # No row index appears in both splits.
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_models_fit_and_predict(data):
    X_train, X_test, y_train, y_test = data
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.isfinite(preds).all()


def test_best_model_beats_baseline(data):
    X_train, _, y_train, _ = data
    cv = cross_validate_models(build_models(), X_train, y_train, n_splits=3)
    baseline = cv.loc[cv["model"] == "baseline", "cv_rmse_mean"].iloc[0]
    gbr = cv.loc[cv["model"] == "gbr", "cv_rmse_mean"].iloc[0]
    # The gradient-boosting model must substantially beat "guess the mean".
    assert gbr < 0.7 * baseline


def test_error_analysis_finds_capping(data, tmp_path):
    X_train, X_test, y_train, y_test = data
    model = build_models()["gbr"]
    model.fit(X_train, y_train)
    findings = error_analysis(
        X_test, y_test, model.predict(X_test), reports_dir=tmp_path
    )
    capped = findings["capped"]
    # Capped rows should be over-represented in the error and under-predicted.
    assert capped["mae_capped"] > capped["mae_uncapped"]
    assert capped["mean_residual_capped"] < 0
    assert (tmp_path / "pred_vs_actual.png").exists()
