"""Data loading and splitting.

The California Housing dataset (1990 census, 20,640 block groups) ships with
scikit-learn, so there is nothing to download. The target ``MedHouseVal`` is the
median house value in a block group, in units of $100,000.

A quirk that matters a lot later: the target is **capped at 5.0** ($500k). About
5% of rows sit exactly at that ceiling, which no model can predict correctly
because the true value was censored before we ever saw it. We surface that in
the error analysis rather than pretending the model is simply "wrong" there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
TARGET_CAP = 5.0  # the censoring ceiling on the target, in $100k units
RANDOM_STATE = 42  # single source of truth for reproducibility


def load_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` as a feature DataFrame and target Series."""
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    return X, y


def train_test_split_housing(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproducible train/test split.

    We stratify on a binned version of the target so the (heavy) right tail and
    the capped rows are represented in both splits. Without this, a random split
    can leave the test set with a different price distribution than the training
    set, making the evaluation misleading.
    """
    # Bins on the target distribution for stratification. quantile-based bins
    # keep group sizes roughly equal; capped rows fall in the top bin.
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_bins,
    )
    return X_train, X_test, y_train, y_test


def is_capped(y: pd.Series | np.ndarray) -> np.ndarray:
    """Boolean mask of rows sitting at the target's censoring ceiling."""
    return np.asarray(y) >= TARGET_CAP
