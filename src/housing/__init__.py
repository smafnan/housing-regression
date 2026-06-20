"""housing - a full classical-ML regression workflow on California housing.

The package is organised to mirror the canonical ML loop, one module per stage:

    data      -> load and split the data (reproducibly)
    features  -> feature engineering as an sklearn-compatible transformer
    model     -> baseline + candidate model pipelines, cross-validation
    evaluate  -> metrics, residual diagnostics, and error analysis

`train.py` (at the project root) wires these together into one runnable command.
"""

from .data import load_housing, train_test_split_housing
from .features import add_engineered_features, make_feature_transformer
from .model import build_models, cross_validate_models
from .evaluate import regression_metrics, error_analysis

__all__ = [
    "load_housing",
    "train_test_split_housing",
    "add_engineered_features",
    "make_feature_transformer",
    "build_models",
    "cross_validate_models",
    "regression_metrics",
    "error_analysis",
]
__version__ = "1.0.0"
