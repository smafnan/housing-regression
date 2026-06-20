"""Metrics, residual diagnostics, and error analysis.

This module answers the project's "Done when" question: *can you explain why the
model is wrong on the cases it gets wrong?* It does that by (a) computing honest
held-out metrics and (b) slicing the error to find the segments where the model
fails and attaching a reason to each.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend so this runs in CI / no-display envs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TARGET_CAP, is_capped


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard regression metrics as a plain dict.

    RMSE punishes large misses (good when big errors are costly), MAE is the
    typical error in dollars-of-$100k, and R^2 is the share of variance explained.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    reports_dir: str | Path = "reports",
) -> dict[str, object]:
    """Run the full error analysis and write figures to ``reports_dir``.

    Returns a dictionary of findings (also handy for tests and the README).
    """
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    y_true = y_test.to_numpy(dtype=float)
    residual = y_pred - y_true  # positive = over-prediction
    abs_err = np.abs(residual)

    findings: dict[str, object] = {}
    findings["overall"] = regression_metrics(y_true, y_pred)

    # ------------------------------------------------------------------ #
    # 1) The headline failure mode: the target is censored at the $500k cap.
    #    The model literally cannot predict the true (unknown) value there,
    #    so it under-predicts those rows. We quantify how much of the total
    #    error those ~5% of rows account for.
    # ------------------------------------------------------------------ #
    capped = is_capped(y_true)
    findings["capped"] = {
        "fraction_of_rows": float(capped.mean()),
        "mae_capped": float(abs_err[capped].mean()),
        "mae_uncapped": float(abs_err[~capped].mean()),
        # Mean residual on capped rows: strongly negative = systematic under-prediction.
        "mean_residual_capped": float(residual[capped].mean()),
        "share_of_total_abs_error": float(abs_err[capped].sum() / abs_err.sum()),
    }

    # ------------------------------------------------------------------ #
    # 2) Error across the price range (target deciles). Shows whether the
    #    model is uniformly good or fails at the extremes.
    # ------------------------------------------------------------------ #
    decile = pd.qcut(y_true, 10, labels=False, duplicates="drop")
    by_decile = (
        pd.DataFrame({"decile": decile, "abs_err": abs_err, "y": y_true})
        .groupby("decile")
        .agg(mae=("abs_err", "mean"), mean_price=("y", "mean"), n=("y", "size"))
    )
    findings["mae_by_price_decile"] = by_decile["mae"].round(3).to_dict()

    # ------------------------------------------------------------------ #
    # 3) The worst individual predictions, with their features, so the
    #    failures are concrete rather than abstract.
    # ------------------------------------------------------------------ #
    worst_idx = np.argsort(abs_err)[-10:][::-1]
    worst = X_test.iloc[worst_idx].copy()
    worst["actual"] = y_true[worst_idx]
    worst["predicted"] = y_pred[worst_idx]
    worst["abs_error"] = abs_err[worst_idx]
    findings["worst_predictions"] = worst.round(2)

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    _plot_pred_vs_actual(y_true, y_pred, capped, reports / "pred_vs_actual.png")
    _plot_residual_hist(residual, reports / "residual_hist.png")
    _plot_error_by_decile(by_decile, reports / "mae_by_price_decile.png")

    return findings


# --------------------------------------------------------------------------- #
# Plot helpers. Each saves a single, captioned figure.
# --------------------------------------------------------------------------- #


def _plot_pred_vs_actual(y_true, y_pred, capped, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true[~capped], y_pred[~capped], s=6, alpha=0.25,
               label="uncapped")
    ax.scatter(y_true[capped], y_pred[capped], s=10, alpha=0.5, color="crimson",
               label="capped at $500k")
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="perfect prediction")
    ax.axvline(TARGET_CAP, color="grey", ls=":", lw=1)
    ax.set_xlabel("Actual median house value ($100k)")
    ax.set_ylabel("Predicted ($100k)")
    ax.set_title("Predicted vs. actual\n(capped homes cluster below the line)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_residual_hist(residual, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residual, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Residual (predicted - actual, $100k)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution (left tail = under-prediction)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_error_by_decile(by_decile: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_decile.index, by_decile["mae"], color="darkorange", alpha=0.85)
    ax.set_xlabel("Price decile (0 = cheapest, 9 = most expensive)")
    ax.set_ylabel("Mean absolute error ($100k)")
    ax.set_title("Error is concentrated at the expensive end")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
