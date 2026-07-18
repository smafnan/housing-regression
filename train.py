"""Run the full classical-ML regression loop end to end.

    python train.py [--output-dir reports] [--no-save-model]

Steps, in order:
    1. Load the California Housing data.
    2. Reproducible train/test split (stratified on the price distribution).
    3. Cross-validate three models: baseline, linear, gradient boosting.
    4. Refit the best model on the full training set.
    5. Evaluate on the held-out test set.
    6. Run the error analysis and write figures + a metrics report.

The point of the script is not just to print a number; it is to show *why* the
chosen model wins and *where* it still fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from housing import (
    build_models,
    cross_validate_models,
    error_analysis,
    load_housing,
    regression_metrics,
    train_test_split_housing,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate housing regressors.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip writing the fitted model to disk.")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1) Loading data ...")
    X, y = load_housing()
    X_train, X_test, y_train, y_test = train_test_split_housing(X, y)
    print(f"   train={len(X_train)}  test={len(X_test)}  features={X.shape[1]}")

    print("\n2) Cross-validating candidate models (5-fold) ...")
    models = build_models()
    cv = cross_validate_models(models, X_train, y_train)
    print(cv.to_string(index=False))

    # The best model is the one with the lowest mean CV RMSE (excluding the
    # baseline, which exists only as a reference floor).
    ranked = cv[cv["model"] != "baseline"].sort_values("cv_rmse_mean")
    best_name = ranked.iloc[0]["model"]
    baseline_rmse = cv.loc[cv["model"] == "baseline", "cv_rmse_mean"].iloc[0]
    best_rmse = ranked.iloc[0]["cv_rmse_mean"]
    print(f"\n   Best model: '{best_name}'  "
          f"(CV RMSE {best_rmse:.3f} vs baseline {baseline_rmse:.3f}, "
          f"{100 * (1 - best_rmse / baseline_rmse):.0f}% better)")

    print(f"\n3) Refitting '{best_name}' on full training set ...")
    best = models[best_name]
    best.fit(X_train, y_train)

    print("\n4) Evaluating on held-out test set ...")
    y_pred = best.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)
    print("   " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in test_metrics.items()))

    print("\n5) Error analysis ...")
    findings = error_analysis(X_test, y_test, y_pred, reports_dir=args.output_dir)
    capped = findings["capped"]
    print(f"   Capped homes are {capped['fraction_of_rows']:.1%} of rows but "
          f"account for {capped['share_of_total_abs_error']:.1%} of total abs error.")
    print(f"   MAE on capped rows: {capped['mae_capped']:.3f}  vs  "
          f"uncapped: {capped['mae_uncapped']:.3f}")
    print(f"   Mean residual on capped rows: {capped['mean_residual_capped']:+.3f} "
          f"(negative = systematic under-prediction).")

    print("\n   10 worst predictions written to reports/worst_predictions.csv")
    findings["worst_predictions"].to_csv(args.output_dir / "worst_predictions.csv")

    # Persist a machine-readable metrics report (everything except the DataFrame).
    report = {
        "cv": cv.to_dict(orient="records"),
        "best_model": best_name,
        "test_metrics": test_metrics,
        "capped_analysis": findings["capped"],
        "mae_by_price_decile": findings["mae_by_price_decile"],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"   Metrics + figures written to {args.output_dir}/")

    if not args.no_save_model:
        joblib.dump(best, args.output_dir / "model.joblib")
        print(f"   Fitted model saved to {args.output_dir}/model.joblib")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
