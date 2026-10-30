"""Train and evaluate the FADE XGBoost Regressor (XGBR).

Place this file in the FADE repository root and run:
    python FADE_XGBR-Code_Test_Set.py

By default, the script reads the literature-derived CSV stored in the same
folder and writes metrics and predictions to the XGBR_results folder.

Optional arguments:
    python FADE_XGBR-Code_Test_Set.py --data "path/to/dataset.csv"
    python FADE_XGBR-Code_Test_Set.py --output "path/to/results"

Dependencies: numpy, pandas, scikit-learn, xgboost
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


REPOSITORY_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = (
    REPOSITORY_DIR
    / "Experimetnal data from Litrature for Machine Learning Training.csv"
)
DEFAULT_OUTPUT = REPOSITORY_DIR / "XGBR_results"

TARGET = "kobs"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def calculate_metrics(observed, predicted):
    """Calculate the seven performance metrics used by FADE.

    R2:   coefficient of determination
    d:    Willmott's index of agreement
    KGE:  Kling-Gupta efficiency (2009)
    E1:   Legates-McCabe efficiency
    MAE:  mean absolute error
    RMSE: root mean squared error
    MBE:  mean bias error; positive values indicate overprediction
    """
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if observed.size != predicted.size or observed.size < 2:
        raise ValueError("Metrics require at least two matched observations.")

    if not (
        np.isfinite(observed).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Observed and predicted values must be finite.")

    error = predicted - observed
    observed_mean = np.mean(observed)
    predicted_mean = np.mean(predicted)

    # Coefficient of determination (R2).
    r2_denominator = np.sum((observed - observed_mean) ** 2)
    r2 = (
        1.0 - np.sum(error ** 2) / r2_denominator
        if r2_denominator > 0
        else np.nan
    )

    # Willmott's index of agreement (d).
    d_denominator = np.sum(
        (
            np.abs(predicted - observed_mean)
            + np.abs(observed - observed_mean)
        ) ** 2
    )
    d = (
        1.0 - np.sum(error ** 2) / d_denominator
        if d_denominator > 0
        else np.nan
    )

    # Kling-Gupta efficiency (KGE, 2009).
    observed_sd = np.std(observed, ddof=1)
    predicted_sd = np.std(predicted, ddof=1)

    if observed_sd > 0 and predicted_sd > 0 and observed_mean != 0:
        correlation = np.corrcoef(observed, predicted)[0, 1]
        alpha = predicted_sd / observed_sd
        beta = predicted_mean / observed_mean
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    else:
        kge = np.nan

    # Legates-McCabe efficiency (E1).
    e1_denominator = np.sum(np.abs(observed - observed_mean))
    e1 = (
        1.0 - np.sum(np.abs(error)) / e1_denominator
        if e1_denominator > 0
        else np.nan
    )

    return {
        "R2": r2,
        "d": d,
        "KGE": kge,
        "E1": e1,
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MBE": np.mean(error),
    }


def load_dataset(file_path):
    """Load and validate a numeric CSV containing the kobs target column."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{file_path}\n"
            "Keep the dataset in the repository root or specify it with --data."
        )

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found.\n"
            f"Available columns: {list(data.columns)}"
        )

    data = data.apply(pd.to_numeric, errors="raise")

    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("The CSV contains missing or infinite values.")

    if len(data) < 5:
        raise ValueError("At least five observations are required.")

    return data


def run_xgbr(data, output_dir):
    """Train XGBR and save Training, Test, and Combined results."""
    X = data.drop(columns=TARGET)
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Apply the same target-scaling procedure used in the original workflow.
    target_scale = float(y_train.std(ddof=0))
    if target_scale == 0:
        raise ValueError("The training target is constant.")

    model.fit(X_train, y_train / target_scale)

    train_pred = model.predict(X_train) * target_scale
    test_pred = model.predict(X_test) * target_scale
    combined_pred = model.predict(X) * target_scale

    results = pd.DataFrame(
        [
            {
                "Dataset": "Training",
                "N": len(y_train),
                **calculate_metrics(y_train, train_pred),
            },
            {
                "Dataset": "Test",
                "N": len(y_test),
                **calculate_metrics(y_test, test_pred),
            },
            {
                "Dataset": "Combined",
                "N": len(y),
                **calculate_metrics(y, combined_pred),
            },
        ]
    ).set_index("Dataset")

    results = results[
        ["N", "R2", "d", "KGE", "E1", "MAE", "RMSE", "MBE"]
    ]

    print("\nXGBR evaluation results:")
    print(results.to_string(float_format=lambda value: f"{value:.10g}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "XGBR_metrics.csv"
    results.to_csv(metrics_file, index=True)

    predictions = pd.DataFrame(
        {
            "CSV_row": np.arange(len(data)) + 2,
            "Dataset": np.where(
                data.index.isin(X_train.index),
                "Training",
                "Test",
            ),
            "Observed_kobs": y.to_numpy(),
            "Predicted_kobs": combined_pred,
        }
    )
    predictions["Error_predicted_minus_observed"] = (
        predictions["Predicted_kobs"] - predictions["Observed_kobs"]
    )

    predictions.to_csv(
        output_dir / "XGBR_predictions_combined.csv",
        index=False,
    )

    for subset in ["Training", "Test"]:
        subset_predictions = predictions.loc[
            predictions["Dataset"] == subset
        ]
        subset_predictions.to_csv(
            output_dir / f"XGBR_predictions_{subset.lower()}.csv",
            index=False,
        )

    print("\nAll XGBR results were saved successfully.")
    print(f"Results folder: {output_dir}")
    print(f"Metrics file: {metrics_file}")


def parse_arguments():
    """Read optional dataset and output-folder arguments."""
    parser = ArgumentParser(
        description="Train and evaluate the FADE XGBoost Regressor."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the input CSV (default: repository literature dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Results folder (default: repository XGBR_results folder).",
    )
    return parser.parse_args()


def main():
    """Run the complete XGBR workflow."""
    args = parse_arguments()
    data = load_dataset(args.data.expanduser().resolve())
    run_xgbr(data, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
