import argparse
from pathlib import Path

import joblib
import pandas as pd

REQUIRED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

DEFAULT_MODEL_PATH = Path("models/tuned_xgboost_model.pkl")
DEFAULT_SCALER_PATH = Path("data/processed/scaler.pkl")
DEFAULT_OUTPUT_PATH = Path("predictions/predictions.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["temp_diff"] = x["lub oil temp"] - x["Coolant temp"]
    x["stress_index"] = x["Engine rpm"] * x["Fuel pressure"]
    x["pressure_ratio"] = x["Lub oil pressure"] / (x["Coolant pressure"] + 1e-6)
    return x


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference using trained tuned XGBoost model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing raw sensor columns.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path for output CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to model .pkl. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--scaler",
        default=str(DEFAULT_SCALER_PATH),
        help=f"Path to scaler .pkl. Default: {DEFAULT_SCALER_PATH}",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    x_raw = df[REQUIRED_COLUMNS]
    x_feat = build_features(x_raw)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Keep feature order consistent with training
    if hasattr(scaler, "feature_names_in_"):
        x_feat = x_feat[list(scaler.feature_names_in_)]

    x_scaled = scaler.transform(x_feat)

    preds = model.predict(x_scaled)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_scaled)[:, 1]

    result = df.copy()
    result["prediction"] = preds
    if probs is not None:
        result["prediction_proba_class_1"] = probs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Batch inference complete.")
    print(f"Input file: {input_path.resolve()}")
    print(f"Output file: {output_path.resolve()}")
    print(f"Rows processed: {len(result)}")


if __name__ == "__main__":
    main()
