import csv
import io
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "severity_model.joblib"
METADATA_PATH = MODEL_DIR / "severity_model.json"
MAX_DATASET_BYTES = 20 * 1024 * 1024
MAX_DATASET_ROWS = 100_000
MAX_DATASET_COLUMNS = 100
MAX_CLASSES = 10
MIN_CLASS_SAMPLES = 5
MODEL_LOCK = threading.RLock()


class DatasetError(ValueError):
    pass


def _validate_column_names(columns) -> list[str]:
    names = [str(column).strip() if column is not None else "" for column in columns]
    if any(not name for name in names):
        raise DatasetError("Every dataset column must have a non-empty name.")
    if len(set(names)) != len(names):
        raise DatasetError("Column names must be unique after trimming whitespace.")
    return names


def _read_dataset(contents: bytes, filename: str) -> pd.DataFrame:
    if not contents:
        raise DatasetError("The uploaded dataset is empty.")
    if len(contents) > MAX_DATASET_BYTES:
        raise DatasetError("Dataset files must be 20 MiB or smaller.")

    extension = Path(filename).suffix.lower()
    try:
        if extension == ".csv":
            try:
                reader = csv.reader(io.StringIO(contents.decode("utf-8-sig")), strict=True)
                raw_columns = next(reader)
            except (StopIteration, UnicodeDecodeError, csv.Error) as error:
                raise DatasetError("The CSV is empty or has an invalid header.") from error
            _validate_column_names(raw_columns)
            frame = pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig", low_memory=False)
        elif extension in (".xlsx", ".xlsm"):
            workbook = load_workbook(io.BytesIO(contents), read_only=True, data_only=True)
            try:
                raw_columns = next(workbook.active.iter_rows(min_row=1, max_row=1, values_only=True), ())
                _validate_column_names(raw_columns)
            finally:
                workbook.close()
            frame = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        else:
            raise DatasetError("Upload a CSV or Excel .xlsx file.")
    except DatasetError:
        raise
    except (BadZipFile, EOFError, InvalidFileException, OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError, ImportError) as error:
        raise DatasetError(f"Could not read the dataset: {error}") from error

    if frame.empty:
        raise DatasetError("The dataset has no data rows.")
    if len(frame) > MAX_DATASET_ROWS:
        raise DatasetError(f"Datasets may contain at most {MAX_DATASET_ROWS:,} rows.")
    if len(frame.columns) > MAX_DATASET_COLUMNS:
        raise DatasetError(f"Datasets may contain at most {MAX_DATASET_COLUMNS} columns.")

    frame.columns = _validate_column_names(frame.columns)
    return frame


def inspect_dataset(contents: bytes, filename: str) -> dict:
    frame = _read_dataset(contents, filename)
    columns = []
    for name in frame.columns:
        values = frame[name].dropna()
        columns.append({
            "name": name,
            "kind": "numeric" if pd.api.types.is_numeric_dtype(frame[name]) else "categorical",
            "non_empty": int(values.size),
            "unique": int(values.nunique()),
        })
    return {"rows": int(len(frame)), "columns": columns, "size_bytes": len(contents)}


def _clean_target(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def _load_metadata() -> dict | None:
    with MODEL_LOCK:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            return None
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def model_status() -> dict:
    metadata = _load_metadata()
    if metadata is None:
        return {"trained": False}
    return {"trained": True, **metadata}


def train_model(
    contents: bytes,
    filename: str,
    target: str,
    features: list[str],
    dataset_source: str,
    dataset_permission: str,
) -> dict:
    frame = _read_dataset(contents, filename)
    if not isinstance(dataset_source, str) or not dataset_source.strip():
        raise DatasetError("Provide the dataset publisher or source citation.")
    if not isinstance(dataset_permission, str) or not dataset_permission.strip():
        raise DatasetError("Provide the dataset license or documented permission basis.")
    if len(dataset_source.strip()) > 300 or len(dataset_permission.strip()) > 300:
        raise DatasetError("Dataset source and permission details must be 300 characters or fewer.")
    if not isinstance(target, str) or target not in frame.columns:
        raise DatasetError("Choose a target column that exists in the uploaded dataset.")
    if not isinstance(features, list) or not features:
        raise DatasetError("Select at least one feature column.")
    if len(features) != len(set(features)):
        raise DatasetError("Feature columns must not contain duplicates.")
    if target in features or any(feature not in frame.columns for feature in features):
        raise DatasetError("Feature columns must exist and must not include the target column.")

    selected = frame[[target, *features]].copy()
    selected[target] = _clean_target(selected[target])
    selected = selected.dropna(subset=[target])
    selected = selected[selected[target] != ""]
    if selected.empty:
        raise DatasetError("The selected target column has no labeled rows.")

    class_counts = selected[target].value_counts()
    if len(class_counts) < 2:
        raise DatasetError("At least two severity classes are required for classification.")
    if len(class_counts) > MAX_CLASSES:
        raise DatasetError(f"The target has more than {MAX_CLASSES} classes; merge or filter labels first.")
    low_count_classes = class_counts[class_counts < MIN_CLASS_SAMPLES]
    if not low_count_classes.empty:
        labels = ", ".join(f"{label} ({count})" for label, count in low_count_classes.items())
        raise DatasetError(f"Each severity class needs at least {MIN_CLASS_SAMPLES} labeled rows for a holdout evaluation. Too few: {labels}.")

    x_values = selected[features].copy()
    numeric_features = [name for name in features if pd.api.types.is_numeric_dtype(x_values[name])]
    categorical_features = [name for name in features if name not in numeric_features]
    for name in categorical_features:
        x_values[name] = x_values[name].astype("string").replace("", pd.NA)
    y_values = selected[target].astype(str)

    class_labels = sorted(class_counts.index.tolist())
    test_count = max(math.ceil(len(selected) * 0.25), len(class_labels))
    if len(selected) - test_count < len(class_labels):
        raise DatasetError("There are not enough labeled rows for a stratified train/test split.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=test_count,
        random_state=42,
        stratify=y_values,
    )
    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    pipeline = Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )),
    ])
    try:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"Could not train from the selected columns: {error}") from error

    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        zero_division=0,
        output_dict=True,
    )
    accuracy = float(accuracy_score(y_test, predictions))
    majority_baseline = float(y_test.value_counts(normalize=True).max())
    balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    warnings = [
        "Holdout metrics are an initial diagnostic, not evidence of safety or field readiness.",
        "Check for identifiers, post-crash information, duplicate incidents, and time/location leakage before trusting any result.",
        "Use an independent, time- or location-separated evaluation before a controlled pilot.",
    ]
    if accuracy <= majority_baseline:
        warnings.append(
            f"Holdout accuracy ({accuracy:.3f}) does not beat the majority-class baseline ({majority_baseline:.3f})."
        )
    if "fatal" in report:
        fatal_recall = float(report["fatal"]["recall"])
        warnings.append(
            f"Fatal-class recall is {fatal_recall:.3f}; the model missed {1 - fatal_recall:.1%} of fatal cases in this holdout."
        )
    if int(class_counts.min()) < 20:
        warnings.append("At least one class has fewer than 20 examples; its metrics are especially uncertain.")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_source": dataset_source.strip(),
        "dataset_permission": dataset_permission.strip(),
        "target": target,
        "features": features,
        "feature_schema": {
            name: "numeric" if name in numeric_features else "categorical"
            for name in features
        },
        "classes": class_labels,
        "rows_used": int(len(selected)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_counts": {str(label): int(count) for label, count in class_counts.items()},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "majority_baseline_accuracy": round(majority_baseline, 4),
            "balanced_accuracy": round(balanced_accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "class_report": {
            label: {
                key: round(float(value), 4) if isinstance(value, (float, np.floating)) else int(value)
                for key, value in report[label].items()
            }
            for label in class_labels
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=class_labels).tolist(),
        "warnings": warnings,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_temp = MODEL_PATH.with_suffix(".joblib.tmp")
    metadata_temp = METADATA_PATH.with_suffix(".json.tmp")
    with MODEL_LOCK:
        try:
            joblib.dump(pipeline, model_temp, compress=3)
            metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(model_temp, MODEL_PATH)
            os.replace(metadata_temp, METADATA_PATH)
        finally:
            model_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
    return {"trained": True, **metadata}


def predict_severity(features: dict) -> dict:
    with MODEL_LOCK:
        metadata = _load_metadata()
        if metadata is None or not MODEL_PATH.exists():
            raise DatasetError("Train a model with a labeled dataset before requesting a severity assessment.")
        model = joblib.load(MODEL_PATH)
    if not isinstance(features, dict):
        raise DatasetError("features must be an object containing the selected model fields.")
    expected = metadata["features"]
    if set(features) != set(expected):
        raise DatasetError("Provide exactly the feature fields used by the trained model.")

    row = {}
    for name in expected:
        value = features[name]
        if value is None or value == "":
            row[name] = np.nan
        elif metadata["feature_schema"][name] == "numeric":
            try:
                row[name] = float(value)
            except (TypeError, ValueError) as error:
                raise DatasetError(f"{name} must be numeric.") from error
            if not math.isfinite(row[name]):
                raise DatasetError(f"{name} must be a finite number.")
        else:
            row[name] = str(value)[:200]

    probabilities = model.predict_proba(pd.DataFrame([row], columns=expected))[0]
    ordered_classes = model.classes_.tolist()
    return {
        "severity": str(ordered_classes[int(np.argmax(probabilities))]),
        "probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(ordered_classes, probabilities)
        },
        "disclaimer": "Historical crash-severity assessment only; not a live accident-risk prediction or an emergency dispatch decision.",
        "warnings": metadata["warnings"],
    }
