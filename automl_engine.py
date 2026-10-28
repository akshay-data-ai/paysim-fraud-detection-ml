"""Conservative, reproducible AutoML baseline for labeled binary CSV datasets.

Resource guardrails and preprocessing are designed for a *demo*, not unlimited CSVs.
All preprocessing is fit inside a pipeline using training data only.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MAX_ROWS = 50_000
MAX_COLUMNS = 50

# Post-event labels and account identifiers must not leak into pre-event inference.
LIKELY_LEAKAGE = {
    "isflaggedfraud", "newbalanceorig", "newbalancedest",
    "fraudstatus", "frauddecision", "reviewoutcome", "chargebackstatus",
    "confirmedfraud", "investigationresult",
}
IDENTIFIER = re.compile(
    r"(?:^|_)(?:id|uuid|hash|customerid|accountid|transactionid|reference)(?:$|_)",
    re.I,
)


def suggested_exclusions(frame: pd.DataFrame, target: str, time_col: str | None):
    result = []
    for col in frame.columns:
        if col in {target, time_col}:
            continue
        normalized = re.sub(r"[^a-z0-9]", "", col.lower())
        if (
            normalized in LIKELY_LEAKAGE
            or col.lower() in {"nameorig", "namedest"}
            or IDENTIFIER.search(col)
        ):
            result.append(col)
            continue
        col_data = frame[col]
        if (
            (pd.api.types.is_object_dtype(col_data) or
             pd.api.types.is_string_dtype(col_data))
            and col_data.nunique(dropna=True) >= 200
            and col_data.nunique(dropna=True) / max(len(frame), 1) > .90
        ):
            result.append(col)
    return result


def _split(frame: pd.DataFrame, y: pd.Series, time_col: str | None):
    if time_col:
        if time_col not in frame:
            raise ValueError("Chronological column missing.")
        if frame[time_col].isna().any():
            raise ValueError("Time column has missing values. Fix these before chronological splitting.")
        chronological = frame.sort_values(time_col, kind="mergesort")
        sorted_y = y.loc[chronological.index]
        # Split at distinct time-step boundaries: the same simulated step
        # must never appear in both training and validation/test periods.
        unique_steps = chronological[time_col].drop_duplicates().sort_values().to_numpy()
        if len(unique_steps) < 5:
            raise ValueError("Need at least five unique time steps for a chronological split.")
        first = unique_steps[int(len(unique_steps) * .60)]
        second = unique_steps[int(len(unique_steps) * .80)]
        period = chronological[time_col]
        masks = (
            period < first,
            (period >= first) & (period < second),
            period >= second,
        )
        parts = tuple(
            (chronological.loc[mask], sorted_y.loc[mask])
            for mask in masks
        )
        # A different label appearing late is common in event data; do not hide it.
        for label, (_, target_subset) in zip(("training", "validation", "test"), parts):
            if len(np.unique(target_subset)) != 2:
                raise ValueError(
                    f"Chronological {label} split has only one class. "
                    "This period cannot support reliable binary evaluation. "
                    "Choose a larger dataset or review the time window."
                )
        return parts

    # 60/20/20, stratified on raw labels. Test set remains untouched.
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        frame, y, test_size=.20, stratify=y, random_state=RANDOM_STATE
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=.25,
        stratify=y_trainval, random_state=RANDOM_STATE
    )
    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))


def _feature_schema(frame: pd.DataFrame):
    if frame.shape[1] == 0:
        raise ValueError("There are no usable features after exclusions.")
    if frame.shape[1] > MAX_COLUMNS:
        raise ValueError(f"Too many feature columns (maximum {MAX_COLUMNS}).")
    numerical = [
        col for col in frame
        if pd.api.types.is_numeric_dtype(frame[col])
        and not pd.api.types.is_bool_dtype(frame[col])
    ]
    categorical = [col for col in frame if col not in numerical]
    # Datetime fields should be excluded explicitly as a chronological column.
    bad_types = [
        col for col in categorical
        if pd.api.types.is_datetime64_any_dtype(frame[col])
    ]
    if bad_types:
        raise ValueError("Remove datetime feature columns or choose one as the time column: "
                         + ", ".join(bad_types))
    # Exclude features with no variation before training, to prevent imputation errors.
    empty = [col for col in frame if frame[col].notna().sum() == 0]
    constant = [
        col for col in frame if frame[col].nunique(dropna=True) <= 1
    ]
    remove = sorted(set(empty + constant))
    numerical = [col for col in numerical if col not in remove]
    categorical = [col for col in categorical if col not in remove]
    if not numerical and not categorical:
        raise ValueError("All features are empty or constant.")
    return numerical, categorical, remove


AUTO_FEATURES = ("auto_amount_balance_ratio", "auto_log_amount")


def add_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive row-local financial signals without looking at fraud labels.

    FunctionTransformer keeps the same logic within exported pipelines,
    including when scoring future unlabeled data.
    """
    result = frame.copy()
    if {"amount", "oldbalanceOrg"}.issubset(result.columns):
        amount = pd.to_numeric(result["amount"], errors="coerce").to_numpy(dtype=np.float64)
        balance = pd.to_numeric(result["oldbalanceOrg"], errors="coerce").to_numpy(dtype=np.float64)
        valid_amount = np.isfinite(amount) & (amount >= 0)
        valid_balance = np.isfinite(balance) & (balance >= 0)
        ratio = np.full(len(result), np.nan, dtype=np.float64)
        valid = valid_amount & valid_balance
        ratio[valid & (balance == 0)] = 0.0
        np.divide(amount, balance, out=ratio, where=valid & (balance > 0))
        result[AUTO_FEATURES[0]] = np.clip(ratio, 0, 100000).astype(np.float32)
        logs = np.full(len(result), np.nan, dtype=np.float64)
        logs[valid_amount] = np.log1p(amount[valid_amount])
        result[AUTO_FEATURES[1]] = logs.astype(np.float32)
    return result


def _pipeline(model, numerical, categorical):
    transformers = []
    if numerical:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=12,
                    sparse_output=False,
                    dtype=np.float32,
                )),
            ]),
            categorical,
        ))
    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("auto_features", FunctionTransformer(add_transaction_features, validate=False)), ("preprocess", preprocessing), ("classifier", model)])


def _models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=2, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=3,
            class_weight="balanced", n_jobs=2, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=25, learning_rate=0.08,
            class_weight="balanced", early_stopping=True,
            random_state=RANDOM_STATE
        ),
    }


def _validation_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.argmax(values))])


def _metric_row(name, y_true, probabilities, threshold, fit_seconds):
    labels = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Average Precision": float(average_precision_score(y_true, probabilities)),
        "Precision": float(precision_score(y_true, labels, zero_division=0)),
        "Recall": float(recall_score(y_true, labels, zero_division=0)),
        "F1": float(f1_score(y_true, labels, zero_division=0)),
        "Threshold": float(threshold),
        "False Positives": int(fp),
        "Missed Frauds": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
        "Train seconds": round(fit_seconds, 2),
    }


def train_automl(
    data: pd.DataFrame,
    target_col: str,
    positive_label: str,
    *,
    excluded_cols: list[str] | None = None,
    time_col: str | None = None,
):
    """Train five candidate models; pick on validation AP; test winner once.

    Returns a pipeline plus metrics and metadata for this session only.
    """
    if len(data) > MAX_ROWS:
        raise ValueError(f"Too many rows for hosted demo (maximum {MAX_ROWS:,}).")
    if len(data) < 100:
        raise ValueError("Use at least 100 labeled rows for this demonstration.")
    if target_col not in data.columns:
        raise ValueError("Selected target column does not exist.")
    if time_col == target_col:
        raise ValueError("Target and time column cannot be the same.")

    working = data.dropna(subset=[target_col]).copy()
    labels = working[target_col].astype(str)
    uniques = sorted(labels.unique())
    if len(uniques) != 2:
        raise ValueError("Only binary classification is supported (two non-missing labels).")
    if positive_label not in uniques:
        raise ValueError("Selected positive label was not found.")
    y = (labels == positive_label).astype(np.int8)
    counts = y.value_counts()
    if counts.min() < 10:
        raise ValueError(
            "At least 10 examples of each class are needed for a meaningful comparison."
        )
    ignored = set(excluded_cols or [])
    ignored.add(target_col)
    if time_col:
        ignored.add(time_col)
    selected = [c for c in working.columns if c not in ignored]
    if len(selected) == 0:
        raise ValueError("No features selected.")
    numeric, categorical, constant = _feature_schema(working[selected])
    cols = numeric + categorical
    # Calculated from permitted pre-transaction columns. The engineered
    # features are computed inside every fitted pipeline and for inference.
    derived = [c for c in AUTO_FEATURES if c in add_transaction_features(working[cols]).columns]
    numeric = numeric + derived
    # Retain time column only for chronological splitting, not training.
    x_data = working[cols + ([time_col] if time_col else [])].copy()
    splits = _split(x_data, y, time_col)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = splits
    train_x = train_x[cols]
    val_x = val_x[cols]
    test_x = test_x[cols]

    comparison = []
    trained = {}
    for name, estimator in _models().items():
        pipeline = _pipeline(estimator, numeric, categorical)
        tic = time.perf_counter()
        pipeline.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - tic
        probabilities = pipeline.predict_proba(val_x)[:, 1]
        threshold = _validation_threshold(val_y, probabilities)
        row = _metric_row(name, val_y, probabilities, threshold, fit_seconds)
        comparison.append(row)
        trained[name] = (pipeline, threshold)

    # Random but constant-probability benchmark: not eligible to "win".
    dummy = _pipeline(DummyClassifier(strategy="prior"), numeric, categorical)
    dummy.fit(train_x, train_y)
    dummy_ap = average_precision_score(val_y, dummy.predict_proba(val_x)[:, 1])

    table = (
        pd.DataFrame(comparison)
        .sort_values(
            ["Average Precision", "F1", "False Positives"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    best_name = str(table.iloc[0]["Model"])
    winner, threshold = trained[best_name]

    # This is evaluated ONCE, after selecting the model and threshold on validation.
    test_probabilities = winner.predict_proba(test_x)[:, 1]
    test_row = _metric_row(
        best_name, test_y, test_probabilities, threshold, 0.0
    )
    test_row["ROC AUC"] = float(roc_auc_score(test_y, test_probabilities))
    return {
        "model": winner,
        "model_name": best_name,
        "threshold": threshold,
        "table": table,
        "test_metrics": test_row,
        "training_rows": len(train_x),
        "validation_rows": len(val_x),
        "test_rows": len(test_x),
        "positive_label": positive_label,
        "negative_label": next(item for item in uniques if item != positive_label),
        "target_column": target_col,
        "features": cols,
        "derived_features": derived,
        "removed_constant": constant,
        "excluded": sorted(ignored - {target_col}),
        "time_column": time_col,
        "dummy_validation_ap": float(dummy_ap),
        "validation_fraud_rate": float(val_y.mean()),
        "test_fraud_rate": float(test_y.mean()),
    }


def predict_new_rows(result: dict, new_data: pd.DataFrame):
    required = result["features"]
    missing = sorted(set(required).difference(new_data.columns))
    if missing:
        raise ValueError("New data is missing: " + ", ".join(missing))
    probabilities = result["model"].predict_proba(new_data[required])[:, 1]
    out = new_data.copy()
    out["automl_score_uncalibrated"] = np.round(probabilities, 6)
    out["predicted_label"] = np.where(
        probabilities >= result["threshold"],
        result["positive_label"],
        result["negative_label"],
    )
    return out
