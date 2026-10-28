"""Streamlit entry point for the guarded binary-classification AutoML demo."""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from automl_engine import (
    MAX_COLUMNS, MAX_ROWS, predict_new_rows,
    suggested_exclusions, train_automl
)

st.set_page_config(page_title="Fraud AutoML Agent", page_icon="🤖", layout="wide")
st.title("🤖 Fraud AutoML Agent")
st.caption("Upload a labeled CSV → inspect → compare five models → review one final holdout")
st.warning(
    "Educational prototype. Results are not proof of real-world fraud detection. "
    "Uploaded training data is processed within this Streamlit session. "
    "Do not upload real customer records, credentials or sensitive data."
)

st.markdown(
    """
    ### How it works
    1. Supply a **labeled binary** CSV with a fraud target and at least 10 observations of each class.
    2. Confirm the target, positive/fraud label, excluded columns and optional chronological step.
    3. Run five ML algorithms with train-only preprocessing and compare validation average precision.
    4. Inspect the winner's one-time held-out test metrics and score new CSV records.
    """
)

uploaded = st.file_uploader(
    "Upload labeled CSV (up to 10 MB, 50,000 rows, 50 feature columns)",
    type="csv", key="training_csv"
)
if uploaded is None:
    st.info("Start by uploading a small, labeled CSV. For the full 6.36M-row PaySim dataset, use Colab or a dedicated training server.")
    st.stop()

if uploaded.size > 10 * 1024 * 1024:
    st.error("This file exceeds the 10 MB hosted demonstration limit.")
    st.stop()

try:
    data = pd.read_csv(uploaded, nrows=MAX_ROWS + 1)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

if len(data) > MAX_ROWS or len(data.columns) > MAX_COLUMNS + 2:
    st.error(f"Use at most {MAX_ROWS:,} rows and approximately {MAX_COLUMNS} features.")
    st.stop()
if len(data) == 0:
    st.error("The uploaded file is empty.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Rows loaded", f"{len(data):,}")
m2.metric("Columns", len(data.columns))
m3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
with st.expander("Preview dataset and missing values", expanded=True):
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
    missing = data.isna().sum()
    st.dataframe(missing[missing > 0].rename("Missing").to_frame(), use_container_width=True)

candidates = [c for c in data if data[c].dropna().nunique() == 2]
if not candidates:
    st.error("No columns with two unique target values were found. Please upload a labeled binary classification dataset.")
    st.stop()

preferred = next(
    (c for c in candidates if c.lower() in ("isfraud", "fraud", "target", "label")),
    candidates[0],
)
target = st.selectbox("Target (the column the model must predict)", candidates, index=candidates.index(preferred))
labels = sorted(data[target].dropna().astype(str).unique())
positive_default = next(
    (str(x) for x in labels if x.lower() in ("1", "true", "fraud", "yes")), labels[-1]
)
positive = st.selectbox(
    "Positive / fraudulent target value", labels,
    index=labels.index(positive_default)
)

time_options = ["None"] + [col for col in data.columns if col != target]
time = st.selectbox(
    "Chronological order column (optional)",
    time_options,
    help="Choose step or an event timestamp if later transactions must be held out. "
         "Otherwise the app uses a stratified random split.",
)
time_col = None if time == "None" else time

suggested = suggested_exclusions(data, target, time_col)
excluded = st.multiselect(
    "Exclude identifiers, post-event information and leakage-prone columns",
    [c for c in data.columns if c not in {target, time_col}],
    default=suggested,
    help="Review these carefully. Post-transaction balances and existing fraud flags "
         "are not usually available before the prediction decision.",
)
st.caption(
    "Model selection uses validation **Average Precision**; thresholds are chosen "
    "using validation F1. The final test split is not used to tune the models."
)
st.info(
    "This demo has a fixed algorithm comparison, not exhaustive hyperparameter search. "
    "It does not automatically guarantee that the recommended model is safe or optimal."
)

if st.button("🚀 Run AutoML comparison", type="primary", use_container_width=True):
    try:
        with st.spinner("Preparing data and comparing models..."):
            result = train_automl(
                data, target, positive, excluded_cols=excluded, time_col=time_col,
            )
        st.session_state["automl_result"] = result
        st.session_state["automl_fingerprint"] = (
            uploaded.name, uploaded.size, target, positive,
            time_col, tuple(excluded)
        )
    except Exception as exc:
        st.session_state.pop("automl_result", None)
        st.error(f"Training cannot proceed: {exc}")

key = (uploaded.name, uploaded.size, target, positive, time_col, tuple(excluded))
result = st.session_state.get("automl_result")
if result is None or st.session_state.get("automl_fingerprint") != key:
    st.stop()

st.success(f"Recommended model: **{result['model_name']}** (highest validation average precision)")
x1, x2, x3, x4 = st.columns(4)
x1.metric("Train rows", f"{result['training_rows']:,}")
x2.metric("Validation rows", f"{result['validation_rows']:,}")
x3.metric("Test rows", f"{result['test_rows']:,}")
x4.metric("Selected threshold", f"{result['threshold']:.4f}")

st.subheader("Validation model leaderboard")
st.dataframe(
    result["table"].round(5), hide_index=True, use_container_width=True
)
st.download_button(
    "Download leaderboard CSV",
    result["table"].to_csv(index=False),
    "automl_validation_models.csv", mime="text/csv"
)
st.caption(
    f"Dummy-classifier validation AP: {result['dummy_validation_ap']:.5f}. "
    f"Validation fraud prevalence: {result['validation_fraud_rate']:.2%}."
)
st.bar_chart(
    result["table"].set_index("Model")[["Average Precision"]],
    horizontal=True,
    height=260
)

st.subheader("Final holdout evaluation — selected model only")
last = result["test_metrics"]
a, b, c, d = st.columns(4)
a.metric("Test AP", f"{last['Average Precision']:.4f}")
b.metric("Test precision", f"{last['Precision']:.2%}")
c.metric("Test recall", f"{last['Recall']:.2%}")
d.metric("Test F1", f"{last['F1']:.2%}")
st.dataframe(pd.DataFrame([last]), hide_index=True, use_container_width=True)
st.caption(
    "The final test set was evaluated once after model selection. "
    "Avoid choosing another model based on this result; doing so invalidates "
    "the unbiased holdout assessment."
)

with st.expander("Training report", expanded=False):
    st.write("Selected predictors:", result["features"])
    st.write("Excluded:", result["excluded"])
    st.write("Automatically removed empty/constant predictors:", result["removed_constant"])
    st.write("Split:", "Chronological by " + time_col if time_col else "Stratified random 60/20/20")
    st.write(f"Final test positive prevalence: {result['test_fraud_rate']:.2%}")

st.subheader("Score new transactions using the recommended model")
st.caption(
    "Upload an unlabeled CSV containing the same feature columns. "
    "Scores may be uncalibrated. Maximum 50,000 rows / 10 MB."
)
new_file = st.file_uploader("Upload new CSV to score", type="csv", key="new_csv")
if new_file:
    if new_file.size > 10 * 1024 * 1024:
        st.error("New CSV exceeds 10 MB.")
    else:
        try:
            new = pd.read_csv(new_file, nrows=MAX_ROWS + 1)
            if len(new) > MAX_ROWS:
                st.error("New CSV exceeds the row limit.")
            else:
                scored = predict_new_rows(result, new)
                st.metric("Flagged as positive", int((scored["predicted_label"] == result["positive_label"]).sum()))
                st.dataframe(scored.head(300), hide_index=True, use_container_width=True)
                st.download_button(
                    "Download all predictions",
                    scored.to_csv(index=False).encode("utf-8"),
                    "automl_predictions.csv", mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Unable to score new rows: {exc}")

st.subheader("Export your winning model")
buffer = io.BytesIO()
joblib.dump({
    "model": result["model"],
    "positive_label": result["positive_label"],
    "negative_label": result["negative_label"],
    "threshold": result["threshold"],
    "features": result["features"],
    "source": "PaySim Streamlit AutoML demo",
    "notes": "Educational use; scores are not guaranteed calibrated.",
}, buffer, compress=3)
st.download_button(
    "Download trained pipeline (.joblib)",
    buffer.getvalue(),
    "automl_best_model.joblib",
    mime="application/octet-stream",
)
st.caption("Model files use Python pickle serialization. Load only trusted files.")
