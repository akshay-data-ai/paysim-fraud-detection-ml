"""PaySim fraud-detection demonstration app.

Uses the exact six-feature transformation from the Colab experiment.
Only load the trusted joblib artifact exported from that notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "fraud_detection_random_forest.joblib"
REQUIRED = ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "nameDest"]
MAX_BATCH_ROWS = 10000

st.set_page_config(page_title="PaySim Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Never load an untrusted joblib/pickle file."""
    if not MODEL_PATH.is_file():
        return None
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or not {
        "model", "feature_columns", "type_mapping", "threshold"
    }.issubset(bundle):
        raise ValueError("Model bundle does not match the expected Colab export.")
    expected = {
        "amount_log", "oldbalanceOrg", "oldbalanceDest",
        "amount_to_oldbalance_ratio", "is_merchant_dest", "type_code"
    }
    if set(bundle["feature_columns"]) != expected:
        raise ValueError("Unexpected feature schema in the model bundle.")
    return bundle


def make_features(transactions: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Reproduce the *no-time, no-SMOTE* Random Forest preprocessing exactly."""
    missing = sorted(set(REQUIRED).difference(transactions.columns))
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))

    txn = transactions.copy()
    txn["type"] = txn["type"].astype(str).str.strip().str.upper()
    known_types = bundle["type_mapping"]
    unknown = sorted(set(txn["type"]).difference(known_types))
    if unknown:
        raise ValueError("Unsupported transaction type(s): " + ", ".join(unknown))

    amount = pd.to_numeric(txn["amount"], errors="raise").to_numpy(dtype=np.float32)
    origin = pd.to_numeric(txn["oldbalanceOrg"], errors="raise").to_numpy(dtype=np.float32)
    dest = pd.to_numeric(txn["oldbalanceDest"], errors="raise").to_numpy(dtype=np.float32)
    for name, values in [
        ("amount", amount), ("oldbalanceOrg", origin), ("oldbalanceDest", dest)
    ]:
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(name + " must contain finite, non-negative numbers.")

    ratio = np.divide(
        amount, origin,
        out=np.zeros(len(txn), dtype=np.float32),
        where=origin > 0,
    ).clip(0, 1e5)

    features = pd.DataFrame({
        "amount_log": np.log1p(amount).astype(np.float32),
        "oldbalanceOrg": np.log1p(origin).astype(np.float32),
        "oldbalanceDest": np.log1p(dest).astype(np.float32),
        "amount_to_oldbalance_ratio": ratio.astype(np.float32),
        "is_merchant_dest": (
            txn["nameDest"].fillna("").astype(str).str.startswith("M")
            .astype(np.float32).to_numpy()
        ),
        "type_code": txn["type"].map(known_types).astype(np.int8).to_numpy(),
    }, index=txn.index)
    return features.loc[:, list(bundle["feature_columns"])]


def score(transactions: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    x = make_features(transactions, bundle)
    raw_score = bundle["model"].predict_proba(x)[:, 1]
    threshold = float(bundle["threshold"])
    output = transactions.copy()
    output["model_score_uncalibrated"] = np.round(raw_score, 6)
    output["prediction"] = np.where(
        raw_score >= threshold, "Flag for review", "Not flagged"
    )
    return output


st.title("🛡️ PaySim Fraud Detection")
st.caption("63.63 lakh simulated mobile-money transactions · Random Forest · Educational demonstration")
st.warning(
    "PaySim is synthetic. Scores are NOT calibrated real-world fraud probabilities. "
    "Do not use this prototype to block actual transactions or make banking decisions."
)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error("Unable to load the saved model. Check model file and package versions.")
    st.code(str(exc))
    st.stop()

overview_tab, single_tab, batch_tab, details_tab = st.tabs(
    ["Overview", "Single transaction", "Batch CSV", "Model details"]
)

with overview_tab:
    st.subheader("Final held-out test performance")
    a, b, c, d = st.columns(4)
    a.metric("Fraud recall", "99.94%")
    b.metric("Precision", "99.88%")
    c.metric("F1-score", "99.91%")
    d.metric("Average precision", "0.999988")
    st.caption(
        "Evaluation from the previously untouched chronological holdout: "
        "123,580 transactions, 1,654 labeled fraud cases, threshold 0.5."
    )
    st.dataframe(pd.DataFrame([
        {"Outcome": "Correct legitimate", "Count": 121924},
        {"Outcome": "False alarms", "Count": 2},
        {"Outcome": "Missed fraud", "Count": 1},
        {"Outcome": "Detected fraud", "Count": 1653},
    ]), hide_index=True, use_container_width=True)
    st.info(
        "The strongest signal in PaySim is the transaction amount divided by "
        "the sender's balance before the transaction. Near-full-balance "
        "TRANSFER and CASH_OUT transactions are especially predictive in this simulation."
    )

if bundle is None:
    st.info(
        "App code is ready, but the trained model file has not been added yet. "
        "Place your trusted Colab export at: "
        "paysim_app/models/fraud_detection_random_forest.joblib"
    )
else:
    with single_tab:
        st.subheader("Review one simulated transaction")
        with st.form("transaction"):
            left, right = st.columns(2)
            with left:
                txn_type = st.selectbox(
                    "Transaction type",
                    ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
                )
                amount = st.number_input("Transaction amount (dataset units)", min_value=0.0, value=10000.0, step=100.0)
                old_origin = st.number_input("Sender balance before transaction", min_value=0.0, value=10000.0, step=100.0)
            with right:
                old_destination = st.number_input("Recipient balance before transaction", min_value=0.0, value=0.0, step=100.0)
                merchant = st.checkbox("Destination is merchant account (M...)", value=False)
            submitted = st.form_submit_button("Analyze transaction", type="primary")
        if submitted:
            try:
                row = pd.DataFrame([{
                    "type": txn_type,
                    "amount": amount,
                    "oldbalanceOrg": old_origin,
                    "oldbalanceDest": old_destination,
                    "nameDest": "M_DEMO" if merchant else "C_DEMO",
                }])
                prediction = score(row, bundle).iloc[0]
                flagged = prediction["prediction"] == "Flag for review"
                if flagged:
                    st.error("Flag for review — model detected a PaySim-like fraud pattern.")
                else:
                    st.success("Not flagged by this model at its saved threshold.")
                st.metric("Uncalibrated model score", f'{prediction["model_score_uncalibrated"]:.3f}')
                st.caption("This score is not a trustworthy probability of real-world fraud.")
            except Exception as exc:
                st.error(f"Unable to score transaction: {exc}")

    with batch_tab:
        st.subheader("Score a CSV of simulated transactions")
        st.write(
            "Required columns: type, amount, oldbalanceOrg, oldbalanceDest, nameDest. "
            f"Maximum {MAX_BATCH_ROWS:,} rows. Avoid uploading real customer data."
        )
        template = pd.DataFrame([
            {"type": "TRANSFER", "amount": 10000, "oldbalanceOrg": 10000, "oldbalanceDest": 0, "nameDest": "C_DEMO"},
            {"type": "PAYMENT", "amount": 100, "oldbalanceOrg": 15000, "oldbalanceDest": 0, "nameDest": "M_DEMO"}
        ])
        st.download_button("Download sample CSV", template.to_csv(index=False), "paysim_sample.csv", "text/csv")
        uploaded = st.file_uploader("Choose a CSV", type="csv")
        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("CSV exceeds the 10 MB demonstration upload limit.")
            else:
                try:
                    transactions = pd.read_csv(uploaded, nrows=MAX_BATCH_ROWS + 1)
                    if len(transactions) > MAX_BATCH_ROWS:
                        st.error(f"CSV exceeds the {MAX_BATCH_ROWS:,}-row limit.")
                    else:
                        results = score(transactions, bundle)
                        flagged_count = int((results["prediction"] == "Flag for review").sum())
                        st.metric("Flagged for review", f"{flagged_count:,} / {len(results):,}")
                        st.dataframe(results.head(200), hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download scored CSV",
                            results.to_csv(index=False).encode("utf-8"),
                            "paysim_scored_transactions.csv",
                            "text/csv",
                        )
                except Exception as exc:
                    st.error(f"Unable to process CSV: {exc}")

with details_tab:
    st.subheader("How the model was developed")
    st.write(
        "Random Forest trained on 305,315 sampled training transactions "
        "(all 5,315 training fraud cases and 300,000 sampled legitimate records). "
        "Validation and chronological test sets retained their original prevalence."
    )
    st.write(
        "Six features are used: log(amount), log(original sender balance), "
        "log(original recipient balance), amount / original sender balance, "
        "merchant destination flag, and encoded transaction type. "
        "Simulated time features, post-transaction balances, and isFlaggedFraud are excluded."
    )
    st.write(
        "SMOTENC did not materially improve validation classification results; "
        "the deployed model was trained without synthetic oversampling."
    )
    if bundle is not None:
        model = bundle["model"]
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": bundle["feature_columns"],
                "Importance": model.feature_importances_,
            }).sort_values("Importance")
            st.bar_chart(importance.set_index("Feature"))
        st.caption(f"Saved decision threshold: {float(bundle['threshold']):.2f}")
    st.info(
        "High scores reflect unusually predictable patterns in the simulated PaySim data. "
        "Further calibration, independent real-world testing, bias and drift checks, "
        "and operational controls would be necessary before real financial use."
    )
