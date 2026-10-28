"""Interactive PaySim fraud-risk demo.

Uses the exact six-feature preprocessing and 0.50 decision threshold from
the final no-SMOTE Random Forest experiment. Synthetic data only.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "fraud_detection_random_forest.joblib"
REQUIRED = ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "nameDest"]
MAX_BATCH_ROWS = 10_000

MODEL_SCORES = pd.DataFrame(
    {
        "Model": [
            "Decision Tree", "Random Forest", "XGBoost",
            "HistGradientBoosting", "Logistic Regression"
        ],
        "F1 (%)": [99.92, 99.88, 98.57, 93.56, 49.72],
        "Precision (%)": [100.00, 99.92, 97.34, 88.02, 58.92],
        "Recall (%)": [99.84, 99.84, 99.84, 99.84, 43.01],
    }
)
PERMUTATION = pd.DataFrame(
    {
        "Feature": [
            "Amount-to-balance ratio", "Transaction amount (log)",
            "Sender opening balance (log)", "Recipient opening balance (log)",
            "Transaction type", "Merchant destination",
        ],
        "AP decrease": [
            0.879364, 0.004931, 0.000737,
            0.000301, 0.000182, 0.000084,
        ],
    }
)
RATIO_HEATMAP = pd.DataFrame(
    {
        "Balance use": ["Zero balance", "Below 50%", "50-90%", "90-99%", "99-101%", "Over 101%"],
        "TRANSFER fraud %": [0.00, 1.31, 3.92, 1.39, 97.90, 0.00],
        "CASH_OUT fraud %": [0.01, 0.00, 0.00, 0.00, 84.82, 0.00],
    }
)
SCENARIOS = {
    "Full-balance transfer": {
        "transaction_type": "TRANSFER", "txn_amount": 10_000.0,
        "origin_balance": 10_000.0, "destination_balance": 0.0, "merchant": False,
    },
    "Small payment": {
        "transaction_type": "PAYMENT", "txn_amount": 150.0,
        "origin_balance": 10_000.0, "destination_balance": 0.0, "merchant": True,
    },
    "Partial cash-out": {
        "transaction_type": "CASH_OUT", "txn_amount": 2_000.0,
        "origin_balance": 10_000.0, "destination_balance": 4_000.0, "merchant": False,
    },
    "Full-balance cash-out": {
        "transaction_type": "CASH_OUT", "txn_amount": 10_000.0,
        "origin_balance": 10_000.0, "destination_balance": 0.0, "merchant": False,
    },
}

st.set_page_config(
    page_title="PaySim | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
    html, body, [class*="css"] {font-family: system-ui, -apple-system, sans-serif;}
    .hero {
        padding: 2.1rem 2.3rem; border-radius: 23px;
        background: linear-gradient(115deg, #102b46 0%, #174765 62%, #087f7c 100%);
        color: #fff; box-shadow: 0 15px 35px rgba(18,48,76,.13);
        margin-bottom: 1.15rem;
    }
    .eyebrow {font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
        font-weight: 750; color: #9ff3dd; margin-bottom: .65rem;}
    .hero h1 {color: #fff; font-size: clamp(1.75rem, 3.2vw, 2.85rem);
        letter-spacing: -.04em; line-height: 1.12; margin: 0 0 .65rem 0;}
    .hero p {font-size: 1rem; color: #d6e9ed; margin: 0; max-width: 790px;}
    .hero-pill {display: inline-block; border: 1px solid #80d9d1;
        background: rgba(255,255,255,.10); padding: .32rem .72rem;
        border-radius: 50px; color: #e8fff7; font-size: .78rem;
        margin-top: 1.25rem; margin-right: .4rem;}
    .section-heading {font-size: 1.12rem; font-weight: 760; color: #15324d;
        margin: 1.2rem 0 .55rem;}
    .note-card {padding: .95rem 1.15rem; border: 1px solid #d5e8e9;
        background: #edf8f7; border-radius: 14px; color: #214a54;
        line-height: 1.6; margin: .6rem 0 1rem;}
    .warning-card {padding: .8rem 1.1rem; background: #fff8e8;
        border: 1px solid #f3dfac; border-radius: 14px;
        color: #825515; margin-bottom: 1rem; font-size: .86rem;}
    .mini-label {font-size:.72rem; color:#688092; letter-spacing:.08em;
        font-weight:760; text-transform:uppercase;}
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #e3ebf1;
        padding: 1rem 1.05rem; border-radius: 16px;
        box-shadow: 0 4px 14px rgba(27,52,73,.035);
    }
    div[data-testid="stMetricLabel"] {color:#567088;}
    div[data-testid="stMetricValue"] {color:#15324d; font-weight:750;}
    button[kind="primary"] {border-radius: 11px !important;}
    div[data-baseweb="tab-list"] {gap: .45rem;}
    div[data-baseweb="tab"] {font-weight: 660; padding-inline: 1rem;}
    .footer-note {font-size: .78rem; color: #6b8595; text-align: center;
        border-top: 1px solid #dce8ef; padding-top: 1.2rem; margin-top: 2rem;}
    @media(max-width: 700px) {
       .hero {padding: 1.45rem 1.25rem; border-radius: 17px;}
       .hero-pill {margin-top:.75rem;}
       .block-container {padding-top:1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Read only the model artifact produced by the user's own Colab run."""
    if not MODEL_PATH.is_file():
        return None
    bundle = joblib.load(MODEL_PATH)  # Never load untrusted pickle/joblib data.
    mandatory = {"model", "feature_columns", "type_mapping", "threshold"}
    if not isinstance(bundle, dict) or not mandatory.issubset(bundle):
        raise ValueError("Saved model bundle is missing required fields.")
    expected = {
        "amount_log", "oldbalanceOrg", "oldbalanceDest",
        "amount_to_oldbalance_ratio", "is_merchant_dest", "type_code",
    }
    if set(bundle["feature_columns"]) != expected:
        raise ValueError("Model's feature schema does not match this app.")
    if not (0 < float(bundle["threshold"]) < 1):
        raise ValueError("Invalid model decision threshold.")
    return bundle


def make_features(transactions: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Reproduce the exact features used when the saved Random Forest was fit."""
    missing = sorted(set(REQUIRED).difference(transactions.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    txn = transactions.copy()
    txn["type"] = txn["type"].astype(str).str.strip().str.upper()
    unmapped = sorted(set(txn["type"]) - set(bundle["type_mapping"]))
    if unmapped:
        raise ValueError("Unknown transaction type(s): " + ", ".join(unmapped))

    amount = pd.to_numeric(txn["amount"], errors="raise").to_numpy(dtype=np.float32)
    origin = pd.to_numeric(txn["oldbalanceOrg"], errors="raise").to_numpy(dtype=np.float32)
    recipient = pd.to_numeric(txn["oldbalanceDest"], errors="raise").to_numpy(dtype=np.float32)
    for label, values in [
        ("amount", amount),
        ("oldbalanceOrg", origin),
        ("oldbalanceDest", recipient),
    ]:
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(label + " must contain only finite non-negative values.")

    ratio = np.divide(
        amount,
        origin,
        out=np.zeros(len(txn), dtype=np.float32),
        where=origin > 0,
    ).clip(0, 1e5)

    prepared = pd.DataFrame(
        {
            "amount_log": np.log1p(amount).astype(np.float32),
            "oldbalanceOrg": np.log1p(origin).astype(np.float32),
            "oldbalanceDest": np.log1p(recipient).astype(np.float32),
            "amount_to_oldbalance_ratio": ratio.astype(np.float32),
            "is_merchant_dest": (
                txn["nameDest"].fillna("").astype(str).str.startswith("M")
                .astype(np.float32).to_numpy()
            ),
            "type_code": txn["type"].map(bundle["type_mapping"])
            .astype(np.int8).to_numpy(),
        },
        index=txn.index,
    )
    return prepared.loc[:, list(bundle["feature_columns"])]


def score(transactions: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features = make_features(transactions, bundle)
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    out = transactions.copy()
    out["model_score_uncalibrated"] = np.round(probabilities, 6)
    out["prediction"] = np.where(
        probabilities >= float(bundle["threshold"]),
        "Flag for review",
        "Not flagged",
    )
    return out


def scenario_to_frame(txn_type, amount, origin, recipient, merchant):
    return pd.DataFrame(
        [{
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": origin,
            "oldbalanceDest": recipient,
            "nameDest": "M_DEMO" if merchant else "C_DEMO",
        }]
    )


def set_preset():
    selected = SCENARIOS[st.session_state["preset_name"]]
    for key, value in selected.items():
        st.session_state[key] = value


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Portfolio project / Financial risk intelligence</div>
      <h1>🛡️ PaySim Fraud Intelligence</h1>
      <p>Explore transaction behaviour, test fraud-detection scenarios and inspect
         the Random Forest model trained on 6.36 million simulated mobile-money transactions.</p>
      <span class="hero-pill">Random Forest · 6 features</span>
      <span class="hero-pill">Chronological holdout</span>
      <span class="hero-pill">Interactive risk exploration</span>
    </div>
    <div class="warning-card">
      <b>Educational demo only.</b> PaySim is synthetic and model scores are uncalibrated.
      They are not real-world fraud probabilities or a basis for banking decisions.
      Please do not upload real customer information.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error("The saved model could not be loaded. Confirm the artifact and scikit-learn version.")
    st.code(str(exc))
    st.stop()

if bundle is None:
    st.error("Model not found. Add models/fraud_detection_random_forest.joblib to the deployment.")
    st.stop()

overview_tab, txn_tab, batch_tab, explain_tab = st.tabs(
    ["📊 Overview", "🔎 Transaction lab", "📂 Batch explorer", "🧠 Model insights"]
)

with overview_tab:
    st.markdown('<div class="section-heading">Final evaluation snapshot</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud recall", "99.94%", help="1,653 of 1,654 frauds detected on the test period.")
    m2.metric("Precision", "99.88%", help="1,653 true detections and 2 false alarms.")
    m3.metric("F1-score", "99.91%")
    m4.metric("Average precision", "0.999988", help="Area-style summary of the precision–recall ranking.")
    st.caption(
        "Final untouched chronological test set: 123,580 transactions; "
        "1,654 labeled fraud cases; threshold 0.50."
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-heading">How the predictions landed</div>', unsafe_allow_html=True)
        confusion = pd.DataFrame(
            [
                {"Outcome": "Legitimate correctly recognized", "Transactions": 121924},
                {"Outcome": "Fraud successfully detected", "Transactions": 1653},
                {"Outcome": "False alarm", "Transactions": 2},
                {"Outcome": "Fraud missed", "Transactions": 1},
            ]
        )
        st.dataframe(confusion, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="note-card"><b>Why so accurate?</b> In PaySim, '
            'transactions that use almost the full sender balance are '
            'especially predictive of fraud. Real transaction data may be far less predictable.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-heading">Explore the five-model benchmark</div>', unsafe_allow_html=True)
        metric_choice = st.selectbox(
            "Compare by", ["F1 (%)", "Precision (%)", "Recall (%)"],
            key="model_metric",
        )
        ranked = MODEL_SCORES.sort_values(metric_choice, ascending=False)
        st.bar_chart(
            ranked.set_index("Model")[[metric_choice]],
            horizontal=True,
            height=280,
        )
        st.caption(
            "Original five-model VALIDATION experiment. Features and optimized "
            "thresholds differed from the later six-feature deployment comparison."
        )

    with st.expander("Why Random Forest instead of Decision Tree?", expanded=False):
        st.write(
            "In the initial five-model experiment, Decision Tree had slightly higher F1. "
            "After the time-feature robustness study, the six-feature comparison "
            "at threshold 0.50 detected all 1,244 validation frauds with each model, "
            "but Random Forest raised 3 false alarms versus 87 for Decision Tree. "
            "The selected Random Forest was then evaluated once on the chronological test set."
        )

with txn_tab:
    st.markdown('<div class="section-heading">Transaction simulator</div>', unsafe_allow_html=True)
    st.write(
        "Load a sample scenario, change the transaction details, and explore what "
        "the saved model flags. These are illustrative PaySim-style inputs."
    )
    if "preset_name" not in st.session_state:
        st.session_state["preset_name"] = "Full-balance transfer"
    for key, value in SCENARIOS["Full-balance transfer"].items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.selectbox(
        "Try a sample scenario",
        list(SCENARIOS),
        key="preset_name",
        on_change=set_preset,
    )
    with st.form("single_transaction"):
        col_a, col_b = st.columns(2)
        with col_a:
            txn_type = st.selectbox(
                "Transaction type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                key="transaction_type",
            )
            amount = st.number_input(
                "Amount transferred (simulation units)",
                min_value=0.0, step=100.0, key="txn_amount",
            )
            origin = st.number_input(
                "Sender balance before transaction",
                min_value=0.0, step=100.0, key="origin_balance",
            )
        with col_b:
            recipient = st.number_input(
                "Recipient balance before transaction",
                min_value=0.0, step=100.0, key="destination_balance",
            )
            merchant = st.checkbox(
                "Recipient is a merchant account (M...)",
                key="merchant",
            )
            st.caption(
                "The model uses balances from BEFORE the transaction. "
                "No account IDs, simulated step or post-transaction balances are needed."
            )
        submitted = st.form_submit_button(
            "🔍 Analyze transaction", type="primary", use_container_width=True,
        )

    if submitted:
        try:
            transaction = scenario_to_frame(txn_type, amount, origin, recipient, merchant)
            prediction = score(transaction, bundle).iloc[0]
            st.session_state["single_result"] = {
                "score": float(prediction["model_score_uncalibrated"]),
                "decision": str(prediction["prediction"]),
                "ratio": float(amount / origin) if origin > 0 else 0.0,
                "scenario_frame": transaction,
                "balance": origin,
            }
        except Exception as exc:
            st.error(f"Could not score this transaction: {exc}")
            st.session_state.pop("single_result", None)

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        st.markdown('<div class="section-heading">Last analyzed transaction</div>', unsafe_allow_html=True)
        if result["decision"] == "Flag for review":
            st.error("⚑ Flag for review — this matches a suspicious PaySim pattern.")
        else:
            st.success("✓ Not flagged at the saved 0.50 threshold.")
        a, b, c = st.columns(3)
        a.metric("Model score · uncalibrated", f'{result["score"]:.3f}')
        b.metric("Share of sender balance", f'{result["ratio"] * 100:,.1f}%')
        c.metric("Decision threshold", f'{float(bundle["threshold"]):.2f}')
        st.progress(min(max(result["score"], 0), 1))
        st.caption(
            "This score is not a calibrated fraud probability. "
            "Change any input and select Analyze transaction to refresh it."
        )

        if result["balance"] > 0:
            if st.checkbox("Show amount sensitivity curve", value=False):
                ratios = np.linspace(0.10, 1.50, 36)
                points = pd.concat(
                    [result["scenario_frame"]] * len(ratios), ignore_index=True
                )
                points["amount"] = (
                    ratios * float(result["balance"])
                ).astype(np.float32)
                curve = score(points, bundle)
                curve_data = pd.DataFrame({
                    "Amount as % of sender balance": np.round(ratios * 100, 1),
                    "Uncalibrated model score": curve["model_score_uncalibrated"].values,
                })
                st.line_chart(
                    curve_data.set_index("Amount as % of sender balance"),
                    height=270,
                )
                st.caption(
                    "What-if analysis changes only the amount; other fields stay fixed. "
                    "A nonlinear peak near 100% may reflect synthetic PaySim rules."
                )

with batch_tab:
    st.markdown('<div class="section-heading">Batch transaction explorer</div>', unsafe_allow_html=True)
    st.write(
        "Score a demonstration CSV, inspect flagged rows, filter by model score, "
        "and export results. Limit: 10,000 rows and 10 MB per file."
    )
    st.caption("Required columns: " + ", ".join(REQUIRED))
    sample = pd.DataFrame(
        [
            {"type": "TRANSFER", "amount": 10000, "oldbalanceOrg": 10000, "oldbalanceDest": 0, "nameDest": "C_DEMO"},
            {"type": "PAYMENT", "amount": 100, "oldbalanceOrg": 15000, "oldbalanceDest": 0, "nameDest": "M_DEMO"},
            {"type": "CASH_OUT", "amount": 2000, "oldbalanceOrg": 12000, "oldbalanceDest": 1200, "nameDest": "C_EXAMPLE"},
        ]
    )
    st.download_button(
        "⬇ Download sample CSV",
        sample.to_csv(index=False),
        "paysim_demo_sample.csv",
        "text/csv",
    )
    uploaded = st.file_uploader(
        "Upload CSV containing simulated transactions", type=["csv"]
    )
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("The file exceeds the 10 MB demo limit.")
        else:
            try:
                transactions = pd.read_csv(uploaded, nrows=MAX_BATCH_ROWS + 1)
                if len(transactions) > MAX_BATCH_ROWS:
                    st.error("This demo accepts at most 10,000 rows.")
                elif len(transactions) == 0:
                    st.warning("The CSV is empty.")
                else:
                    results = score(transactions, bundle)
                    flagged = results["prediction"].eq("Flag for review")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Transactions analyzed", f"{len(results):,}")
                    b2.metric("Flagged for review", f"{int(flagged.sum()):,}")
                    b3.metric("Not flagged", f"{int((~flagged).sum()):,}")
                    st.markdown('<div class="section-heading">Model score distribution</div>', unsafe_allow_html=True)
                    categories = pd.cut(
                        results["model_score_uncalibrated"],
                        bins=np.linspace(0, 1.000001, 11),
                        include_lowest=True,
                    )
                    histogram = categories.value_counts(sort=False)
                    histogram.index = [
                        f"{i / 10:.1f}–{(i + 1) / 10:.1f}"
                        for i in range(10)
                    ]
                    st.bar_chart(histogram.rename("Transactions"), height=250)

                    f1, f2 = st.columns(2)
                    with f1:
                        view = st.radio(
                            "Show", ["All transactions", "Flagged only", "Not flagged only"],
                            horizontal=False,
                        )
                    with f2:
                        min_score = st.slider(
                            "Minimum uncalibrated model score",
                            min_value=0.0, max_value=1.0,
                            value=0.0, step=0.05,
                        )
                    filtered = results.loc[
                        results["model_score_uncalibrated"] >= min_score
                    ]
                    if view == "Flagged only":
                        filtered = filtered[filtered["prediction"] == "Flag for review"]
                    elif view == "Not flagged only":
                        filtered = filtered[filtered["prediction"] == "Not flagged"]

                    st.caption(f"Showing {len(filtered):,} of {len(results):,} scored rows")
                    st.dataframe(
                        filtered.head(500),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "⬇ Export filtered results",
                        filtered.to_csv(index=False).encode("utf-8"),
                        "paysim_filtered_scores.csv",
                        "text/csv",
                        type="primary",
                    )
            except Exception as exc:
                st.error(f"CSV could not be processed: {exc}")

with explain_tab:
    st.markdown('<div class="section-heading">What drives the model?</div>', unsafe_allow_html=True)
    st.write(
        "This final Random Forest uses six engineered features and excludes "
        "the simulated step and hour, account balances after the transaction, "
        "and the existing fraud-flag variable."
    )

    if hasattr(bundle["model"], "feature_importances_"):
        ordered_importance = (
            pd.DataFrame(
                {
                    "Feature": list(bundle["feature_columns"]),
                    "Impurity-based importance (%)": (
                        bundle["model"].feature_importances_ * 100
                    ),
                }
            )
            .sort_values("Impurity-based importance (%)", ascending=False)
        )
        st.markdown('<div class="section-heading">Tree-based feature importance</div>', unsafe_allow_html=True)
        st.bar_chart(
            ordered_importance.set_index("Feature"),
            horizontal=True,
            height=300,
        )
        st.caption(
            "Importance is based on impurity reduction within the trained trees. "
            "It is not a causal percentage or a real-world risk contribution."
        )

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-heading">Permutation importance</div>', unsafe_allow_html=True)
        st.dataframe(
            PERMUTATION,
            hide_index=True,
            use_container_width=True,
            column_config={
                "AP decrease": st.column_config.NumberColumn(
                    "Decrease in validation AP", format="%.6f"
                )
            },
        )
        st.caption(
            "Validation average precision fell by 0.879364 on average when "
            "the amount-to-balance ratio was shuffled (three repeats)."
        )
    with right:
        st.markdown('<div class="section-heading">Fraud hotspots</div>', unsafe_allow_html=True)
        st.dataframe(
            RATIO_HEATMAP,
            hide_index=True,
            use_container_width=True,
            column_config={
                "TRANSFER fraud %": st.column_config.ProgressColumn(
                    "TRANSFER fraud %", min_value=0, max_value=100,
                    format="%.2f%%",
                ),
                "CASH_OUT fraud %": st.column_config.ProgressColumn(
                    "CASH_OUT fraud %", min_value=0, max_value=100,
                    format="%.2f%%",
                ),
            },
        )
        st.caption(
            "Fraud rates by amount-to-sender-balance band, measured on "
            "the chronological validation set. The 99–101% band is unusually high."
        )

    with st.expander("Training, class imbalance & caveats", expanded=True):
        st.write(
            "Training used 305,315 sampled records: all 5,315 early-period "
            "fraud cases and 300,000 legitimate records. "
            "SMOTENC did not materially improve the validation classification "
            "result; the selected Random Forest uses no synthetic oversampling."
        )
        st.write(
            "The full PaySim dataset contains 6,362,620 simulated transactions. "
            "Our final test period contains 123,580 records. "
            "The 0.50 operating threshold was fixed for final testing. "
            "Sampled training data means scores should not be treated as "
            "calibrated probabilities of real-world fraud."
        )
        st.write(
            "Model selection and model explainability were evaluated using "
            "simulated transactions; independent real-world evaluation "
            "would be required before any operational use."
        )

st.markdown(
    '<div class="footer-note">PaySim Fraud Intelligence · '
    'Random Forest demonstration · Synthetic transactions only · '
    'No real-world banking decisions</div>',
    unsafe_allow_html=True,
)
