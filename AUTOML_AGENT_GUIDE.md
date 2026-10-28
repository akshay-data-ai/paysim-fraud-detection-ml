# CSV AutoML Agent (MVP)

Open the **AutoML Agent** page in the Streamlit sidebar.

## What it does

1. Upload a labeled **binary classification** CSV (maximum 10 MB, 50,000 rows, approximately 50 columns).
2. Select the target, positive label, optionally a time-order column, and columns to exclude.
3. The engine performs a train/validation/test split (60/20/20). It uses stratification for random splits or chronological order for time splits.
4. Preprocessing is inside scikit-learn pipelines; it is fitted only on training data.
5. Five candidate models are trained: Logistic Regression, Decision Tree, Random Forest, Extra Trees, and HistGradientBoosting.
6. The engine selects the winner by validation Average Precision and selects its operating threshold by validation F1.
7. A *single* final holdout test is produced, plus downloadable predictions and a trained joblib pipeline.

## Safeguards and limitations

This is a **guided AutoML baseline**, not a trustworthy zero-config agent. It does not support regression or multiclass classification. The user must confirm which target label means fraud and which features are available before a decision. Suggested leakage exclusions are not guaranteed exhaustive. Datetime, account identifiers, post-transaction balances, and label-derived fields require domain review.

No SMOTE is applied automatically; the candidate models instead use class weighting where supported. SMOTE should be a separate training-only experiment, never applied to validation or test periods.

The default random split is not suitable for every use case: when future performance matters, select a valid chronological column. Repeated customers and accounts may require group-aware splitting, which is not implemented here.

The very large PaySim dataset (~493 MB, 6.36M rows) exceeds the Streamlit demo limits; use Google Colab or a larger dedicated training environment. Dataset-wide automated training should be implemented as a separate worker with job queues and persistent storage if needed.

The winner is a **suggestion based on validation Average Precision**, not a guarantee. The model scores may not be calibrated probabilities. Preprocessing supports categorical data but has deliberate cardinality limits. A downloadable .joblib can execute code on deserialization: load only trusted local model files. Do not upload sensitive production customer data to the public demo.

For deployment, install `requirements.txt`, use Python 3.11 with scikit-learn 1.6.1, and select `app.py` as the main Streamlit application. This additional page is automatically discovered under `pages/` in the Streamlit sidebar.
