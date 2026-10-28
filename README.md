# PaySim Fraud Detection — 6.36 Million Transactions

Machine-learning fraud detection on the **synthetic PaySim dataset** using Random Forest. Includes an interactive [Streamlit](https://streamlit.io/) app for individual transaction screening, batch CSV scoring, and model explainability.

## Results on the chronological holdout

| Metric | Random Forest |
| --- | ---: |
| Test transactions | 123,580 |
| Labeled fraud cases | 1,654 |
| Precision | 99.88% |
| Recall | 99.94% |
| F1 | 99.91% |
| Average precision (PR-AUC-style summary) | 0.999988 |
| Detected fraud / missed fraud | 1,653 / 1 |
| False positives | 2 |

**These results are specific to PaySim's synthetic fraud patterns, not evidence of real-world banking accuracy.** The classifier was trained on 5,315 fraud rows and 300,000 sampled legitimate rows. Scores are **uncalibrated**; the threshold is 0.50.

### Experiments

We compared Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost; experimented with SMOTENC; checked model performance without the simulated `step` and `hour_in_cycle` fields; examined the strong full-balance transaction pattern; and calculated both impurity-based and permutation importance. See the finished experiment notebook and CSV charts in the project Google Drive for the methodology.

## Setup

This repository contains the **Streamlit source**, but the trained binary model is **not yet included**.

1. In Google Drive, find `Fraud_Detection/Models/fraud_detection_random_forest.joblib`.
2. Verify the file came from your own trusted Colab training run. A joblib file can execute code when loaded; never load unknown artifacts.
3. Create a folder `models` in this repository and place the file at `models/fraud_detection_random_forest.joblib`. Model weight files in public GitHub repositories are publicly accessible. Check GitHub's upload limits before trying to upload it.
4. Check the training environment's scikit-learn version in Colab using `import sklearn; print(sklearn.__version__)`, and match the version in `requirements.txt` if needed.
5. Install and run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy through Streamlit Community Cloud, select this repo, branch `main`, and the app entry point `app.py`. Wait until the model binary is present; the app will otherwise show an explanation instead of predictions.

## Required inference fields

`type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, `nameDest`.

The app mirrors six training features: logarithm of amount, logarithm of sender and destination opening balances, amount-to-sender-balance ratio (clipped at 100,000, zero when sender opening balance is zero), destination merchant indicator, and transaction-type integer encoding. The exact feature order comes from the saved model bundle.

The raw 493 MB training dataset is **not required** to serve predictions and should not be uploaded to the GitHub repository. The application is a prototype for simulated transactions only, not an automated financial-decision system.
