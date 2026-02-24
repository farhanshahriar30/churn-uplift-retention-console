# Retention Uplift Console 🎯  
**Churn + Uplift + Targeting Simulator (Causal Decision Dashboard)**

A deployed Streamlit dashboard that helps answer a practical retention question:

> **Who should we contact, and with which campaign, to get the most extra purchases under a fixed budget?**

Instead of only predicting “who is likely to churn/buy,” this project estimates **uplift** (the *causal* impact of an intervention) and turns it into a **targeting policy**.

---

## What this project does

You can think of this as a “campaign decision console”:

1) **Outcome / risk model**  
   Predicts each customer’s probability of purchasing.

2) **Uplift (causal impact) model**  
   Estimates how much sending a campaign changes purchase probability vs doing nothing:
   - Mens E-Mail vs No E-Mail  
   - Womens E-Mail vs No E-Mail  

3) **Targeting policy under constraints**  
   Given:
   - budget ($)
   - cost per email
   - max outreach capacity  
   …the app recommends:
   - **who to target**
   - **which campaign to send**
   - expected **extra purchases** and **cost per extra purchase**

---

## Dataset

This project uses the **MineThatData E-Mail Analytics** dataset (Hillstrom dataset).  
Download and details:  
https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

### Dataset summary (high level)
- ~64,000 customers
- Randomized assignment:
  - 1/3 received **Mens E-Mail**
  - 1/3 received **Womens E-Mail**
  - 1/3 received **No E-Mail** (control)
- Outcomes tracked in the following weeks:
  - `conversion` (0/1 purchase)
  - `spend` (amount spent)

We use **conversion** as the primary outcome.

---

## Tech stack

- **Python 3.12**
- **Streamlit** (interactive dashboard)
- **pandas / numpy** (data + analysis)
- **scikit-learn** (logistic regression, preprocessing, calibration)
- **XGBoost** (uplift T-learner outcome models)
- **joblib** (artifact saving/loading)
- **matplotlib** (optional plotting utilities)

---


---

## Quickstart (run locally)

This repo includes **precomputed artifacts** (`artifacts/`) and **processed splits** (`data/processed/`) so the dashboard can run immediately.

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the Streamlit app
```
streamlit run app/streamlit_app.py
```
## Full reproducibility (retrain from raw data)

If you want to reproduce everything from scratch:

### 1) Download the dataset

Get the CSV from the link above and place it here: data/raw/hillstrom.csv

### 2) Generate splits + dataset metadata
```
python -m src.data.preprocess
```

#### Outputs:

- data/processed/train.csv

- data/processed/val.csv

- data/processed/test.csv

- artifacts/dataset_profile.json

- artifacts/splits.json

### 3) Train the outcome model (with calibrated probabilities)
```
python -m src.models.churn
```
#### Outputs:

- artifacts/preprocess.joblib

- artifacts/churn_model.joblib

- artifacts/churn_model_calibrated.joblib

- artifacts/churn_metrics.json

### 4) Train the multi-treatment uplift model
```
python -m src.models.uplift.multitreatment
```

#### Output:

- artifacts/uplift_model.joblib

### 5) Run the targeting policy simulation

```
python -m src.eval.policy_metrics
```

#### Output:

- artifacts/policy_simulation_val.json

### 6) Run the app
```
streamlit run app/streamlit_app.py
```

## Results snapshot (from the randomized experiment)

Because assignment is randomized (RCT), we can directly estimate average campaign lift:

- **Mens E-Mail conversion:** ~1.253% vs **No E-Mail:** ~0.573% (**+0.68pp**)
- **Womens E-Mail conversion:** ~0.884% vs **No E-Mail:** ~0.573% (**+0.31pp**)

In the validation policy simulation (**5,000 outreach capacity, $200 budget at $0.02/email**):
- **Uplift targeting:** ~41 expected incremental conversions (CPI ≈ $2.43)  
- **Risk targeting:** ~38 (CPI ≈ $2.62)  
- **Random targeting:** ~36 (CPI ≈ $2.80)

> “Expected incremental conversions” are model-based estimates from predicted uplift, used for policy comparison under identical constraints.

## Dashboard pages (what each does)

### 1) Overview
Dataset summary + train/val/test split metadata + latest policy simulation snapshot.

### 2) Outcome Model (Conversion Probability)
Shows model ranking quality using **deciles**, **lift**, and top customers by predicted probability.

### 3) Uplift Modeling
Visualizes per-treatment uplift distributions and the recommended action mix.  
Includes **AUUC** (impact ranking score) to validate uplift ordering.

### 4) Targeting Simulator
The decision engine: set budget/cost/capacity and compare:
- **uplift targeting** vs **risk targeting** vs **random**  
Reports expected **extra purchases** and **cost per extra purchase**.

### 5) Model Monitoring
Quick stability checks: base rates, treatment mix, and feature summary drift scan.


## Notes on deployment
This project is deployed as a read-only demo:

- Training is disabled on Streamlit Cloud by default.

- The deployed app loads precomputed artifacts from artifacts/.

To enable training locally from the UI:
```
# PowerShell (Windows)
$env:ENABLE_LOCAL_TRAINING="1"
streamlit run app/streamlit_app.py
```
## License

MIT License (see LICENSE).
