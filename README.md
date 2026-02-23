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

## Repo structure

