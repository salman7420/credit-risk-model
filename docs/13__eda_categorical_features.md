# Categorical Features - EDA Summary

| Feature | Unique | Missing | Signal | Recommendation |
|---------|--------|---------|--------|----------------|
| **term** | 2 | 0% | STRONG ✅ | One-hot encode → `term_60_months` |
| **emp_title** | 300k+ | N/A | N/A ❌ | Drop (too many values) |
| **emp_length** | 11 | 5.8% | WEAK ⚠️ | Convert to numeric (0-10), median impute, drop original |
| **home_ownership** | 6 | 0% | MODERATE ✅ | Group rare (ANY/NONE/OTHER → OTHER), one-hot encode → 3 binary cols |
| **verification_status** | 3 | 0% | WEAK ⚠️ | One-hot encode → 2 binary cols (drop_first=True) |
| **purpose** | 14 | 0% | WEAK ⚠️ | Group into 3 risk buckets (high/medium/low), one-hot encode → 2 cols, drop original | 
| **addr_state** | 51 | 0% | NEGLIGIBLE ⚠️ | Group into 4 US regions, one-hot encode → 3 cols, drop original | 
| **earliest_cr_line** | 738 dates | 0% | WEAK ⚠️ | Group into 4 US regions, one-hot encode → Create TWO features: (1) `credit_history_years` (use application date(year) - extract year from earliest_cr_line) and (2) `credit_history_months` (take credit_history_years and * 12 ), (3)`credit_maturity` bins (veteran/established/moderate/new) → 3 binary cols. Drop original. Let model pick best during training. | 
| **application_type** | 2 | 0% | WEAK ❌ | FILTER: Keep only 'Individual' (98.2%), drop column + all joint-specific features (annual_inc_joint, dti_joint, sec_app_*, etc.) |
| **verification_status_joint** | 3 | 98.7% | DROP ❌ | Joint application column so will be removed |  
| **sec_app_earliest_cr_line** | N/A | 98.7% | DROP ❌ | Joint application column so will be removed | 


Maybe use WOE on binning features and use that number! 
## Additional Notes:
### Purpose
high_risk = ['small_business', 'renewable_energy', 'moving', 'medical']
medium_risk = ['house', 'debt_consolidation', 'other', 'vacation', 'major_purchase', 'home_improvement', 'educational', 'credit_card']
low_risk = [ 'car', 'wedding']

# earliest_cr_line and addr_state
2.1. Core transformation: “credit history length”
Industry‑standard practice is:
Parse to datetime
Convert strings like "1st August 1958" or "Aug-1958" into a proper datetime column.
In many public Lending Club–style datasets, people convert earliest_cr_line into a datetime and then compute months since earliest credit line as of a fixed reference date (usually the portfolio snapshot or application date).
​

Compute credit history age relative to application date
The key variable is not the calendar date itself, but how long the applicant has had credit at the time of application:

Handle anomalies
Sometimes parsing errors or odd formats yield negative values (e.g., date parsed in the future). A simple fix (as done in practice) is to cap negatives at the max valid value observed.
Optionally cap extreme long histories (e.g., > 40 years) because risk differences past a certain age are minimal.
Optional: convert months to years, maybe log‑scale

2.2. Binning / WOE for logistic or scorecard‑style models
In scorecard‑style PD models, credit history length is usually binned into risk buckets and transformed using Weight of Evidence (WOE):
Decide coarse bins by domain intuition and EDA, e.g.:
< 1 year
1–3 years
3–5 years
5–10 years
10–20 years
> 20 years
Compute PD (or WOE) per bin and check if risk is roughly monotonic with credit history age.
Use the WOE‑encoded variable in a logistic model; this lets you capture non‑linearity without manually adding splines or polynomial terms.
Many production credit models use exactly this combination: continuous credit history length → categorical bins → WOE encoding → logistic regression.
​

2.3. Additional engineered features from earliest_cr_line

Depending on what else you have, you can extract more signal:
Relative age vs. current account age
If you have open_acc (number of open accounts) and/or total_acc, you can compute:
avg_account_age = (application_dt - earliest_cr_line_dt) / total_acc (rough approximation of how frequently new accounts are opened).
new_to_credit_indicator = 1 if yrs_since_earliest_cr_line < 1 else 0.
Bucket by vintage of origination environment
Sometimes borrowers who first entered the credit system during particular macro regimes (e.g., pre‑crisis vs post‑crisis) have different risk. You can encode:
credit_vintage = decade(earliest_cr_line_dt) or pre_2008 vs 2008_2012 vs 2013_2019 vs 2020_plus.
This captures cohort effects (who got their first credit under what macro/underwriting standards).

2.4. Why it probably looks weak now
Typical failure modes:
Treating earliest_cr_line as:
A string → effectively a 738‑level categorical with no ordering.
A year only → compresses a lot of variation and may create artificial non‑monotonic patterns (e.g., 1999 vs 2000).
A raw ordinal (e.g., timestamp integer) in a linear model with no transformation → the model expects a linear effect, but risk vs history is clearly non‑linear.
Once you convert to an age metric (months_since_earliest), you should:
Plot PD by binned credit_history_length and check if short histories are meaningfully riskier.
For tree‑based models, use the raw continuous feature; they will learn non‑linear splits automatically.
For GLM/Logit, use binning or non‑linear transforms (WOE, splines).
You will almost always see credit history length emerge as a non‑trivial predictor once represented this way; consumer credit scoring models explicitly include it as a key factor.
​