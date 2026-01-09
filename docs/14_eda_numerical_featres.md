# EDA Strategy – Numerical Features

This document defines how numerical features will be analyzed and how final keep/drop/transform decisions will be recorded for the master dataset (`X_master`). The goal is to keep EDA **consistent, concise, and model‑agnostic**.

---

## 1. Per‑Feature EDA Checklist

For **each numerical column**:

1. **Basic Profile**
   - Compute: `min`, `max`, `mean`, `median`, `std`, `skew`, `n_unique`
   - Record:
     - `% missing`
     - Any impossible values (e.g., negative income, DTI > 300)

2. **Distribution & Outliers**
   - Plot histogram (full + zoomed on 1–99th percentile)
   - Plot boxplot
   - Decide and document:
     - Whether to **cap / Winsorize** extremes (e.g., at 99th percentile)
     - Whether to **log-transform** (for heavy right‑skewed positive data like income, balances)

3. **Relationship with Default**
   - Bin into 10–20 groups (quantiles or domain‑based bins).
   - For each bin:
     - Count of loans
     - Default rate (%)
   - Plot **default rate vs. bin**:
     - Note whether the pattern is:
       - Monotonic increasing / decreasing
       - U‑shaped
       - Flat / noisy

4. **Strength of Association (Optional but Recommended)**
   - Compute one of:
     - Point‑biserial / Pearson correlation with binary target
     - Information Value (IV) if already binned
   - Use IV rule of thumb:
     - `< 0.02`: negligible
     - `0.02–0.10`: weak
     - `0.10–0.30`: medium
     - `> 0.30`: strong

5. **Business & Leakage Check**
   - Confirm the feature is **available at application time** (no leakage).
   - Check if the default pattern is **business‑sensible** (e.g., higher DTI → higher PD)[109][112].

6. **Recommendation for `X_master`**
   - Based on steps 1–5, specify:
     - Keep as is
     - Keep + cap
     - Keep + log
     - Keep + engineered variant (ratio, age, etc.)
     - Create binned / WOE version (for later logistic models)
     - Drop (no signal / bad quality)

---

## 2. Recommendation Table (Numerical Features)

| Feature | Unique | Missing | Signal | Recommendation |
|---------|--------|---------|--------|----------------|
| **loan_amt** | 1553 | 0% | MODERATE ✅  | Keep as numeric; optionally cap at 99.5th percentile (~38k) to limit extreme tails, no log/WOE needed for tree models |
| **annual_inc** | 62989 | 0% | MODERATE ✅  | Keep as numeric; Heavy right tail → cap at ~320k. |
| **dti** | 6870 | 0.02% | STRONG  ✅  | Keep as numeric; Cap lower at 0 and higher at 50%. impute missing data with median|
| **payment‑to‑income ratio (PTI)** | N/A | N/A | N/A  | Create a new feature that calculate % of income the new loan payment would take|
| **delinq_2yrs** | 30 | 0% | WEAK–MODERATE ⚠️ | Cap at 5 (p99.5), keep as numeric count; add binary flag `has_delinq_2yrs` (=1 if >0) for extra signal; no further binning for tree models |
| **inq_last_6mths** | 9 | ~0% | MODERATE ✅ |  Cap at 5 (p99), impute 0 for missing, keep as numeric count; optional flag `high_inq_6m` (>=2) for extra signal | 
| **mths_since_last_delinq** | 163 | 50.46% | NEGLIGIBLE ❌  |  Drop this column | 
| **mths_since_last_record** | 124 | 82.99% | NEGLIGIBLE ❌  |  Drop this column | 
| **open_acc** | 84 | 0% |  WEAK–MODERATE ⚠️  |  Cap at 99.5 % that is 32. | 
| **pub_rec** | 37 | 0% |  WEAK–MODERATE ⚠️  |  Make new feature from this column called as `has_pub_rec`. if pub_rec exist then 1 otherwise 0. This feature performing better. | 
| **open_acc** | 84 | 0% |  WEAK–MODERATE ⚠️  |  Cap at 99.5 % that is 32. | 
| **revol_bal** | 82819 | 0% |  NEGLIGIBLE ❌   |  Make a new feature `revol_bal_to_income` which is giving better results. Drop revol_bal column after that | 
| **revol_util** | 82819 | 0.06% |  STRONG  ✅   |  Cap at 99.5% (upper limit). Median imputation for missing values. Create these new feature which is giving good results too: `high_revol_util`, `revol_util_tier`, `revol_stress_score` make these feature after capping + imputing missing values| 
| **total_acc** | 142 | 0% |  NEGLIGIBLE ❌  |  Drop the column, no predictive power! | 
| **total_acc** | 142 | 0% |  NEGLIGIBLE ❌  |  Drop the column, no predictive power! | 





























