# Feature Glossary — Credit Risk ML Pipeline

This file is the single source of truth for every **active** feature fed into the model.
Use it as LLM context to explain predictions in plain English.

---

## 🔢 Raw Numerical Features

| Feature | What It Means | Why It Matters for Default |
|---|---|---|
| `loan_amnt` | Dollar amount the borrower requested. | Higher loan = more total exposure and repayment burden. |
| `annual_inc` | Borrower's self-reported yearly income in USD. | Higher income = more ability to repay; lowers default risk. |
| `dti` | Debt-to-income ratio — monthly debt payments ÷ monthly income. | Higher DTI = less income headroom; strong default predictor. |
| `delinq_2yrs` | Number of 30+ day late payments in the past 2 years. | Any recent delinquency signals repayment unreliability. |
| `inq_last_6mths` | Number of credit inquiries in the last 6 months. | More inquiries = actively seeking credit, often a stress signal. |
| `open_acc` | Total number of open credit lines. | Used to compute the share of new vs. old accounts. |
| `revol_util` | Revolving credit utilization — balance ÷ revolving limit (%). | Above 75% is a high-risk zone; strong default predictor. |
| `total_rev_hi_lim` | Total revolving credit limit across all accounts. | Combined with utilization to measure total credit exposure. |
| `bc_util` | Bankcard utilization — bankcard balance ÷ bankcard limit (%). | High bankcard utilization is a leading indicator of financial stress. |
| `percent_bc_gt_75` | % of bankcard accounts with utilization above 75%. | Measures how many bankcards are near their limit (breadth of stress). |
| `all_util` | Overall balance-to-limit ratio across all credit trades. | Broader than revol_util — captures installment + revolving together. |
| `tot_cur_bal` | Total current balance across all accounts. | Higher balance relative to income raises default risk. |
| `open_acc_6m` | Accounts opened in the last 6 months. | Rapid recent account opening signals aggressive credit-seeking. |
| `open_act_il` | Currently open active installment accounts. | Used to measure how much of installment portfolio is brand new. |
| `open_il_12m` | Installment accounts opened in the past 12 months. | High recent openings = limited payment history on new debt. |
| `open_il_24m` | Installment accounts opened in the past 24 months. | Wider window than open_il_12m for sustained credit-seeking behavior. |
| `il_util` | Installment account utilization — balance ÷ installment limit. | Complements revol_util by capturing installment-specific pressure. |
| `open_rv_12m` | Revolving accounts opened in the last 12 months. | Recent revolving credit acquisition signals financial stress. |
| `open_rv_24m` | Revolving accounts opened in the last 24 months. | Longer lookback version of open_rv_12m. |
| `inq_fi` | Number of personal finance inquiries. | Finance inquiries can indicate borrowers seeking payday/personal loans. |
| `inq_last_12m` | Credit inquiries in the last 12 months. | Wider window than inq_last_6mths; sustained credit-seeking behavior. |
| `acc_open_past_24mths` | Accounts opened in the last 24 months. | High count = rapid credit acquisition; strong default signal. |
| `pub_rec_bankruptcies` | Number of public record bankruptcies. | Direct indicator of past severe financial distress. |
| `mort_acc` | Number of mortgage accounts. | Mortgage holders tend to have lower unsecured default risk. |
| `mo_sin_old_rev_tl_op` | Months since the oldest revolving account was opened. | Longer history = more established credit behavior. |
| `num_rev_accts` | Total number of revolving accounts. | Used to compute account accumulation rate and indebted ratio. |
| `tot_hi_cred_lim` | Total high credit limit across all accounts. | High limits + high balances signal elevated leverage risk. |
| `total_bc_limit` | Total bankcard credit limit. | Used with revolving balance to measure bankcard-specific headroom. |
| `avg_cur_bal` | Average current balance per account. | High average balance relative to income increases default probability. |
| `num_op_rev_tl` | Number of open revolving trade lines. | More active revolving lines = more total revolving exposure. |
| `num_actv_rev_tl` | Revolving trade lines actively carrying a balance. | Active (not dormant) lines are riskier; combined with utilization. |
| `num_rev_tl_bal_gt_0` | Revolving trade lines with a balance greater than zero. | Fraction of revolving accounts actively in debt (used in indebt_rev_ratio). |

---

## 🚩 Binary Flag Features

| Feature | What It Means | Why It Matters for Default |
|---|---|---|
| `has_delinq_2yrs` | 1 if borrower had any 30+ day delinquency in past 2 years, else 0. | Any recent delinquency is a meaningful risk signal; presence/absence matters more than count. |
| `has_pub_rec` | 1 if borrower has any derogatory public record (bankruptcy, lien, judgment), else 0. | Public records are rare but highly predictive — any single occurrence is a strong red flag. |
| `high_revol_util` | 1 if revolving utilization ≥ 75%, else 0. | Captures the non-linear risk jump at the 75% threshold that a raw percentage may miss. |

---

## ⚙️ Engineered Numerical Features

| Feature | What It Means | Why It Matters for Default |
|---|---|---|
| `installment` | Estimated monthly payment = loan amount ÷ term in months. | Absolute monthly payment obligation; used as numerator in PTI. |
| `pti` | Payment-to-income ratio = (monthly installment ÷ monthly income) × 100. | What % of income the new loan consumes; direct affordability measure. |
| `revol_bal_to_income` | Total revolving balance ÷ annual income. | Normalizes revolving debt by income — captures relative burden. |
| `rev_accts_to_age` | Number of revolving accounts ÷ age of oldest revolving account (months). | How fast the borrower accumulated revolving accounts relative to history. |
| `indebt_rev_ratio` | Revolving accounts with balance > 0 ÷ total revolving accounts. | Fraction of revolving lines actively carrying debt (breadth of indebtedness). |
| `new_account_share` | Accounts opened in last 24 months ÷ total open accounts (clipped 0–1). | High share = credit portfolio is new and unproven; strong risk signal. |
| `bc_limit_util` | Total bankcard limit ÷ total revolving balance. | Low ratio = borrower is nearly maxed out on bankcards. |
| `il_recent_share` | Installment accounts opened in last 12 months ÷ total active installment accounts. | High share = most installment debt is brand new with no payment history. |
| `actv_rev_util` | Active revolving trade lines × revolving utilization. | Joint intensity: many active lines AND high utilization = compounded risk. |
| `bc_util_stress` | (Bankcard utilization × % bankcards above 75%) ÷ 100. | Compound bankcard stress — both depth and breadth of near-maxed cards. |
| `revol_stress_score` | Revolving utilization × revolving balance. | High utilization + large balance = compounded revolving pressure. |
| `il_recent_intensive` | Active installment accounts × installment accounts opened in last 12 months. | Large portfolio growing rapidly = aggressive installment credit expansion. |
| `recent_intensive` | Accounts opened in last 6 months² (squared). | Amplifies high-end values; captures non-linear risk from rapid credit-seeking. |
| `stress_util_income` | (All utilization ÷ 100) ÷ log(annual income). | Low income + high utilization together; far riskier than either alone. |
| `util_total_rev_stress_log` | log1p(all utilization × total revolving limit). | Total revolving dollar exposure under utilization pressure, log-scaled. |

---

## 🏷️ Engineered Categorical Features

| Feature | What It Means | Why It Matters for Default |
|---|---|---|
| `emp_length_stability` | Employment stability tier from emp_length: `unstable` (0–2 yrs) / `transitional` (2–5 yrs) / `stable` (5–10+ yrs). | Stable employment = consistent income = lower default risk. Ordinal encoded. |
| `purpose_bucket` | Loan purpose grouped by observed default rate: `high_risk` (small_business, moving, renewable_energy >22%) / `medium_risk` / `low_risk` (credit_card, car, wedding <18%). | Purpose is a strong proxy for borrower intent and financial situation. |
| `addr_region` | US state mapped to 4 regions: `northeast` / `southeast` / `midwest` / `west`. | Preserves geographic economic patterns at lower cardinality than 51 state codes. |
| `credit_maturity` | Credit history age tier: `new` (<5 yrs) / `moderate` (5–15 yrs) / `established` (15–30 yrs) / `veteran` (30+ yrs). | Longer history = lower default risk; non-linear thresholds matter more than exact years. |
| `home_ownership` | Cleaned housing status: `RENT` / `OWN` / `MORTGAGE`. | RENT = higher default risk; MORTGAGE = more stable; proxy for financial stability. |
| `term` | Loan term: `36 months` or `60 months`. | 60-month loans default more — longer term = more time for conditions to deteriorate. |
| `verification_status` | Income verification level: `Verified` / `Source Verified` / `Not Verified`. | Nuanced signal — verified borrowers sometimes show higher default (selection effect). |
| `revol_util_tier` | Revolving utilization bucketed: `low` (<30%) / `moderate` (30–60%) / `high` (60–80%) / `critical` (80–100%). | Default rates jump sharply at the critical tier (>80%). Ordinal encoded. |
| `grade` | LendingClub loan grade (A–G) assigned by their internal risk model. | Direct risk tier — Grade G borrowers default far more than Grade A. Strong predictor. |

---

## 🗑️ Dropped Features (not fed to model)

| Feature | Why Dropped |
|---|---|
| `pub_rec` | Replaced by `has_pub_rec` binary flag. |
| `revol_bal` | Replaced by `revol_bal_to_income` and `revol_stress_score`. |
| `emp_length` | Replaced by `emp_length_stability`. |
| `purpose` | Replaced by `purpose_bucket`. |
| `addr_state` | Replaced by `addr_region` (51 states → 4 regions). |
| `earliest_cr_line` | Replaced by `credit_maturity`. |
| `mths_since_last_delinq` | 50% missing; negligible signal. |
| `mths_since_last_record` | 83% missing; negligible signal. |
| `tot_coll_amt` | 51% missing; negligible signal. |
| `mths_since_last_major_derog` | 73% missing; flat default pattern. |
| `total_acc` | Negligible predictive power. |
| `total_bal_il` | No interpretable default pattern. |
| `mths_since_rcnt_il` | Captured by `il_recent_share` already. |
| `acc_now_delinq` | Negligible predictive power. |
| `collections_12_mths_ex_med` | Too sparse; negligible signal. |
| `max_bal_bc` | Negligible predictive power. |
| `total_cu_tl` | Negligible predictive power. |
| `pct_tl_nvr_dlq` | Weak signal. |
| `num_il_tl` | Negligible predictive power. |
| `mo_sin_old_il_acct` | Negligible predictive power. |
