# Dataset description, columns

This document summarizes the **features we will use** and **features we will exclude** for modeling, based on the updated Lending Club data dictionary (`LCDataDictionary_updated.xlsx`, `browseNotes` sheet).

---

## 1. Summary

- Total columns in data dictionary (`browseNotes`): **120**
- Columns marked **Use = Yes** → **95 features** (candidate model inputs)
- Columns marked **Use = No** → **25 features** (excluded)
- Target variable: `loan_status` (defined in project charter, not in `browseNotes`)

We only use information **available at application time** and avoid:
- Post-origination info (leakage)
- LC’s own risk outputs (grades, expected default rate)
- Pure identifiers and metadata
- High–fair-lending-risk geographic detail (ZIP, MSA)

---

## 2. Features to Use (95) – By Category

These are all columns with `Use? = Yes` in `browseNotes`, grouped into logical categories.

### 2.1 Credit Scores (4)

- `ficoRangeLow` – Lower boundary of borrower’s FICO at origination  
- `ficoRangeHigh` – Upper boundary of borrower’s FICO at origination  
- `sec_app_fico_range_low` – FICO low for secondary applicant  
- `sec_app_fico_range_high` – FICO high for secondary applicant  

### 2.2 Debt Metrics (2)

- `dti` – Debt-to-income ratio for primary borrower  
- `dti_joint` – Debt-to-income ratio for joint applications  

### 2.3 Revolving Credit (13)

- `all_util` – Balance to credit limit on all trades  
- `bcOpenToBuy` – Total open-to-buy on revolving bankcards  
- `bcUtil` – Bankcard utilization (balance / limit)  
- `il_util` – Installment utilization  
- `mthsSinceRecentRevolDelinq` – Months since most recent revolving delinquency  
- `num_actv_bc_tl` – Number of active bankcard accounts  
- `num_bc_sats` – Number of satisfactory bankcard accounts  
- `num_bc_tl` – Number of bankcard accounts  
- `revolBal` – Total revolving balance  
- `revolUtil` – Revolving utilization rate  
- `total_il_high_credit_limit` – Total installment high credit/limit  
- `revol_bal_joint` – Revolving balance for co-borrowers (joint)  
- `sec_app_revol_util` – Revolving utilization for secondary applicant  

### 2.4 Delinquencies & Derogatories (14)

- `accNowDelinq` – Number of accounts currently delinquent  
- `chargeoff_within_12_mths` – Charge-offs in last 12 months  
- `collections_12_mths_ex_med` – Collections in last 12 months (excl. medical)  
- `delinq2Yrs` – 30+ day delinquencies in past 2 years  
- `delinqAmnt` – Amount currently past due  
- `mths_since_last_major_derog` – Months since most recent 90+ day derogatory  
- `mthsSinceLastDelinq` – Months since last delinquency  
- `mthsSinceRecentLoanDelinq` – Months since most recent loan delinquency  
- `pubRec` – Number of derogatory public records  
- `pub_rec_bankruptcies` – Bankruptcy count  
- `tax_liens` – Number of tax liens  
- `sec_app_chargeoff_within_12_mths` – Charge-offs (secondary applicant, last 12m)  
- `sec_app_collections_12_mths_ex_med` – Collections (secondary applicant, last 12m, excl. medical)  
- `sec_app_mths_since_last_major_derog` – Months since last major derogatory (secondary applicant)  

### 2.5 Credit Inquiries (5)

- `inq_fi` – Number of personal finance inquiries  
- `inq_last_12m` – Credit inquiries in past 12 months  
- `inqLast6Mths` – Inquiries in past 6 months (excl. auto/mortgage)  
- `mthsSinceMostRecentInq` – Months since most recent inquiry  
- `sec_app_inq_last_6mths` – Inquiries in last 6 months (secondary applicant)  

### 2.6 Credit History & Account Counts (36)

- `accOpenPast24Mths` – Trades opened in past 24 months  
- `earliestCrLine` – Date of earliest credit line  
- `mo_sin_old_rev_tl_op` – Months since oldest revolving account opened  
- `mo_sin_rcnt_rev_tl_op` – Months since most recent revolving account opened  
- `mo_sin_rcnt_tl` – Months since most recent account opened  
- `mortAcc` – Number of mortgage accounts  
- `mths_since_oldest_il_open` – Months since oldest installment account opened  
- `mths_since_rcnt_il` – Months since most recent installment account opened  
- `num_actv_rev_tl` – Number of active revolving trades  
- `num_il_tl` – Number of installment accounts  
- `num_op_rev_tl` – Number of open revolving accounts  
- `num_rev_accts` – Number of revolving accounts  
- `num_rev_tl_bal_gt_0` – Revolving trades with balance >0  
- `num_sats` – Number of satisfactory accounts  
- `num_tl_120dpd_2m` – Accounts currently 120+ DPD (last 2 months)  
- `num_tl_30dpd` – Accounts currently 30+ DPD  
- `num_tl_90g_dpd_24m` – Accounts 90+ DPD in last 24 months  
- `num_tl_op_past_12m` – Accounts opened in past 12 months  
- `open_acc_6m` – Open trades in last 6 months  
- `open_il_12m` – Installment accounts opened in past 12 months  
- `open_il_24m` – Installment accounts opened in past 24 months  
- `open_act_il` – Active installment trades  
- `open_rv_12m` – Revolving trades opened in past 12 months  
- `open_rv_24m` – Revolving trades opened in past 24 months  
- `openAcc` – Number of open credit lines  
- `total_bal_il` – Total balance of all installment accounts  
- `total_cu_tl` – Number of finance trades  
- `total_rev_hi_lim` – Total revolving high credit/limit  
- `totalAcc` – Total number of credit lines  
- `totalBalExMort` – Total balance excluding mortgage  
- `totalBcLimit` – Total bankcard high credit/limit  
- `sec_app_earliest_cr_line` – Secondary applicant earliest credit line  
- `sec_app_mort_acc` – Secondary applicant mortgage accounts  
- `sec_app_open_acc` – Secondary applicant open trades  
- `sec_app_open_act_il` – Secondary applicant active installment trades  
- `sec_app_num_rev_accts` – Secondary applicant revolving accounts  

### 2.7 Income & Employment (5)

- `annualInc` – Borrower’s annual income  
- `annual_inc_joint` – Joint annual income (co-borrowers)  
- `emp_title` – Borrower’s job title (free text)  
- `empLength` – Employment length in years (string, to be parsed)  
- `verified_status_joint` – Joint income verification status  

### 2.8 Loan Characteristics (2)

- `loanAmnt` – Loan amount requested  
- `term` – Loan term (36 or 60 months as string)  

### 2.9 Loan Purpose (1)

- `purpose` – Borrower-declared purpose of the loan  

### 2.10 Demographics & Geography (2)

- `addrState` – State of the borrower (2-letter code)  
- `homeOwnership` – Home ownership status (`RENT`, `OWN`, `MORTGAGE`, etc.)  

### 2.11 Other / Miscellaneous (11)

- `application_type` – Individual vs joint application  
- `avg_cur_bal` – Average current balance across all accounts  
- `isIncV` – Income verification status (primary borrower)  
- `max_bal_bc` – Maximum current balance on bankcards  
- `mthsSinceLastRecord` – Months since most recent public record  
- `mthsSinceRecentBc` – Months since most recent bankcard account opened  
- `pct_tl_nvr_dlq` – Percent of trades never delinquent  
- `percentBcGt75` – Percent of bankcard accounts >75% utilization  
- `tot_coll_amt` – Total collection amounts ever owed  
- `tot_cur_bal` – Total current balance across all accounts  
- `tot_hi_cred_lim` – Total high credit/credit limit  

---

## 3. Features to Exclude (25) – By Reason

These columns have `Use? = No` in `browseNotes`. We’re explicitly excluding them **before coding**.

### 3.1 Post-Origination / Process Timing / LC Internals

- `acceptD` – Date borrower accepted the offer  
- `creditPullD` – Date LC pulled credit  
- `expD` – Listing expiration date  
- `listD` – Date application was listed  
- `reviewStatusD` – Date LC reviewed the application  
- `ils_exp_d` – Whole-loan platform expiration date  

**Reason**: Process timing / platform mechanics; not borrower risk drivers and can unintentionally leak process behaviour.

### 3.2 LC Risk Outputs / Target-Leakage Features

- `grade` – LC-assigned grade (A–G)  
- `subGrade` – LC-assigned subgrade (A1–G5)  
- `intRate` – Interest rate assigned by LC  
- `installment` – Monthly installment (function of rate, term, amount)  
- `effective_int_rate` – Effective interest rate net of expected uncollected interest  
- `expDefaultRate` – LC’s expected default rate for the loan  
- `reviewStatus` – APPROVED / NOT_APPROVED  
- `initialListStatus` – Initial listing status (W/F)  

**Reason**: These are **outputs** of LC’s own underwriting and/or incorporate knowledge of default. Using them would create circular logic (model learns LC’s decision instead of borrower risk).

### 3.3 Identifiers / Metadata

- `id` – Loan listing ID  
- `memberId` – Borrower ID  
- `url` – URL of the listing page  

**Reason**: Pure identifiers or metadata; no predictive content.

### 3.4 Text / Free-Form Fields (High Noise, Out of Scope)

- `desc` – Free-text loan description  
- `title` – Loan title provided by borrower  

**Reason**: NLP on free text is out of scope for this iteration; text is noisy and partially redundant with `purpose`. We **do** keep `emp_title` as a potential engineered signal.

### 3.5 Detailed Geography (Fair Lending Risk / Redundant)

- `zip_code` – First 3 digits of ZIP code  
- `msa` – Metropolitan Statistical Area  

**Reason**: Extra-fine geography can act as a proxy for protected classes (fair lending risk). We retain only `addrState` at a coarser level.

### 3.6 Investor / Platform-Specific Fields

- `serviceFeeRate` – Service fee rate paid by the investor  
- `disbursement_method` – CASH vs DIRECT_PAY  

**Reason**: Investor/platform economics; not borrower credit risk.

### 3.7 Potentially Redundant Credit-Performance Feature

- `num_accts_ever_120_pd` – Accounts ever 120+ days past due  

**Reason**: Overlaps with other delinquency counts (e.g., `num_tl_90g_dpd_24m`, `num_tl_120dpd_2m`); excluded to avoid redundancy and complexity in first version.

---

## 4. Target Variable (For Clarity)

Although not in the `browseNotes` sheet, our **target** from the main Lending Club loan file will be:

- Column: `loan_status`  
- Classes for modeling:
  - **Good (0)**: `Fully Paid`
  - **Bad (1)**: `Charged Off`
- Rows with statuses like `Current`, `In Grace Period`, `Late (xx-xx days)`, etc. will be **dropped** from the training set.

---

## 5. Next Step (Coding Plan)

With this feature inventory fixed, our next coding steps will be:

1. Load the raw Lending Club file (CSV).
2. Filter rows to `loan_status` in `{Fully Paid, Charged Off}`.
3. Select only the **95 “Use = Yes” columns + `loan_status`**.
4. Start **data profiling** (missingness, types, ranges) using this agreed feature set.

This markdown is our **data contract**: we should not add/remove features without updating this file.

## 6. Summary of Data Collected
### Summary by Decision:
### ✅ 91 KEEP columns - Application-time credit data

1. Loan Request Details (2)
2. Employment & Income (4)
3. Housing (1)
4. Loan Purpose (1)
5. Geography (1)
6. Credit Metrics (1)
7. Delinquencies & Public Records (8)
8. Credit History & Accounts (10)
9. Revolving Credit (8)
10. Installment & Other Accounts (11)
11. Account Activity Metrics (28)
12. Joint Application Fields (10)

### ❌ REMOVE columns - Leakage + Identifiers

1. Identifiers (3): id, member_id, url
2. LC Risk Outputs (6): grade, sub_grade, int_rate, installment, etc.
3. Post-Origination Payment Data (19): All total_pymnt, out_prncp, last_pymnt_d, etc.
4. Hardship Data (15): All hardship_* columns
5. Settlement Data (7): All settlement_* columns
6. Text Fields (2): desc, title
7. Fine Geography (1): zip_code
8. Platform-Specific (1): disbursement_method
9. Redundant (1): num_accts_ever_120_pd

| Category             | Count           |
| -------------------- | --------------- |
| Loan Details         | 2               |
| Employment & Income  | 4               |
| Housing              | 1               |
| Purpose              | 1               |
| Geography            | 1               |
| Credit Metrics       | 1               |
| Delinquencies        | 8               |
| Credit History       | 7               |
| Revolving Credit     | 8               |
| Installment Accounts | 11              |
| Account Activity     | 31              |
| Joint App Fields     | 15              |
| Target               | 1 (loan_status) |
| TOTAL                | 91              |
