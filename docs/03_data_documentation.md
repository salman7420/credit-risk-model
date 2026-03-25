# Data Documentation

**Source**: Lending Club Accepted Loans (2007–2018)
**Raw size**: 2.26M rows, 153 columns
**Post-filter**: Closed loans only (Fully Paid + Charged Off), ~1.1M rows
**Features selected for modeling**: 35 numerical, 7 categorical (42 inputs → 62 after engineering)

---

## Feature Categories (Kept)

| Category | Count | Examples |
|----------|-------|---------|
| Loan characteristics | 2 | `loan_amnt`, `term` |
| Income and employment | 3 | `annual_inc`, `emp_length`, `verification_status` |
| Housing | 1 | `home_ownership` |
| Loan purpose | 1 | `purpose` |
| Geography | 1 | `addr_state` |
| Revolving credit | 5 | `revol_util`, `revol_bal`, `bc_util`, `all_util`, `total_rev_hi_lim` |
| Delinquencies | 4 | `delinq_2yrs`, `pub_rec`, `pub_rec_bankruptcies`, `inq_last_6mths` |
| Credit history | 7 | `earliest_cr_line`, `open_acc`, `mort_acc`, `mo_sin_old_rev_tl_op` |
| Account activity | 11 | `open_acc_6m`, `open_il_12m`, `num_rev_accts`, `num_actv_rev_tl` |

---

## Features Excluded

| Reason | Examples |
|--------|---------|
| Post-origination leakage | `total_pymnt`, `recoveries`, `last_pymnt_amnt` |
| LC risk outputs | `grade`, `sub_grade`, `int_rate`, `installment` |
| Identifiers / metadata | `id`, `member_id`, `url` |
| Free text (out of scope) | `desc`, `title` |
| Fine geography (fair lending risk) | `zip_code`, `msa` |
| Hardship / settlement data | All `hardship_*`, `settlement_*` columns |

---

## Target Variable

| Value | Label | Class |
|-------|-------|-------|
| Fully Paid | Non-default | 0 |
| Charged Off | Default | 1 |

Class distribution: approximately 80% non-default, 20% default.

---

## Data Splits

| Split | File |
|-------|------|
| Train | `data/splits/train.parquet` |
| Validation | `data/splits/val.parquet` |
| Test | `data/splits/test.parquet` |

Splits are stratified by target. No feature engineering is applied before splitting; all
transforms are fit on train only and applied to val/test.
