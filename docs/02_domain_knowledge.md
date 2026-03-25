# Domain Knowledge: Credit Risk

---

## 1. Credit Risk Fundamentals

Credit risk is the probability that a borrower fails to repay a loan. Banks decompose it into:

- **PD** (Probability of Default) — will this borrower default?
- **LGD** (Loss Given Default) — what fraction of the loan is lost if they do?
- **EAD** (Exposure at Default) — how much is outstanding at time of default?

Expected Loss = PD x LGD x EAD

This project focuses exclusively on PD modeling.

---

## 2. Key Risk Drivers

| Feature | Why It Matters |
|---------|----------------|
| FICO score | Summarizes credit history; higher = lower default risk |
| DTI (debt-to-income) | High DTI leaves little buffer for income shocks |
| Revolving utilization | High utilization signals financial stress |
| Delinquencies | Past missed payments strongly predict future defaults |
| Employment stability | Longer tenure = more stable income |
| Loan term | 60-month loans default at roughly 2x the rate of 36-month loans |
| Loan purpose | Small business and credit card loans carry higher risk than home improvement |
| Credit inquiries | Many recent inquiries signal active credit-seeking behavior |

---

## 3. Target Variable

`loan_status` is binarized as:
- **0 (non-default)**: Fully Paid
- **1 (default)**: Charged Off
- **Excluded**: Current, In Grace Period, Late — these are unresolved and dropped from training

---

## 4. Leakage Control

The following are excluded because they are LC's own risk outputs or post-origination data:

- `grade`, `sub_grade`, `int_rate` — derived from LC's scoring system, not borrower inputs
- `installment` — calculated from rate, term, and amount
- `total_pymnt`, `recoveries`, `last_pymnt_*` — post-disbursement payment behavior

---

## 5. Regulatory Context

US lending regulations (ECOA, Fair Housing Act) restrict the use of protected characteristics in
credit decisions. This model uses only financial and behavioral inputs (FICO, DTI, utilization,
delinquencies, employment), which are standard in credit underwriting and can be explained to
regulators and borrowers.
