# Credit risk, key concepts

---

## 1. Credit Risk in Banking

### 1.1 What Is Credit Risk?

Credit risk is the risk that a borrower will **fail to repay** a loan, causing a financial loss to the lender. Losses occur when customers miss payments, go severely delinquent, or are ultimately **charged off** (written off as uncollectable).

Banks typically decompose credit risk into three components:

- **PD (Probability of Default)** – likelihood the borrower will default within a given horizon (e.g., 12 months).
- **LGD (Loss Given Default)** – percentage of the exposure the bank expects to lose if a default occurs.
- **EAD (Exposure at Default)** – the outstanding amount owed at the time of default.

The expected loss on a loan is often summarized as:

\[
\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}
\]

In this project, the focus is on **PD modeling**; LGD and EAD are out of scope for simplicity but are important in real‑world capital and pricing models.[web:89][web:92]

---

## 2. Lending Club and the Use Case

### 2.1 What Is Lending Club?

Lending Club is/was a **peer‑to‑peer lending platform** that connected individual borrowers with individual or institutional investors instead of using a bank’s own balance sheet. Borrowers applied for personal loans (e.g., debt consolidation, home improvement), and investors funded these loans in exchange for interest payments.[web:41][web:46]

Between 2007 and 2018, Lending Club issued roughly **millions of personal loans** (the public dataset is ~2.26M rows). A non‑trivial share of these loans ended in **default/charge‑off**, meaning investors did not fully recover their principal.[web:41][web:46]

### 2.2 Our Modeling Goal

For this project, the target is to predict whether a loan will end as:

- **Good outcome**: *Fully Paid*
- **Bad outcome**: *Charged Off* (treated as default)

This is framed as a **binary classification** problem: estimate the probability of default **at application time**, using only information available before the loan is funded.

Business uses of the model:

- Support **approve / decline** decisions.
- Inform **risk‑based pricing** (higher rates for higher PD), even if the pricing step is manual in this project.[web:90][web:95]

---

## 3. Core Risk Drivers (Key Concepts)

### 3.1 FICO Score

**FICO** is a credit score (roughly 300–850) summarizing a consumer’s creditworthiness based on credit bureau data (payment history, amounts owed, length of history, new credit, and credit mix).[web:100][web:103]

Typical ranges:

- 750–850: Excellent – very low default rates.
- 700–749: Good.
- 650–699: Fair.
- < 650: Poor to very poor – elevated default risk.

In the Lending Club data, FICO appears as:

- `fico_range_low`
- `fico_range_high`

These represent the lower and upper bounds of the borrower’s FICO band at origination.

**Intuition**: Higher FICO → lower PD, all else equal.

---

### 3.2 Debt‑to‑Income Ratio (DTI)

**DTI** measures how much of a borrower’s gross monthly income goes to debt payments:

\[
\text{DTI} = \frac{\text{Total Monthly Debt Payments}}{\text{Gross Monthly Income}} \times 100
\]

Regulatory and industry thresholds:

- < 36%: generally considered comfortable.
- 36–43%: borderline; often requires closer review.
- > 43%: historically linked to higher mortgage risk under the **Qualified Mortgage (QM)** rules.[web:108][web:109]

Under the post‑crisis **QM/Ability‑to‑Repay** framework in the US, mortgages above ~43% DTI generally do not qualify for certain legal protections, so lenders became more conservative around that threshold.[web:108][web:109]

In the dataset:

- `dti` – Debt‑to‑income ratio provided by Lending Club.
- `annual_inc` – Annual income (useful to sanity‑check DTI and compute additional ratios).

**Intuition**: Very high DTI means the borrower has little slack; any income shock can lead to default.

---

### 3.3 Revolving Credit Utilization

**Revolving utilization** quantifies how much of available revolving credit (e.g., credit cards) is being used:

\[
\text{Utilization} = \frac{\text{Current Revolving Balance}}{\text{Total Revolving Credit Limit}} \times 100
\]

High utilization (e.g., > 50–70%) is often a sign of financial stress and is associated with higher default risk.[web:100][web:101]

In the dataset:

- `revol_bal` – Total revolving balance.
- `revol_util` – Utilization percentage.

---

### 3.4 Employment Length and Stability

Stable employment generally reduces credit risk:

- Short tenure (< 1 year) → higher risk (probation, income instability).
- Longer tenure (5+ years) → more stable income, lower risk.

In the dataset:

- `emp_length` – Employment length in string form (e.g., “< 1 year”, “10+ years”), which should be parsed into numeric years.
- `emp_title` – Free‑text job title (can be noisy but sometimes informative).

---

### 3.5 Loan Purpose

Borrowers specify what the loan is for. Different purposes have different risk profiles.[web:82]

Common categories (via `purpose`):

- `debt_consolidation` – Often medium risk; can be positive if it lowers overall interest costs.
- `credit_card` – Higher risk; adding more unsecured debt.
- `home_improvement` – Often lower risk; tied to a tangible asset.
- `small_business` – Higher risk; business income is volatile.
- `medical`, `moving`, `vacation` – Often associated with stress or discretionary spending.

EDA will test whether certain purposes systematically default more.

---

### 3.6 Delinquencies and Public Records

Past behavior is a strong predictor of future behavior.

Important variables:

- `delinq_2yrs` – Number of 30+ day delinquencies in the past 2 years.
- `mths_since_last_delinq` – Months since last delinquency (if present).
- `pub_rec` – Number of derogatory public records (e.g., bankruptcies, liens).
- `pub_rec_bankruptcies` – Bankruptcy count.

Higher counts or recent delinquencies/public records usually correspond to higher PD.[web:95]

---

### 3.7 Credit Inquiries

Each hard inquiry corresponds to a recent credit application. Many inquiries in a short window can indicate “credit‑seeking” behavior and elevated risk, except for controlled “rate shopping” in mortgages or auto lending.[web:109]

Key variables:

- `inq_last_6mths`
- `inq_last_12m`

---

## 4. Lending Club–Specific Features

### 4.1 Loan Amounts and Terms

- `loan_amnt` – Amount requested.
- `funded_amnt`, `funded_amnt_inv` – Amount actually funded by the platform/investors.
- `term` – Loan term in months (e.g., 36, 60).
- `installment` – Monthly payment amount.

Longer terms and higher loan amounts increase total exposure and often correlate with higher default risk at a given borrower profile.

---

### 4.2 Grades and Subgrades

Lending Club pre‑assigns a **credit grade**:

- `grade` – A–G (A = lowest risk, G = highest risk).
- `sub_grade` – A1–G5 (finer granularity).

These grades are derived from a proprietary scoring algorithm using FICO, DTI, delinquencies, income, and other factors. Grades map to interest rate bands via `int_rate`.[web:55][web:82]

From a modeling perspective:

- Grades can be used as features.
- It’s also valuable to compare our model’s discrimination power to the grade system.

---

### 4.3 Target Variable: Loan Status

The key outcome is:

- `loan_status`

In the raw data, this has many categories (e.g., “Fully Paid”, “Charged Off”, “Current”, “Late”, “In Grace Period”, etc.). For PD modeling:

- **Positive class (default)**: `loan_status = "Charged Off"` (and possibly certain severe late statuses).
- **Negative class (non‑default)**: `loan_status = "Fully Paid"`.
- **Excluded**: “Current”, “In Grace Period”, “Late” — these are not fully resolved and will be dropped from the training sample.

This binarization is standard practice in credit risk model development.[web:72][web:87]

---

## 5. Regulatory & Governance Context (High Level)

### 5.1 Fair Lending

Lenders in the US must comply with laws like:

- **Equal Credit Opportunity Act (ECOA)**
- **Fair Housing Act**

These limit the use of **protected characteristics** (race, gender, age, etc.) and certain proxies (e.g., detailed geolocation) in models.[web:92][web:95]

Implications for our project:

- We will avoid explicitly modeling on protected attributes (they are generally not present in the public dataset).
- We must ensure the model is explainable and decisions can be justified based on legitimate financial factors (FICO, DTI, delinquencies, etc.).

---

### 5.2 Ability‑to‑Repay and Qualified Mortgage (DTI 43%)

Post‑2008, the US introduced **Ability‑to‑Repay (ATR)** and **Qualified Mortgage (QM)** rules. For mortgages, loans above certain DTI thresholds (historically around 43%) often do not qualify for QM status, which removes certain legal protections for lenders.[web:108][web:109]

Even though Lending Club loans are unsecured personal loans, similar DTI thresholds influence bank risk appetite more broadly. Very high DTI values (e.g., > 45–50%) are typically considered high risk in consumer lending.

---

## 6. Typical High‑ and Low‑Risk Profiles

### 6.1 Low‑Risk (Illustrative)

- High FICO (e.g., 740+).
- DTI < 30%.
- Revolving utilization < 30%.
- Stable employment (5+ years).
- No recent delinquencies or public records.
- Few recent inquiries.
- Loan purpose: debt consolidation or home improvement.
- Home ownership: mortgage or own.

### 6.2 High‑Risk (Illustrative)

- FICO < 650.
- DTI > 40–45%.
- Revolving utilization > 70%.
- Employment < 1 year.
- Multiple delinquencies in past 2 years.
- Many recent inquiries.
- Loan purpose: small business, credit card, or discretionary categories.
- Income unverified or unstable.

These patterns will be tested and refined through EDA rather than assumed blindly.

---

## 7. How This Informs Our Modeling

This domain knowledge will guide:

- **Feature selection & engineering**  
  - Using FICO ranges, DTI, utilization, delinquencies, employment, grade, purpose, etc.
  - Creating ratios (e.g., loan‑to‑income, installment‑to‑income) and bins (e.g., FICO tiers, DTI buckets).

- **Leakage control**  
  - Excluding variables that contain *post‑origination* information (e.g., `total_pymnt`, `last_pymnt_d`, `recoveries`, `last_fico_range_*`).

- **EDA focus**  
  - Default rates by FICO tier, DTI bands, purpose, grade, term, utilization.

- **Interpretability**  
  - Being able to explain decisions in terms of standard risk drivers used by real credit underwriters.

This file serves as a living reference for the project and can be extended as new insights emerge during EDA and model development.
