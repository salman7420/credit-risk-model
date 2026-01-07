# Key findings from exploration
# Preliminary Data Analysis Report
---

## Executive Summary

This preliminary analysis examines 1.3M loan records from Lending Club to understand data structure, quality, and readiness for credit risk modeling. Key findings:

- ✅ **Dataset Size:** 1,303,638 loans with 91 features (sufficient for ML modeling)
- ⚠️ **Missing Data:** 29.81% overall - requires strategic imputation (expected pattern for credit data)
- ✅ **Target Balance:** 20.07% default rate - moderate class imbalance, manageable with standard techniques
- ✅ **Feature Mix:** 79 numeric + 12 categorical features - appropriate for tree-based and linear models

**Readiness Assessment:** Dataset is suitable for modeling after handling missing values and feature engineering.

---

## 1. Dataset Structure

### 1.1 Dimensions

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Rows** | 1,303,638 | ✅ Large sample size supports complex models and train/test split |
| **Total Features** | 91 | ✅ Rich feature set for predictive modeling |
| **Target Variable** | `loan_status` (3 categories) | ✅ Clear binary outcome after mapping |
| **Memory Footprint** | ~140 MB (Parquet) | ✅ Manageable for in-memory processing |

### 1.2 Key Observations

- **Scale:** 1.3M loans provide statistically robust training set for ensemble models
- **Feature Density:** 91 features is optimal range (not too sparse, not curse of dimensionality)
- **Processing:** Dataset fits in memory, enabling rapid iteration during EDA and modeling

### 1.3 Implications for Modeling

✅ **Sample size sufficient for:**
- 70/15/15 train/validation/test split with >900K training samples
- K-fold cross-validation (5-10 folds) without statistical power concerns
- Rare event modeling (20% default rate = 260K+ default cases)

---

## 2. Data Type Distribution

### 2.1 Type Breakdown

| Data Type | Count | Percentage | Examples |
|-----------|-------|------------|----------|
| **Float64** | 77 | 84.6% | `loan_amnt`, `annual_inc`, `dti`, `revol_util` |
| **Object** | 12 | 13.2% | `term`, `home_ownership`, `purpose`, `emp_title` |
| **Int64** | 2 | 2.2% | `delinq_2yrs`, `open_acc` |

### 2.2 Analysis

**Numeric Dominance (86.8%):**
- Credit data is inherently numeric (balances, ratios, counts)
- Favorable for regression-based models and gradient boosting
- Minimal encoding complexity compared to text-heavy datasets

**Categorical Features (13.2%):**
- Manageable number of categorical variables (12 columns)
- Require encoding strategies based on cardinality (addressed in Section 6)
- Key risk segmentation variables (purpose, home ownership, employment)

**Integer vs Float:**
- Only 2 pure integers (likely count variables: delinquencies, accounts)
- Most counts stored as float64 (pandas default) - acceptable for modeling


## 3. Missing Value Analysis

### 3.1 Overall Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Missing Values** | 35,359,852 | Substantial but expected for credit bureau data |
| **Missing Percentage** | 29.81% | ⚠️ Requires careful imputation strategy |
| **Columns with Missing** | 74 (81.3%) | Most features have some missing data |
| **Complete Columns** | 17 (18.7%) | Core features are complete |

### 3.2 Assessment: Why 30% Missing is EXPECTED in Credit Data

**This is NOT a data quality problem.** Credit bureau data has three patterns of missingness:

#### Pattern 1: "Never Happened" Features (MNAR - Missing Not at Random)
**Examples:**
- `mths_since_last_delinq`: Missing = borrower never had delinquency (GOOD signal)
- `mths_since_last_record`: Missing = no public records (GOOD signal)
- `sec_app_*` fields: Missing = Individual application (not Joint)

**Imputation Strategy:** 
- Create binary flag: `has_delinq` (0/1)
- Impute missing with large value (e.g., 999) indicating "never"
- **This missingness is predictive!**

#### Pattern 2: Thin Credit File (MAR - Missing at Random)
**Examples:**
- New borrowers lack long credit history
- Missing values correlate with `earliest_cr_line` (recent)
- More common in younger applicants

**Imputation Strategy:**
- Median/mean imputation for recent borrowers
- Group by credit age bins for context-aware imputation

#### Pattern 3: Data Gaps (MCAR - Missing Completely at Random)
**Examples:**
- Reporting errors from credit bureaus
- System issues during data collection

**Imputation Strategy:**
- Simple imputation (median for numeric, mode for categorical)

4. Target Variable Distribution
4.1 Loan Status Breakdown

Status	            Count	        Percentage	        Binary Target
Fully Paid	        1,041,952	    79.93%	            0 (Good)
Charged Off	        261,655	        20.07%	            1 (Bad)
Default		        31              0.002%	            1 (Bad)

4.2 Class Imbalance Assessment

Default Rate: 20.07%
Industry Benchmark Comparison:
Consumer unsecured loans: 15-25% default rate (industry standard)
Lending Club typical: 18-22%
Our data: 20.07% → Within expected range ✅

Imbalance Level: MODERATE
Imbalance Ratio: 4:1 (79% vs 21%)
Not severe (severe would be >10:1 or <1% minority class)
Standard ML techniques can handle this without extreme measures

4.3 Implications for Modeling

Class Imbalance Handling Strategy
✅ Recommended Approaches:
Stratified Sampling (Required)
Maintain 80/20 split in train/validation/test
Use stratify=y in train_test_split()

## 5. Numeric Features Analysis

### 5.1 Overview

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Numeric Features** | 79 | ✅ Rich feature set for predictive modeling |
| **Complete Features** | 77 | Most numeric features have data |
| **Constant Columns** | 0 | ✅ No zero-variance features detected |
| **Low-Variance Features** | 5 | ⚠️ Require special handling (rare events) |

### 5.2 Key Feature Statistics

#### Core Loan & Borrower Metrics

| Feature | Mean | Median | Std Dev | Min | Max | Observations |
|---------|------|--------|---------|-----|-----|--------------|
| **loan_amnt** | $14,417 | $12,000 | $8,700 | $500 | $40,000 | ✅ Right-skewed, median < mean |
| **annual_inc** | $76,159 | $65,000 | $70,048 | $0 | $9,500,000+ | ⚠️ Extreme outliers detected (max $9.5M) |
| **dti** | 18.26% | 17.61% | 10.94% | -1.0% | 999%+ | ⚠️ Invalid values: negative & >100% |

**Key Insights:**

1. **Loan Amount Distribution:**
   - Median ($12K) < Mean ($14.4K) → Right-skewed distribution
   - IQR: $8K - $20K (middle 50% of loans)
   - **Implication:** Consider log transformation for linear models to normalize distribution

2. **Income Outliers:**
   - Max value $9.5M suggests data quality issues or extreme cases
   - 99th percentile likely ~$200K-$300K (need verification in univariate analysis)
   - **Action Required:** Cap at 99th percentile to prevent outlier dominance in models

3. **DTI Data Quality Issues:**
   - Negative values (-1.0%) are impossible → data error or missing indicator
   - Values >100% suggest debt exceeds income (valid but rare)
   - Max 999%+ likely data error or placeholder for missing
   - **Action Required:** Clean invalid values, cap at reasonable threshold (60-80%)

#### Credit Behavior Metrics

| Feature | Mean | Key Finding | Model Impact |
|---------|------|-------------|--------------|
| **delinq_2yrs** | 0.32 | 75th percentile = 0 (most have no delinquencies) | Strong default predictor (rare event) |
| **inq_last_6mths** | 0.66 | Median = 0, 75th = 1 (low inquiry activity) | Multiple inquiries = credit shopping risk |
| **revol_util** | ~54% | High utilization common | >80% is critical default predictor |

**Distribution Patterns:**
- Most credit metrics are **right-skewed** (median < mean)
- Many features have **floor at 0** (counts, balances)
- **Implication:** Non-normal distributions favor tree-based models over linear regression

### 5.3 Low-Variance Features Analysis

**Detected: 5 features with >99% same value**

| Feature | % Same Value | Most Common | Count Non-Zero | Recommendation |
|---------|--------------|-------------|----------------|----------------|
| **acc_now_delinq** | 99.53% | 0 | ~6,100 (0.47%) | ✅ KEEP - Rare but critical |
| **chargeoff_within_12_mths** | 99.19% | 0 | ~10,500 (0.81%) | ✅ KEEP - Recent chargeoff = strong signal |
| **delinq_amnt** | 99.63% | 0 | ~4,800 (0.37%) | ✅ KEEP - Amount delinquent indicates severity |
| **num_tl_120dpd_2m** | 99.92% | 0 | ~1,000 (0.08%) | ⚠️ ANALYZE - Extremely rare, check predictive power |
| **num_tl_30dpd** | 99.68% | 0 | ~4,200 (0.32%) | ✅ KEEP - Current delinquency status |

#### Why Low-Variance Features Are VALUABLE in Credit Models

**Standard ML Advice:** "Remove features with >95% same value"  
**Credit Risk Reality:** These are **rare default indicators** - the most predictive features!

**Example Analysis Needed:**
```python
# For each low-variance feature, check default rate:
# Non-zero group (the 0.5%):
default_rate_with_delinq = df[df['acc_now_delinq'] > 0]['target'].mean()  
# Expected: 70-90% default rate

# Zero group (the 99.5%):
default_rate_no_delinq = df[df['acc_now_delinq'] == 0]['target'].mean()  
# Expected: 18-20% default rate

# If 70% vs 20% → STRONG PREDICTOR despite low variance!

6. Categorical Features Analysis
6.1 Overview

Metric	Value	Assessment
Total Categorical Features	12	✅ Manageable number for encoding
Low Cardinality (<10)	8	✅ Direct one-hot encoding
Medium Cardinality (10-50)	3	⚠️ May need rare category grouping
High Cardinality (>50)	1	⚠️ Requires special encoding strategy
6.2 Cardinality Breakdown

Low Cardinality Features (<10 unique values)

Feature	Unique Values	Example Values	Encoding Strategy
term	2	"36 months", "60 months"	One-hot (creates 1 feature)
home_ownership	4-6	RENT, OWN, MORTGAGE, OTHER	One-hot (creates 3-5 features)
verification_status	3	Verified, Not Verified, Source Verified	One-hot (creates 2 features)
loan_status	3	Fully Paid, Charged Off, Default	Target variable (binary encode)
application_type	2	Individual, Joint App	One-hot (creates 1 feature)
verification_status_joint	3-4	Same as above	One-hot
Total one-hot columns: ~15-20 (acceptable for modeling)

Action:

✅ Direct one-hot encoding

✅ Check for typos/inconsistencies ("RENT" vs "Rent" vs "rent")

✅ Verify no "Unknown" or placeholder categories

Medium Cardinality Features (10-50 unique values)

Feature	Unique Values	Top Categories	Strategy
purpose	~14	debt_consolidation (46%), credit_card (19%), home_improvement (12%)	Group rare (<1000 samples) → One-hot
addr_state	51	CA, TX, NY, FL (high volume states)	Regional grouping OR target encoding
emp_length	11	"10+ years", "2 years", "< 1 year"	Ordinal encoding (0→10)
Concerns:

purpose: Likely has 2-3 rare categories (<0.1% of data) → Group into "Other"

addr_state: 51 one-hot columns is high → Consider grouping by region (West, South, Midwest, Northeast)

emp_length: Has natural order → Ordinal better than one-hot