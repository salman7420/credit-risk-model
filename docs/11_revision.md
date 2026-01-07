# What Lending Club Did (2007-2018):
- Cut out the bank as the middleman
- Connected borrowers (people who need money) with investors (people who want to earn interest)
Think of it like "Airbnb for money"
- Airbnb: Connect travelers with homeowners
- Lending Club: Connect borrowers with investors

## How It Worked (Step-by-Step)

1. You Apply for a Loan

Go to LendingClub.com
Fill out: "I need $15,000 for debt consolidation"
Provide: Income, credit score (FICO), employment

2. Lending Club Assigns a Risk Grade

They look at your profile
Assign a grade: A (safest) to G (riskiest)
Set an interest rate: Grade A = 6%, Grade G = 28%

3. Investors Fund Your Loan

Your loan gets listed on the platform
300 different investors each invest $50
Example: Retiree Bob invests $50, hoping to earn 8% interest

4. You Make Monthly Payments

You pay back $500/month for 36 months
Lending Club distributes your payment to all 300 investors
Everyone earns interest, IF you pay back

5. What Happens if You Don't Pay?

Lending Club tries to collect for a while
After 120+ days late → "Charged Off" (declared a loss)
Investors lose their money (Bob loses his $50)

# The Problem:
Out of 2.26 million loans, about 15-20% were charged off (defaulted). That means investors lost millions of dollars.

# Our Goal:
Build a model to predict BEFORE approving the loan: "Will this person pay back or default?"

# If Our Model Works:
Reject high-risk borrowers → Save investors from losses
Or charge them higher interest to compensate for risk

# What is Credit Risk? (Simple Explanation)
The Core Concept

Credit Risk = The risk that someone won't pay you back
Everyday Example (Non-Banking)

Scenario 1: Your Friend Borrows Money
Friend A: Has a job, always paid you back before, saves money
Low Credit Risk → You'd lend them $100 without worry
Friend B: Unemployed, borrowed from 5 other friends and didn't pay back, gambles a lot
High Credit Risk → You probably wouldn't lend them money
That's exactly what banks/investors do, just with math and data!
Credit Risk in Banking (3 Questions)
When a bank considers a loan, they ask:

1. Will they default? → PD (Probability of Default)
Our project focuses on THIS

2. If they default, how much will we lose? → LGD (Loss Given Default)
Example: Lent $10,000, recovered $3,000 → Lost $7,000 (70% LGD)

3. How much did they owe when they defaulted? → EAD (Exposure at Default)
Example: $10,000 loan, paid back $4,000, defaulted on remaining $6,000
Real-World Example: Two Loan Applications

Factor	Borrower A	            Borrower B	    Who's Riskier?
FICO Score	780 (Excellent)	    620 (Poor)	    B is riskier
Annual Income	$90,000	$       35,000	        B is riskier
Existing Debts	$500/month	    $1,800/month	B is riskier
Loan Request	$10,000	        $10,000	        Same
Employment	8 years at Google	6 months at small startup	B is riskier
Conclusion: Borrower B has HIGH credit risk (might not pay back).

Bank's Decision:
Borrower A: Approved at 6% interest (low risk)
Borrower B: Either rejected OR approved at 22% interest (high risk = high reward needed)


# Accuracy
The Problem: Class Imbalance
Let's say we have 1,000 loan applications:
800 people will pay back (80%)
200 people will default (20%)
The "Dumb Model" Scenario
Imagine I create this lazy model:

python
def predict_loan(borrower):
    return "Will Pay Back"  # Always predict "Paid"
This model NEVER thinks anyone will default. It just says "approve everyone."

Let's calculate accuracy:
Total predictions: 1,000
Correct predictions: 800 (all the people who actually paid)
Incorrect predictions: 200 (the defaulters we missed)
Accuracy = 800/1,000 = 80% ✅

Wait, 80% accuracy sounds great, right? WRONG!
Why This Model is Useless

Business Impact:
We approved ALL 200 defaulters → Lost money on every single one
The model didn't help at all! We could have flipped a coin and done the same
Bank loses millions because we couldn't identify ANY risky borrowers

The Lesson:
In imbalanced datasets, accuracy is misleading because a model can score high by just predicting the majority class.
This is why banks NEVER use accuracy for credit risk models.

1. Understanding the Confusion Matrix First

Before we talk about metrics, you need to understand the Confusion Matrix - it's the foundation:

Actual vs Predicted:

                                   Predicted: Will Pay	                        Predicted: Will Default
Actually Paid	                   True Negative (TN) = 700	❌                  False Positive (FP) = 100
Actually Defaulted:                False Negative (FN) = 50	✅                  True Positive (TP) = 150 

In plain English:
True Positive (TP): We said "will default" and they DID default ✅ (Good catch!)
True Negative (TN): We said "will pay" and they DID pay ✅ (Correct approval)
False Positive (FP): We said "will default" but they actually paid ❌ (False alarm - rejected good customer)
False Negative (FN): We said "will pay" but they defaulted ❌ (Missed a bad loan)

2. Recall (Sensitivity) - "How many defaults did we catch?"

Formula:
Recall = TP / (TP + FN) 
       = True Positives / All Actual Defaulters

Using our example:
Recall = 150 / (150 + 50) = 150/200 = 0.75 = 75%
What this means:
"Out of 200 people who actually defaulted, we correctly identified 150 of them (75%)"

Real-World Analogy:
Recall is like a security metal detector at an airport:
High Recall (90%) = Catches 90% of weapons (but beeps a lot on belts/coins too)
Low Recall (50%) = Only catches 50% of weapons (dangerous!)

In Credit Risk:
High Recall = We catch most defaulters (but might reject some good customers too)
Low Recall = We miss many defaulters (they get loans and don't pay back)

3. Precision - "When we predict default, how often are we right?"

Formula:
Precision = TP / (TP + FP)
          = True Positives / All Predicted Defaults
Using our example:
Precision = 150 / (150 + 100) = 150/250 = 0.60 = 60%

What this means:
"When we say someone will default, we're correct 60% of the time"

Real-World Analogy:
Precision is like a spam filter:
High Precision (95%) = When it marks something as spam, it's almost always spam
Low Precision (40%) = It marks lots of emails as spam, but many are actually important

In Credit Risk:
High Precision = When we reject someone, they probably would have defaulted
Low Precision = We're rejecting many good customers unnecessarily (lost business!)

🎪 ROC-AUC: The "Best of Both Worlds" Metric
What is ROC?

ROC = Receiver Operating Characteristic Curve
It's a graph that shows the tradeoff between:
True Positive Rate (TPR) = Recall = TP/(TP+FN)
False Positive Rate (FPR) = FP/(FP+TN)

The curve shows performance at EVERY possible threshold:

Threshold 0.9: Very strict (reject if >90% probability of default)
  → High Precision, Low Recall

Threshold 0.5: Moderate (reject if >50% probability)
  → Balanced

Threshold 0.1: Very loose (reject if >10% probability)
  → Low Precision, High Recall

What is AUC?
AUC = Area Under the ROC Curve

Simple Explanation:
It's a single number (0 to 1) that summarizes the curve.​​

What the numbers mean:

AUC = 0.50: Model is guessing randomly (coin flip) - USELESS
AUC = 0.70: Good model - Industry minimum for credit risk​
AUC = 0.80: Very good model
AUC = 0.90+: Excellent model (rare in real-world credit data)

Intuitive Meaning:
"If I pick one random defaulter and one random non-defaulter, AUC is the probability my model assigns a higher risk score to the defaulter."

Example:

AUC = 0.75 means: "75% of the time, the model correctly ranks a defaulter as riskier than a non-defaulter"
Why Banks Use ROC-AUC

Advantages:

✅ Threshold-independent: Works across all decision thresholds
✅ Handles imbalance: Not fooled by 80/20 class split
✅ Single number: Easy to compare models (Model A: 0.73, Model B: 0.78 → Pick B)
✅ Industry standard: All banks use it, so models are comparable​

Real-World Benchmarks:​
AUC 0.70-0.80: Acceptable (most internal bank models)
AUC 0.80-0.90: Strong (competitive advantage)
AUC 0.90+: Exceptional (research-level, hard to achieve with messy real-world data)

📋 Summary Table: When to Use Which Metric
Metric	Use When	Credit Risk Example
Accuracy	❌ NEVER use for imbalanced data	Misleading (80% accuracy by approving everyone)
Recall	Need to catch most defaults	"Catch 80% of defaults even if we reject some good customers"
Precision	Need to avoid false alarms	"When we reject someone, be 90% sure they'd default"
ROC-AUC	✅ PRIMARY metric for credit risk	"Overall model quality across all thresholds"
F1-Score	Balance Recall and Precision	Harmonic mean: 2×(Precision×Recall)/(Precision+Recall)



📚 Concept 1: Hypothesis Testing Framework
The Core Idea:

You have a claim (hypothesis) about your data, and you want to know: "Is this claim true, or just random chance?"
The Two Hypotheses:

Null Hypothesis (H₀):
The "nothing special is happening" hypothesis.
Example: "Home ownership and default are independent - they have no relationship"
This assumes: Renters, owners, mortgage holders all default at the same rate (20%)

Alternative Hypothesis (H₁):
The "something interesting is happening" hypothesis.
Example: "Home ownership and default are associated - they DO have a relationship"
This means: Renters default at different rate than owners (e.g., 23% vs 15%)

The Goal:
Use data to decide: Should we reject H₀ (something is real) or fail to reject H₀ (not enough evidence)?

Simple Analogy:
Courtroom Trial:
H₀ = Defendant is innocent (default assumption)
H₁ = Defendant is guilty

Evidence (data) = Witnesses, fingerprints
Verdict = "Guilty beyond reasonable doubt" (reject H₀) OR "Not guilty" (fail to reject H₀)
Statistical testing:
H₀ = Feature has no relationship with default
H₁ = Feature IS related to default
Evidence = Your 1.3M loan records
Verdict = p-value tells you how strong the evidence is

📚 Concept 2: P-Value (The Evidence Measure)
What It Is:
The probability that your observed pattern could occur by random chance alone if H₀ were true.
​
Intuitive Explanation:
Imagine you flip a coin 100 times:

Expected: 50 heads, 50 tails (if coin is fair = H₀)
You get: 60 heads, 40 tails
Question: "Is this coin unfair (H₁), or just bad luck (H₀)?"
P-value answers: "If the coin were fair, what's the chance of getting 60+ heads by luck?"
If p = 0.30 (30%) → "Pretty common, probably just luck" → Don't reject H₀
If p = 0.01 (1%) → "Very rare by luck, coin is probably unfair" → Reject H₀

Credit Risk Example:

Your data:
RENT: 23% default rate
MORTGAGE: 15% default rate
Overall: 20% default rate

H₀: "Home ownership doesn't matter - everyone defaults at 20%"
Observed: RENT is 3% higher, MORTGAGE is 5% lower
P-value answers: "If home ownership truly didn't matter, what's the chance of seeing these differences just by random sampling?"
p = 0.40 → "Could easily happen by chance" → Home ownership probably doesn't matter
p = 0.001 → "Extremely unlikely by chance" → Home ownership DOES affect default risk

Interpretation Thresholds:

P-Value	Meaning	Decision
p < 0.001	Extremely strong evidence against H₀	✅ Feature is definitely related to default
p < 0.01	Strong evidence	✅ Feature is very likely related
p < 0.05	Moderate evidence (standard cutoff)	✅ Feature is statistically significant
p ≥ 0.05	Weak/no evidence	❌ Can't conclude relationship exists
Common Misconception:

❌ "p = 0.03 means there's 3% chance H₀ is true"
✅ "p = 0.03 means IF H₀ were true, there's 3% chance of seeing our data"

Subtle but important! P-value assumes H₀ is true and asks "How surprising is my data?"

📚 Concept 3: Sample Size Effect (Why Large Data Changes Things)
The Problem with Big Data:
With 1.3M loans, everything becomes statistically significant - even meaningless patterns.

Example:
Scenario A: Small Sample (1,000 loans)
RENT: 22% default (220/1000)
MORTGAGE: 20% default (200/1000)
Difference: 2 percentage points
Chi-square test:
p-value = 0.30 → Not significant
Conclusion: "Could be random variation, not enough evidence"

Scenario B: Large Sample (1,000,000 loans)
RENT: 20.2% default (202,000/1M)
MORTGAGE: 20.0% default (200,000/1M)
Difference: 0.2 percentage points (tiny!)
Chi-square test:
p-value = 0.001 → Highly significant!
Conclusion: "Statistically significant relationship detected"

The Paradox:
Scenario A: Bigger difference (2%) but NOT significant
Scenario B: Tiny difference (0.2%) but HIGHLY significant

Why? With 1M samples, you can detect tiny real effects that are statistically meaningful but practically useless for prediction.
​
The Solution: Effect Size

Don't just ask "Is there a relationship?" (p-value)
Ask "How STRONG is the relationship?" (Cramér's V, effect size)
This is why we need both chi-square AND Cramér's V!
​

📚 Concept 4: Chi-Square Test (The Relationship Detector)
What It Does:
Tests if two categorical variables are independent or associated.
In credit risk: "Does this category (RENT/OWN/MORTGAGE) affect default rate?"

How It Works (Step-by-Step):

Step 1: Observed Counts (What You Actually See)
Default (1)	No Default (0)	Total
RENT	110,000	370,000	480,000
MORTGAGE	78,000	442,000	520,000
OWN	34,000	246,000	280,000
Total	222,000	1,058,000	1,280,000

Observed default rates:
RENT: 110K/480K = 22.9%
MORTGAGE: 78K/520K = 15.0%
OWN: 34K/280K = 12.1%

Step 2: Expected Counts (If H₀ Were True)
If home ownership didn't matter, everyone would default at the overall rate: 222K/1.28M = 17.3%

Expected:
RENT: 480,000 × 0.173 = 83,040 defaults
MORTGAGE: 520,000 × 0.173 = 89,960 defaults
OWN: 280,000 × 0.173 = 48,440 defaults

Step 3: Calculate Differences
Category	Observed Defaults	Expected Defaults	Difference
RENT	110,000	83,040	+26,960 (more than expected)
MORTGAGE	78,000	89,960	-11,960 (fewer than expected)
OWN	34,000	48,440	-14,440 (fewer than expected)

Chi-square statistic:
χ² = Σ [(Observed - Expected)² / Expected]

For RENT defaults: (110,000 - 83,040)² / 83,040 = 8,731
For MORTGAGE defaults: (78,000 - 89,960)² / 89,960 = 1,589
... (repeat for all cells)

χ² = 8,731 + 1,589 + ... = ~15,000 (hypothetical)
Step 4: Determine Significance
Large χ² value → Big differences between observed and expected → Relationship exists!
Small χ² value → Small differences → Could be random → No relationship
Convert χ² to p-value using chi-square distribution table
χ² = 15,000 with degrees of freedom = (3-1)×(2-1) = 2
p-value < 0.0001

Conclusion: Reject H₀ → Home ownership IS associated with default

Real-World Interpretation:
✅ p < 0.05 means: "Renters, mortgage holders, and owners have genuinely different default rates - it's not random"
❌ p ≥ 0.05 means: "The differences we see could easily happen by chance - no evidence of real relationship"
Chi-square tells you: "Is there a relationship?" (Yes/No)
Cramér's V tells you: "How STRONG is that relationship?" (0 to 1 scale)
​

The Scale:
Cramér's V	Interpretation	Example
0.00	No association	Like flipping two unrelated coins
0.05	Negligible	State explains 0.25% of default variation
0.10	Weak but detectable	Purpose explains 1% of default variation
0.20	Moderate	Home ownership explains 4% of variation
0.40	Strong	Employment length explains 16% of variation
0.70+	Very strong	(Rare in credit data)

Formula:
Cramér's V = √(χ² / (n × min(rows-1, cols-1)))
Where:
- χ² = Chi-square statistic
- n = Total sample size
- min(rows-1, cols-1) = Smaller of (categories-1) or (outcomes-1)
For binary target (2 outcomes), this simplifies to measure how much variance in default the feature explains.
​

Real Example:

Feature A: Home Ownership
Chi-square: p < 0.001 → Significant!
Cramér's V: 0.08
Meaning: "Yes, there's a relationship, but it only explains ~0.6% of default variation - weak effect"

Feature B: Recent Delinquency
Chi-square: p < 0.001 → Significant!
Cramér's V: 0.35
Meaning: "Strong relationship - explains ~12% of default variation - very important feature!"

Why Both Features Are "Significant" But One Is More Valuable:
With 1.3M loans:
Both features pass p < 0.05 (chi-square)
But V = 0.35 feature is 16x more powerful than V = 0.08 feature (0.35²/0.08² ≈ 16)
Priority: Model Recent Delinquency more heavily
​
| Test         | Question                    | Output        | Use Case                  |
| ------------ | --------------------------- | ------------- | ------------------------- |
| Frequency    | How balanced?               | Counts/%      | Check for rare categories |
| Default Rate | Which categories are risky? | % by category | Direct risk measure       |
| Chi-Square   | Is relationship real?       | p-value       | Gate: Keep or drop        |
| Cramér's V   | How strong?                 | 0-1 scale     | Rank importance           |