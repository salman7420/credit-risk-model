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








## Additional Notes:
### Purpose
high_risk = ['small_business', 'renewable_energy', 'moving', 'medical']

medium_risk = ['house', 'debt_consolidation', 'other', 'vacation', 'major_purchase', 'home_improvement', 'educational', 'credit_card']

low_risk = [ 'car', 'wedding']