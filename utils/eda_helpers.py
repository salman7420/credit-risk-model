# utils/eda_helpers.py
"""
Categorical Feature Analysis Helper Functions
Credit Risk EDA - Lending Club Dataset

Functions:
1. analyze_frequency_distribution() - Count and percentage analysis
2. calculate_default_rates() - Default rates by category
3. perform_chi_square_test() - Statistical significance test
4. calculate_cramers_v() - Effect size measure
5. interpret_results() - Final recommendation based on test results

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from typing import Dict, List, Optional
from loguru import logger

# Import plotting utilities
from .plotting_config import (
    COLORS, FIGSIZE, TARGET_COLORS,
    format_axis_labels, add_value_labels, 
    add_reference_line, get_figsize
)

# ============================================================
# SECTION 1: FREQUENCY ANALYSIS
# ============================================================

def analyze_frequency_distribution(
    df: pd.DataFrame,
    column: str,
    rare_threshold: float = 1.0,
    plot: bool = True
) -> Dict:
    """
    Analyze frequency distribution of categorical feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Categorical column name
    rare_threshold : float, default=1.0
        Percentage threshold for rare categories (e.g., 1.0 = 1%)
    plot : bool, default=True
        Whether to display visualizations
        
    Returns
    -------
    dict
        Dictionary with counts, percentages, rare categories info
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FREQUENCY DISTRIBUTION: {column}")
    logger.info(f"{'='*60}")
    
    # Input validation
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Calculate counts and percentages
    counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100
    
    # Identify rare categories
    rare_categories = percentages[percentages < rare_threshold].index.tolist()
    
    # Check if quasi-constant (one category >99%)
    is_quasi_constant = percentages.max() > 99.0
    
    # Print results
    print("\nAbsolute Counts:")
    print(counts)
    print("\nPercentage Distribution:")
    print(percentages.round(2))
    
    print(f"\n{'─'*60}")
    print(f"Summary:")
    print(f"  Total categories: {len(counts)}")
    print(f"  Total observations: {len(df):,}")
    
    if rare_categories:
        print(f"\n⚠️  Rare categories (<{rare_threshold}%):")
        for cat in rare_categories:
            print(f"    - {cat}: {counts[cat]:,} ({percentages[cat]:.2f}%)")
    else:
        print(f"\n✅ No rare categories detected (all >{rare_threshold}%)")
    
    if is_quasi_constant:
        print(f"\n⚠️  WARNING: Quasi-constant feature!")
        print(f"    '{counts.index[0]}' represents {percentages.iloc[0]:.2f}% of data")
        print(f"    Consider dropping this feature (low variance)")
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['large'])
        
        # Plot 1: Count bar chart
        counts.plot(kind='bar', ax=axes[0], color=COLORS['primary_blue'], alpha=0.8)
        format_axis_labels(axes[0], 
                          xlabel=column, 
                          ylabel='Count', 
                          title=f'Frequency Distribution: {column}')
        add_value_labels(axes[0], fmt='{:.0f}')
        
        # Plot 2: Percentage bar chart
        percentages.plot(kind='bar', ax=axes[1], color=COLORS['primary_orange'], alpha=0.8)
        format_axis_labels(axes[1], 
                          xlabel=column, 
                          ylabel='Percentage (%)', 
                          title=f'Percentage Distribution: {column}')
        add_value_labels(axes[1], fmt='{:.1f}%')
        
        # Add rare threshold line on percentage plot
        if rare_threshold > 0:
            axes[1].axhline(y=rare_threshold, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7, label=f'Rare threshold ({rare_threshold}%)')
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'counts': counts,
        'percentages': percentages,
        'rare_categories': rare_categories,
        'total_categories': len(counts),
        'is_quasi_constant': is_quasi_constant
    }


# ============================================================
# SECTION 2: DEFAULT RATE ANALYSIS (COMPLETE FIXED VERSION)
# ============================================================

def calculate_default_rates(
    df: pd.DataFrame,
    column: str,
    target: str = 'target',
    plot: bool = True
) -> pd.DataFrame:
    """
    Calculate default rates by category with robust type handling.
    
    This function automatically converts the target column to numeric type,
    handles any data type issues, and provides comprehensive analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Categorical feature column to analyze
    target : str, default='target'
        Binary target column name (will be converted to 0/1 if needed)
    plot : bool, default=True
        Whether to generate visualization plots
        
    Returns
    -------
    pd.DataFrame
        DataFrame with default rate statistics by category, sorted by default rate
        
    Raises
    ------
    ValueError
        If column or target not found in dataframe
        
    Examples
    --------
    >>> results = calculate_default_rates(df, 'term', plot=True)
    >>> results = calculate_default_rates(df, 'grade', target='is_default', plot=False)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DEFAULT RATE ANALYSIS: {column}")
    logger.info(f"{'='*60}")
    
    # ============================================================
    # VALIDATION
    # ============================================================
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    print("\nCalculating default rates...")
    print(f"Analyzing column: {column}")
    print(f"Target column: {target}")
    
    # ============================================================
    # TARGET CONVERSION (CRITICAL FIX)
    # ============================================================
    # Convert target to numeric - handles string "0"/"1" or already numeric
    target_series = pd.to_numeric(df[target], errors='coerce')
    
    # Check for conversion issues
    null_count = target_series.isnull().sum()
    if null_count > 0:
        print(f"⚠️  Warning: {null_count} target values couldn't be converted to numeric")
        print(f"   Original target dtype: {df[target].dtype}")
        print(f"   Unique original values: {df[target].unique()}")
        print(f"   Dropping {null_count} rows with invalid target values")
    
    # Create working dataframe with numeric target
    temp_df = df[[column]].copy()
    temp_df['_target_numeric'] = target_series
    
    # Remove rows where target conversion failed
    temp_df = temp_df.dropna(subset=['_target_numeric'])
    
    print(f"Working with {len(temp_df):,} valid rows")
    
    # ============================================================
    # CALCULATE DEFAULT RATES
    # ============================================================
    results = []
    
    for category in sorted(temp_df[column].unique()):
        # Filter to this category
        mask = temp_df[column] == category
        subset = temp_df[mask]
        
        # Calculate statistics
        total_loans = len(subset)
        defaults = int(subset['_target_numeric'].sum())
        no_defaults = total_loans - defaults
        
        # Calculate rates
        default_rate = (defaults / total_loans * 100) if total_loans > 0 else 0.0
        pct_of_total = (total_loans / len(temp_df) * 100)
        
        results.append({
            column: category,
            'total_loans': total_loans,
            'defaults': defaults,
            'no_defaults': no_defaults,
            'default_rate_%': round(default_rate, 2),
            'pct_of_total': round(pct_of_total, 2)
        })
    
    # Create results dataframe
    default_analysis = pd.DataFrame(results)
    default_analysis = default_analysis.sort_values('default_rate_%', ascending=False).reset_index(drop=True)
    
    # ============================================================
    # CALCULATE SUMMARY STATISTICS
    # ============================================================
    overall_default_rate = temp_df['_target_numeric'].mean() * 100
    min_rate = default_analysis['default_rate_%'].min()
    max_rate = default_analysis['default_rate_%'].max()
    rate_range = max_rate - min_rate
    
    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print("\nDefault Rate by Category:")
    print(default_analysis[[column, 'total_loans', 'defaults', 'default_rate_%']].to_string(index=False))
    
    print(f"\n{'─'*60}")
    print(f"Key Statistics:")
    print(f"  Overall default rate:     {overall_default_rate:.2f}%")
    print(f"  Minimum default rate:     {min_rate:.2f}%")
    print(f"  Maximum default rate:     {max_rate:.2f}%")
    print(f"  Range:                    {rate_range:.2f} percentage points")
    
    if min_rate > 0:
        risk_ratio = max_rate / min_rate
        print(f"  Risk ratio (max/min):     {risk_ratio:.2f}x")
    
    # ============================================================
    # IDENTIFY HIGH/LOW RISK CATEGORIES
    # ============================================================
    high_risk = default_analysis[default_analysis['default_rate_%'] > 30]
    low_risk = default_analysis[default_analysis['default_rate_%'] < 15]
    
    if len(high_risk) > 0:
        print(f"\n⚠️  High-risk categories (>30% default):")
        for _, row in high_risk.iterrows():
            print(f"    - {row[column]}: {row['default_rate_%']:.2f}% ({row['total_loans']:,} loans)")
    
    if len(low_risk) > 0:
        print(f"\n✅ Low-risk categories (<15% default):")
        for _, row in low_risk.iterrows():
            print(f"    - {row[column]}: {row['default_rate_%']:.2f}% ({row['total_loans']:,} loans)")
    
    # ============================================================
    # PREDICTIVE POWER INTERPRETATION
    # ============================================================
    print(f"\n{'─'*60}")
    if rate_range > 10:
        print("✅ STRONG SIGNAL: Large variation in default rates (>10 points)")
        print("   This feature has good predictive power for modeling")
    elif rate_range > 5:
        print("✅ MODERATE SIGNAL: Moderate variation in default rates (5-10 points)")
        print("   This feature has some predictive value")
    else:
        print("⚠️  WEAK SIGNAL: Small variation in default rates (<5 points)")
        print("   This feature may have limited predictive power")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    if plot:
        print("\nGenerating plots...")
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Prepare data
            categories = default_analysis[column].values
            n_cats = len(categories)
            x_pos = np.arange(n_cats)
            
            # --------------------------------------------------------
            # PLOT 1: Stacked Bar Chart (Counts)
            # --------------------------------------------------------
            no_defaults_counts = default_analysis['no_defaults'].values
            defaults_counts = default_analysis['defaults'].values
            
            axes[0].bar(x_pos, no_defaults_counts, label='No Default (0)', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[0].bar(x_pos, defaults_counts, bottom=no_defaults_counts, 
                       label='Default (1)', color='#e74c3c', alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
            
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(categories, rotation=45, ha='right')
            axes[0].set_xlabel(column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Number of Loans', fontsize=12, fontweight='bold')
            axes[0].set_title(f'Loan Distribution by {column.replace("_", " ").title()}', 
                            fontsize=14, fontweight='bold', pad=20)
            axes[0].legend(loc='upper right', framealpha=0.9)
            axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[0].set_axisbelow(True)
            
            # --------------------------------------------------------
            # PLOT 2: Default Rate Bar Chart
            # --------------------------------------------------------
            default_rates = default_analysis['default_rate_%'].values
            bars = axes[1].bar(x_pos, default_rates, color='#ff7f0e', alpha=0.8,
                              edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, default_rates)):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{rate:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
            
            # Add overall average reference line
            axes[1].axhline(y=overall_default_rate, color='red', linestyle='--', 
                           linewidth=2, alpha=0.7, 
                           label=f'Overall Avg: {overall_default_rate:.2f}%')
            
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(categories, rotation=45, ha='right')
            axes[1].set_xlabel(column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Default Rate (%)', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Default Rate by {column.replace("_", " ").title()}', 
                            fontsize=14, fontweight='bold', pad=20)
            axes[1].legend(loc='best', framealpha=0.9)
            axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
            axes[1].set_axisbelow(True)
            
            # Add y-axis limit with padding
            axes[1].set_ylim(0, max(default_rates) * 1.15)
            
            plt.tight_layout()
            plt.show()
            print("✅ Plots generated successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return default_analysis



# ============================================================
# SECTION 3: CHI-SQUARE TEST
# ============================================================

def perform_chi_square_test(
    df: pd.DataFrame,
    column: str,
    target: str = 'target',
    alpha: float = 0.05
) -> Dict:
    """
    Perform chi-square test of independence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Categorical feature
    target : str, default='target'
        Binary target
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    dict
        Dictionary with chi-square test results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CHI-SQUARE TEST: {column}")
    logger.info(f"{'='*60}")
    
    # Input validation
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    
    # Create contingency table
    contingency = pd.crosstab(df[column], df[target])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Determine significance level
    if p_value < 0.001:
        significance_level = "Extremely Strong"
        emoji = "✅✅✅"
    elif p_value < 0.01:
        significance_level = "Very Strong"
        emoji = "✅✅"
    elif p_value < alpha:
        significance_level = "Moderate"
        emoji = "✅"
    else:
        significance_level = "Not Significant"
        emoji = "❌"
    
    is_significant = p_value < alpha
    
    # Format p-value display (FIXED)
    if p_value < 0.001:
        p_value_display = f"{p_value:.4e}"
    else:
        p_value_display = f"{p_value:.4f}"
    
    # Print results
    print("\nChi-Square Test Results:")
    print(f"  Chi-square statistic: {chi2:,.2f}")
    print(f"  P-value:             {p_value_display}")
    print(f"  Degrees of freedom:  {dof}")
    print(f"  Significance level:  α = {alpha}")
    
    print(f"\n{'─'*60}")
    print("INTERPRETATION:")
    print(f"{'─'*60}")
    
    if is_significant:
        print(f"{emoji} RESULT: p-value ({p_value_display}) < α ({alpha})")
        print(f"   Evidence strength: {significance_level}")
        print(f"   Decision: REJECT H₀")
        print(f"   → {column} IS significantly associated with default")
    else:
        print(f"{emoji} RESULT: p-value ({p_value_display}) ≥ α ({alpha})")
        print(f"   Evidence strength: {significance_level}")
        print(f"   Decision: FAIL TO REJECT H₀")
        print(f"   → No evidence that {column} affects default")
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'is_significant': is_significant,
        'significance_level': significance_level
    }

# ============================================================
# SECTION 4: CRAMÉR'S V
# ============================================================

def calculate_cramers_v(
    df: pd.DataFrame,
    column: str,
    target: str = 'target'
) -> Dict:
    """
    Calculate Cramér's V effect size.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Categorical feature
    target : str, default='target'
        Binary target
        
    Returns
    -------
    dict
        Dictionary with Cramér's V results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CRAMÉR'S V (EFFECT SIZE): {column}")
    logger.info(f"{'='*60}")
    
    # Input validation
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    
    # Calculate Cramér's V
    contingency = pd.crosstab(df[column], df[target])
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    variance_explained = cramers_v ** 2 * 100
    
    # Interpret strength
    strength = _interpret_cramers_v(cramers_v)
    
    # Print results
    print(f"\nCramér's V: {cramers_v:.4f}")
    print(f"\n{'─'*60}")
    print("Interpretation Scale:")
    print("  0.00 - 0.05: Negligible")
    print("  0.05 - 0.10: Weak")
    print("  0.10 - 0.20: Moderate")
    print("  0.20 - 0.40: Strong")
    print("  0.40+:       Very Strong")
    
    print(f"\n{'─'*60}")
    
    # Emoji based on strength
    if cramers_v >= 0.20:
        emoji = "✅✅"
    elif cramers_v >= 0.10:
        emoji = "✅"
    elif cramers_v >= 0.05:
        emoji = "⚠️"
    else:
        emoji = "❌"
    
    print(f"{emoji} Cramér's V = {cramers_v:.4f} → {strength.upper()}")
    print(f"\nVariance Explained: {variance_explained:.2f}%")
    
    return {
        'cramers_v': cramers_v,
        'variance_explained': variance_explained,
        'strength': strength
    }


# ============================================================
# SECTION 5: FINAL INTERPRETATION
# ============================================================

def interpret_results(
    chi2_results: Dict,
    cramers_results: Dict,
    freq_results: Dict = None,
    default_results: pd.DataFrame = None,
    column: str = "Feature"
) -> None:
    """
    Interpret statistical test results and provide final recommendation.
    
    Parameters
    ----------
    chi2_results : dict
        Results from perform_chi_square_test()
    cramers_results : dict
        Results from calculate_cramers_v()
    freq_results : dict, optional
        Results from analyze_frequency_distribution()
    default_results : pd.DataFrame, optional
        Results from calculate_default_rates()
    column : str, default="Feature"
        Column name for display
        
    Returns
    -------
    None
        Prints recommendation to console
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print(f"║ {'FINAL RECOMMENDATION: ' + column:^66} ║")
    print("╚" + "═"*68 + "╝\n")
    
    # Extract key metrics
    is_significant = chi2_results['is_significant']
    p_value = chi2_results['p_value']
    cramers_v = cramers_results['cramers_v']
    strength = cramers_results['strength']
    variance_explained = cramers_results['variance_explained']
    
    # Optional metrics
    is_quasi_constant = freq_results['is_quasi_constant'] if freq_results else False
    has_rare = len(freq_results['rare_categories']) > 0 if freq_results else False
    n_categories = freq_results['total_categories'] if freq_results else None
    rate_range = (default_results['default_rate_%'].max() - 
                  default_results['default_rate_%'].min()) if default_results is not None else None
    
    # Decision logic
    decision = ""
    rationale = []
    encoding = ""
    
    # Rule 1: Quasi-constant check
    if is_quasi_constant:
        decision = "❌ DROP FEATURE"
        rationale.append("⚠️  Quasi-constant: One category >99% of data")
        rationale.append("   No variation = no predictive power")
        encoding = "N/A"
    
    # Rule 2: Not statistically significant
    elif not is_significant:
        decision = "❌ DROP FEATURE"
        rationale.append(f"❌ Not statistically significant (p = {p_value:.4f} ≥ 0.05)")
        rationale.append("   No evidence of association with default")
        encoding = "N/A"
    
    # Rule 3: Significant but negligible effect
    elif cramers_v < 0.03:
        decision = "❌ DROP FEATURE"
        rationale.append(f"✅ Statistically significant (p < 0.05)")
        rationale.append(f"❌ But Cramér's V = {cramers_v:.4f} (negligible)")
        rationale.append("   Effect too small to be useful")
        encoding = "N/A"
    
    # Rule 4: Strong predictor
    elif cramers_v >= 0.15:
        decision = "✅✅ KEEP FEATURE (Strong Predictor)"
        rationale.append(f"✅ Statistically significant (p < 0.05)")
        rationale.append(f"✅✅ Strong effect size (V = {cramers_v:.4f})")
        rationale.append(f"✅ Explains {variance_explained:.1f}% of default variation")
        if rate_range:
            rationale.append(f"✅ Large default rate range ({rate_range:.1f} points)")
        
        if n_categories and n_categories > 20:
            encoding = "Target Encoding (high cardinality)"
        elif has_rare:
            encoding = "Group rare categories → One-Hot"
        else:
            encoding = "One-Hot Encoding"
    
    # Rule 5: Moderate predictor
    elif cramers_v >= 0.10:
        decision = "✅ KEEP FEATURE (Moderate Predictor)"
        rationale.append(f"✅ Statistically significant (p < 0.05)")
        rationale.append(f"✅ Moderate effect size (V = {cramers_v:.4f})")
        rationale.append(f"   Explains {variance_explained:.1f}% of variation")
        
        if n_categories and n_categories > 20:
            encoding = "Target Encoding or Group by similarity"
        elif has_rare:
            encoding = "Group rare categories → One-Hot"
        else:
            encoding = "One-Hot Encoding"
    
    # Rule 6: Weak but detectable
    elif cramers_v >= 0.05:
        decision = "⚠️  KEEP (Weak Predictor)"
        rationale.append(f"✅ Statistically significant (p < 0.05)")
        rationale.append(f"⚠️  Weak effect size (V = {cramers_v:.4f})")
        rationale.append("   Limited value, but may help in combination")
        
        if has_rare:
            encoding = "Group rare → One-Hot"
        else:
            encoding = "One-Hot Encoding"
    
    # Rule 7: Very weak
    else:
        decision = "❌ DROP FEATURE (Too Weak)"
        rationale.append(f"✅ Statistically significant (p < 0.05)")
        rationale.append(f"❌ Very weak effect (V = {cramers_v:.4f})")
        rationale.append("   Not worth modeling complexity")
        encoding = "N/A"
    
    # Print decision
    print(f"Decision: {decision}\n")
    print("Rationale:")
    for reason in rationale:
        print(f"  {reason}")
    
    print(f"\nRecommended Encoding: {encoding}")
    
    # Implementation code
    if "KEEP" in decision and encoding != "N/A":
        print(f"\nImplementation:")
        if "One-Hot" in encoding:
            print(f"  df['{column}'] = df['{column}'].str.strip()")
            if has_rare and freq_results:
                print(f"  # Group rare categories: {freq_results['rare_categories']}")
                print(f"  rare_cats = {freq_results['rare_categories']}")
                print(f"  df['{column}'] = df['{column}'].apply(lambda x: 'Other' if x in rare_cats else x)")
            print(f"  dummies = pd.get_dummies(df['{column}'], prefix='{column[:4]}', drop_first=True)")
        elif "Target" in encoding:
            print(f"  from category_encoders import TargetEncoder")
            print(f"  encoder = TargetEncoder(cols=['{column}'])")
            print(f"  df['{column}_encoded'] = encoder.fit_transform(df['{column}'], df['target'])")
    
    print("\n" + "═"*70 + "\n")


# ============================================================
# SECTION 6: MISSING VALUES CHECK (NEW)
# ============================================================

def check_missing_values(
    df: pd.DataFrame,
    column: str
) -> Dict:
    """
    Check missing values in categorical feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check
        
    Returns
    -------
    dict
        Dictionary with:
        - 'missing_count': int
        - 'missing_percentage': float
        - 'total_rows': int
        - 'non_missing': int
        - 'has_missing': bool
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"MISSING VALUES CHECK: {column}")
    logger.info(f"{'='*60}")
    
    # Input validation
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Calculate missing values
    total_rows = len(df)
    missing_count = df[column].isnull().sum()
    missing_percentage = (missing_count / total_rows) * 100
    non_missing = total_rows - missing_count
    has_missing = missing_count > 0
    
    # Print results
    print(f"\nMissing Value Summary:")
    print(f"  Total rows:           {total_rows:,}")
    print(f"  Non-missing values:   {non_missing:,} ({(non_missing/total_rows)*100:.2f}%)")
    print(f"  Missing values:       {missing_count:,} ({missing_percentage:.2f}%)")
    
    print(f"\n{'─'*60}")
    
    if missing_count == 0:
        print("✅ No missing values detected")
        print("   This feature is complete - no imputation needed")
    elif missing_percentage < 1:
        print(f"✅ Minimal missing values (<1%)")
        print(f"   Can safely drop or impute with mode")
    elif missing_percentage < 5:
        print(f"⚠️  Small amount of missing values ({missing_percentage:.2f}%)")
        print(f"   Recommended: Impute with mode or create 'Unknown' category")
    elif missing_percentage < 30:
        print(f"⚠️  Moderate missing values ({missing_percentage:.2f}%)")
        print(f"   Recommended: Create 'Unknown' category or investigate pattern")
    else:
        print(f"❌ High proportion of missing values ({missing_percentage:.2f}%)")
        print(f"   Consider dropping feature or investigating if missing = meaningful")
    
    # Additional info for categorical
    if not has_missing:
        print(f"\n   All {total_rows:,} rows have valid values")
    else:
        print(f"\n   {non_missing:,} rows available for analysis")
        print(f"   {missing_count:,} rows would be excluded if dropping NaN")
    
    return {
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
        'total_rows': total_rows,
        'non_missing': non_missing,
        'has_missing': has_missing
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _interpret_cramers_v(v: float) -> str:
    """Return strength category for Cramér's V value"""
    if v < 0.05:
        return "Negligible"
    elif v < 0.10:
        return "Weak"
    elif v < 0.20:
        return "Moderate"
    elif v < 0.40:
        return "Strong"
    else:
        return "Very Strong"


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("✅ EDA Helpers module loaded successfully!")
    print("\nAvailable functions:")
    print("  1. analyze_frequency_distribution()")
    print("  2. calculate_default_rates()")
    print("  3. perform_chi_square_test()")
    print("  4. calculate_cramers_v()")
    print("  5. interpret_results()")
    print("\nUsage:")
    print("  freq = analyze_frequency_distribution(df, 'term')")
    print("  rates = calculate_default_rates(df, 'term')")
    print("  chi2 = perform_chi_square_test(df, 'term')")
    print("  cramers = calculate_cramers_v(df, 'term')")
    print("  interpret_results(chi2, cramers, freq, rates, 'term')")
