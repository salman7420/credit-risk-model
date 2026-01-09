# ============================================================
# SECTION 3: NUMERICAL FEATURES EDA
# ============================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr



def profile_numeric_feature(
    df: pd.DataFrame,
    column: str,
    target: str = 'target',
    bins: int = 10,
    plot: bool = True
) -> dict:
    """
    Comprehensive EDA for a single numeric feature.
    
    Parameters
    ----------
    df : DataFrame
    column : numeric column name
    target : binary target column (0/1)
    bins : number of bins for default rate analysis
    plot : whether to show visualizations
    
    Returns
    -------
    dict with profile, default rates, correlation, IV, recommendations
    """
    from loguru import logger
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pointbiserialr
    
    logger.info(f"Profiling numeric feature: {column}")
    
    # ============================================================
    # 1. BASIC PROFILE
    # ============================================================
    feature = df[column].dropna()
    profile = {
        'feature': column,
        'n_total': len(df),
        'n_valid': len(feature),
        'n_missing': df[column].isnull().sum(),
        'pct_missing': (df[column].isnull().sum() / len(df)) * 100,
        'n_unique': df[column].nunique(),
        'min': feature.min(),
        'max': feature.max(),
        'mean': feature.mean(),
        'median': feature.median(),
        'std': feature.std(),
        'skew': feature.skew(),
        'kurtosis': feature.kurtosis()
    }
    
    # ============================================================
    # 2. PERCENTILES (for outlier/cap decisions)
    # ============================================================
    percentiles = {
        'p1': feature.quantile(0.01),
        'p5': feature.quantile(0.05),
        'p25': feature.quantile(0.25),
        'p75': feature.quantile(0.75),
        'p95': feature.quantile(0.95),
        'p99': feature.quantile(0.99),
        'p99_5': feature.quantile(0.995)
    }
    
    # ============================================================
    # 3. DEFAULT RATE BY BIN
    # ============================================================
    temp_df = df[[column, target]].dropna()
    temp_df[target] = pd.to_numeric(temp_df[target], errors='coerce')
    temp_df = temp_df.dropna()
    
    # Create quantile-based bins
    temp_df['_bin'] = pd.qcut(temp_df[column], q=bins, duplicates='drop')
    
    default_by_bin = temp_df.groupby('_bin', observed=True)[target].agg([
        ('count', 'count'),
        ('defaults', 'sum'),
        ('default_rate_%', lambda x: x.mean() * 100)
    ]).reset_index()
    
    # Extract bin midpoints for plotting
    default_by_bin['bin_midpoint'] = default_by_bin['_bin'].apply(
        lambda x: (x.left + x.right) / 2
    )
    
    # ============================================================
    # 4. CORRELATION & IV (Information Value)
    # ============================================================
    corr, pval = pointbiserialr(temp_df[target], temp_df[column])
    
    # IV calculation (binned)
    iv_sum = 0
    for _, row in default_by_bin.iterrows():
        n_good = row['count'] - row['defaults']
        n_bad = row['defaults']
        
        # Avoid division by zero
        if n_good > 0 and n_bad > 0:
            pct_good = n_good / (temp_df[target] == 0).sum()
            pct_bad = n_bad / (temp_df[target] == 1).sum()
            iv_sum += (pct_bad - pct_good) * np.log(pct_bad / pct_good)
    
    # ============================================================
    # 5. PRINT SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print(f"NUMERIC FEATURE PROFILE: {column}")
    print("="*70)
    
    print(f"\n{'─'*70}")
    print("Basic Statistics:")
    print(f"  Valid rows: {profile['n_valid']:,} ({100 - profile['pct_missing']:.2f}%)")
    print(f"  Missing: {profile['n_missing']:,} ({profile['pct_missing']:.2f}%)")
    print(f"  Unique values: {profile['n_unique']}")
    print(f"  Range: [{profile['min']:.2f}, {profile['max']:.2f}]")
    print(f"  Mean: {profile['mean']:.2f}, Median: {profile['median']:.2f}")
    print(f"  Std Dev: {profile['std']:.2f}, Skew: {profile['skew']:.2f}")
    
    print(f"\n{'─'*70}")
    print("Percentiles (for capping decisions):")
    print(f"  p1: {percentiles['p1']:.2f}")
    print(f"  p5: {percentiles['p5']:.2f}")
    print(f"  p95: {percentiles['p95']:.2f}")
    print(f"  p99: {percentiles['p99']:.2f}")
    print(f"  p99.5: {percentiles['p99_5']:.2f}")
    
    print(f"\n{'─'*70}")
    print("Association with Default:")
    print(f"  Point-biserial correlation: {corr:.4f}")
    print(f"  P-value: {pval:.2e}")
    print(f"  Information Value (IV): {iv_sum:.4f}")
    
    # Interpret IV
    if iv_sum < 0.02:
        iv_interpretation = "NEGLIGIBLE"
    elif iv_sum < 0.1:
        iv_interpretation = "WEAK"
    elif iv_sum < 0.3:
        iv_interpretation = "MODERATE"
    else:
        iv_interpretation = "STRONG"
    print(f"  IV Interpretation: {iv_interpretation}")
    
    print(f"\n{'─'*70}")
    print("Default Rate by Bin:")
    print(default_by_bin[['_bin', 'count', 'defaults', 'default_rate_%']].to_string(index=False))
    
    # Check monotonicity
    rates = default_by_bin['default_rate_%'].values
    is_monotonic_inc = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    is_monotonic_dec = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    
    if is_monotonic_inc or is_monotonic_dec:
        pattern = "MONOTONIC" + (" (increasing)" if is_monotonic_inc else " (decreasing)")
    else:
        pattern = "NON-MONOTONIC"
    
    print(f"  Pattern: {pattern}")
    
    logger.success(f"Profiling complete for {column}")
    
    # ============================================================
    # 6. VISUALIZATION
    # ============================================================
    if plot:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Distribution (histogram)
        axes[0, 0].hist(feature, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(profile['mean'], color='red', linestyle='--', 
                           linewidth=2, label=f"Mean: {profile['mean']:.2f}")
        axes[0, 0].axvline(profile['median'], color='orange', linestyle='--', 
                           linewidth=2, label=f"Median: {profile['median']:.2f}")
        axes[0, 0].set_xlabel(column, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title(f'Distribution of {column}', fontweight='bold', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # Plot 2: Boxplot
        axes[0, 1].boxplot(feature, vert=True)
        axes[0, 1].set_ylabel(column, fontweight='bold')
        axes[0, 1].set_title(f'Boxplot of {column}', fontweight='bold', fontsize=12)
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Plot 3: Default rate by bin
        axes[1, 0].bar(range(len(default_by_bin)), default_by_bin['default_rate_%'], 
                      color='#ff7f0e', alpha=0.8, edgecolor='black')
        axes[1, 0].axhline(temp_df[target].mean() * 100, color='red', linestyle='--',
                          linewidth=2, label=f"Overall: {temp_df[target].mean()*100:.1f}%")
        axes[1, 0].set_xlabel('Bin', fontweight='bold')
        axes[1, 0].set_ylabel('Default Rate (%)', fontweight='bold')
        axes[1, 0].set_title(f'Default Rate by {column} Bin', fontweight='bold', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Plot 4: Zoomed distribution (1-99 percentile)
        p1, p99 = feature.quantile([0.01, 0.99])
        feature_zoom = feature[(feature >= p1) & (feature <= p99)]
        axes[1, 1].hist(feature_zoom, bins=40, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel(column, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title(f'{column} (1-99th percentile)', fontweight='bold', fontsize=12)
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        print("✅ Plots generated")
    
    return {
        'profile': profile,
        'percentiles': percentiles,
        'default_by_bin': default_by_bin,
        'correlation': corr,
        'pvalue': pval,
        'iv': iv_sum,
        'iv_interpretation': iv_interpretation,
        'monotonic_pattern': pattern
    }


# ============================================================
# HELPER: Create summary of multiple numeric features
# ============================================================

def summarize_numeric_features(
    results_dict: dict
) -> pd.DataFrame:
    """
    Create a summary table from multiple profile_numeric_feature() results.
    
    Parameters
    ----------
    results_dict : dict
        Keys = feature names, values = output from profile_numeric_feature()
    
    Returns
    -------
    DataFrame with one row per feature, ready for documentation
    """
    rows = []
    
    for feature_name, result in results_dict.items():
        rows.append({
            'Feature': feature_name,
            'Unique': result['profile']['n_unique'],
            'Missing %': f"{result['profile']['pct_missing']:.1f}%",
            'Skew': f"{result['profile']['skew']:.2f}",
            'Correlation': f"{result['correlation']:.4f}",
            'IV': f"{result['iv']:.4f}",
            'IV Signal': result['iv_interpretation'],
            'Pattern': result['monotonic_pattern']
        })
    
    summary_df = pd.DataFrame(rows)
    return summary_df