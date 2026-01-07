
"""
Plotting Configuration for Credit Risk EDA
Centralized settings for consistent visualizations across all notebooks
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Tuple

# ============================================================
# COLOR PALETTES
# ============================================================

# Credit Risk Theme Colors
COLORS = {
    # Primary colors
    'primary_blue': '#1f77b4',
    'primary_orange': '#ff7f0e',
    
    # Risk indicators
    'good': '#2ecc71',      # Green - No default
    'bad': '#e74c3c',       # Red - Default
    'warning': '#f39c12',   # Orange - Warning
    'neutral': '#95a5a6',   # Gray - Neutral
    
    # Category colors (for multi-category plots)
    'cat_palette': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
    
    # Gradient (for heatmaps/continuous)
    'risk_gradient': ['#2ecc71', '#f39c12', '#e74c3c'],  # Green → Orange → Red
}

# Seaborn color palettes
TARGET_COLORS = [COLORS['good'], COLORS['bad']]  # For binary target (0, 1)
CATEGORICAL_PALETTE = sns.color_palette(COLORS['cat_palette'])

# ============================================================
# FIGURE SETTINGS
# ============================================================

# Default figure sizes
FIGSIZE = {
    'small': (8, 5),
    'medium': (12, 6),
    'large': (14, 8),
    'wide': (16, 5),
    'square': (8, 8),
    'report': (10, 6),  # Good for reports/presentations
}

# DPI settings
DPI = {
    'screen': 100,      # For notebook display
    'print': 300,       # For saving high-quality images
}

# ============================================================
# STYLE CONFIGURATION
# ============================================================

def set_plot_style(style: str = 'whitegrid', context: str = 'notebook') -> None:
    """
    Set global plotting style for all visualizations.
    
    Parameters
    ----------
    style : str, default='whitegrid'
        Seaborn style: 'whitegrid', 'darkgrid', 'white', 'dark', 'ticks'
    context : str, default='notebook'
        Context: 'paper', 'notebook', 'talk', 'poster'
        Controls font sizes and line widths
        
    Examples
    --------
    >>> set_plot_style('whitegrid', 'notebook')
    """
    # Set seaborn style and context
    sns.set_style(style)
    sns.set_context(context, font_scale=1.1)
    
    # Set matplotlib defaults
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': FIGSIZE['medium'],
        'figure.dpi': DPI['screen'],
        'figure.facecolor': 'white',
        'savefig.dpi': DPI['print'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Axes settings
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.fancybox': True,
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


def get_figsize(size: str = 'medium') -> Tuple[int, int]:
    """
    Get predefined figure size.
    
    Parameters
    ----------
    size : str, default='medium'
        Size name: 'small', 'medium', 'large', 'wide', 'square', 'report'
        
    Returns
    -------
    tuple
        (width, height) in inches
        
    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=get_figsize('large'))
    """
    if size not in FIGSIZE:
        raise ValueError(f"Size must be one of {list(FIGSIZE.keys())}, got '{size}'")
    return FIGSIZE[size]


# ============================================================
# PLOT FORMATTING UTILITIES
# ============================================================

def format_axis_labels(ax, xlabel: str = None, ylabel: str = None, 
                       title: str = None, title_size: int = 14) -> None:
    """
    Apply consistent formatting to axis labels and title.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to format
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    title_size : int, default=14
        Title font size
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> format_axis_labels(ax, xlabel='Term', ylabel='Count', title='Distribution')
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=title_size, fontweight='bold', pad=15)
    
    # Rotate x-labels if they're long
    labels = ax.get_xticklabels()
    if labels and max([len(label.get_text()) for label in labels] or [0]) > 10:
        ax.tick_params(axis='x', rotation=45, labelsize=10)
    else:
        ax.tick_params(axis='x', rotation=0)


def add_value_labels(ax, orientation: str = 'vertical', 
                     fmt: str = '{:.0f}', padding: int = 3) -> None:
    """
    Add value labels on top of bars.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis with bar plot
    orientation : str, default='vertical'
        'vertical' or 'horizontal' bars
    fmt : str, default='{:.0f}'
        Format string for labels (e.g., '{:.1f}%' for percentages)
    padding : int, default=3
        Padding between bar and label
        
    Examples
    --------
    >>> ax.bar(categories, values)
    >>> add_value_labels(ax, fmt='{:.1f}%')
    """
    for container in ax.containers:
        labels = [fmt.format(v) if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=padding, fontsize=9)


def add_reference_line(ax, value: float, label: str = None, 
                       color: str = 'red', linestyle: str = '--',
                       linewidth: float = 2, alpha: float = 0.7) -> None:
    """
    Add horizontal reference line (e.g., overall average).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add line to
    value : float
        Y-value for horizontal line
    label : str, optional
        Legend label
    color : str, default='red'
        Line color
    linestyle : str, default='--'
        Line style
    linewidth : float, default=2
        Line width
    alpha : float, default=0.7
        Transparency (0-1)
        
    Examples
    --------
    >>> ax.bar(categories, default_rates)
    >>> add_reference_line(ax, 20.07, label='Overall Avg (20.07%)')
    """
    ax.axhline(y=value, color=color, linestyle=linestyle, 
               linewidth=linewidth, alpha=alpha, label=label, zorder=10)
    if label:
        ax.legend(loc='best')


def format_percentage_axis(ax, axis: str = 'y') -> None:
    """
    Format axis to display percentages (0-100 scale).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format
    axis : str, default='y'
        Which axis to format: 'x' or 'y'
        
    Examples
    --------
    >>> ax.plot(x, percentages)
    >>> format_percentage_axis(ax, axis='y')
    """
    from matplotlib.ticker import PercentFormatter
    
    if axis == 'y':
        ax.yaxis.set_major_formatter(PercentFormatter())
    elif axis == 'x':
        ax.xaxis.set_major_formatter(PercentFormatter())


def save_plot(fig, filename: str, dpi: int = None, 
              bbox_inches: str = 'tight', **kwargs) -> None:
    """
    Save plot with consistent settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename (include path and extension)
    dpi : int, optional
        Resolution (defaults to print quality)
    bbox_inches : str, default='tight'
        Bounding box setting
    **kwargs
        Additional arguments for plt.savefig()
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> save_plot(fig, 'outputs/term_distribution.png')
    """
    if dpi is None:
        dpi = DPI['print']
    
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                facecolor='white', edgecolor='none', **kwargs)
    print(f"✅ Plot saved: {filename}")


# ============================================================
# SPECIALIZED EDA PLOT TEMPLATES
# ============================================================

def create_categorical_plot_grid(n_plots: int = 4) -> Tuple:
    """
    Create standard grid layout for categorical analysis.
    
    Parameters
    ----------
    n_plots : int, default=4
        Number of subplots (2 or 4)
        
    Returns
    -------
    tuple
        (fig, axes) - Figure and axes array
        
    Examples
    --------
    >>> fig, axes = create_categorical_plot_grid(4)
    >>> axes[0, 0].bar(...)  # Frequency plot
    >>> axes[0, 1].bar(...)  # Default rate plot
    """
    if n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['large'])
    elif n_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    else:
        raise ValueError("n_plots must be 2 or 4")
    
    plt.tight_layout(pad=3.0)
    return fig, axes


# ============================================================
# INITIALIZE ON IMPORT
# ============================================================

# Automatically apply default style when module is imported
set_plot_style('whitegrid', 'notebook')

print("✅ Plotting configuration loaded")
print(f"   Default figure size: {FIGSIZE['medium']}")
print(f"   Available sizes: {list(FIGSIZE.keys())}")
print(f"   Color palette: {len(COLORS['cat_palette'])} colors")
