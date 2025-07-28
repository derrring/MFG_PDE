#!/usr/bin/env python3
"""
Matplotlib Font Configuration Utility

This utility provides standard font settings to avoid font warnings
and ensure consistent visualization across the MFG_PDE examples.
"""

import matplotlib.pyplot as plt
import warnings

def configure_matplotlib_fonts():
    """
    Configure matplotlib to use safe fonts and avoid glyph warnings.
    
    This function sets up matplotlib to use DejaVu Sans font which
    has good Unicode support and avoids the common subscript/superscript
    glyph warnings that occur with default Arial font.
    """
    
    # Set font family to DejaVu Sans (widely available and Unicode-complete)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Disable Unicode minus to avoid font warnings
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set default font sizes for better readability
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Improve figure quality
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Filter specific matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*Glyph.*missing from font.*')
    
    print("✅ Matplotlib fonts configured successfully")
    print(f"   Font family: {plt.rcParams['font.family']}")
    print(f"   Unicode minus: {plt.rcParams['axes.unicode_minus']}")


def get_safe_math_symbols():
    """
    Return a dictionary of safe mathematical symbols that work across fonts.
    
    Returns:
    --------
    dict
        Dictionary mapping common math symbols to safe alternatives
    """
    
    safe_symbols = {
        'subscript_0': '0',
        'subscript_1': '1', 
        'subscript_2': '2',
        'arrow_right': '->',
        'arrow_left': '<-',
        'arrow_both': '<->',
        'theta': 'theta',
        'sigma': 'sigma',
        'mu': 'mu',
        'lambda': 'lambda',
        'alpha': 'alpha',
        'beta': 'beta',
        'gamma': 'gamma',
        'delta': 'delta',
        'epsilon': 'epsilon',
        'infinity': 'inf',
        'partial': 'd',
        'integral': 'integral',
        'sum': 'sum',
        'product': 'product'
    }
    
    return safe_symbols


def demo_safe_plotting():
    """Demonstrate safe plotting with proper font configuration."""
    
    configure_matplotlib_fonts()
    
    import numpy as np
    
    # Generate sample data
    t = np.linspace(0, 10, 100)
    y1 = np.sin(t)
    y2 = np.cos(t)
    
    # Create demonstration plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, y1, 'b-', linewidth=2, label='u0 (Home)')
    ax.plot(t, y2, 'r-', linewidth=2, label='u1 (Bar)')
    ax.plot(t, y1 - y2, 'k--', linewidth=2, label='u1 - u0 (Difference)')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Value Functions')
    ax.set_title('Example: Safe Mathematical Notation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation with safe symbols
    ax.text(0.05, 0.95, 'P(Home -> Bar) transition probability', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('font_demo_safe.png', dpi=300)
    plt.show()
    
    print("📊 Safe plotting demonstration completed!")
    print("   No font warnings should appear above.")


if __name__ == "__main__":
    print("🎨 Matplotlib Font Configuration Utility")
    print("=" * 50)
    
    # Configure fonts
    configure_matplotlib_fonts()
    
    # Show available safe symbols
    symbols = get_safe_math_symbols()
    print("\n📝 Available safe mathematical symbols:")
    for key, value in list(symbols.items())[:10]:  # Show first 10
        print(f"   {key}: '{value}'")
    print(f"   ... and {len(symbols)-10} more")
    
    # Run demonstration
    print("\n🎯 Running demonstration...")
    demo_safe_plotting()
    
    print("\n✅ Font configuration utility completed!")
    print("   Use configure_matplotlib_fonts() in your scripts to avoid font warnings.")