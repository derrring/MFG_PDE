#!/usr/bin/env python3
"""Test the font fix for matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

# Test the fixed configuration
from mfg_pde.utils.advanced_visualization import MFGVisualizer

def test_font_configuration():
    """Test that fonts work without warnings."""
    
    print("🔤 Testing matplotlib font configuration...")
    
    # Create a simple test plot
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) * np.exp(-x)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='$u(t,x) = \sin(2\pi x)e^{-x}$')
    ax.set_xlabel('Space $x$')
    ax.set_ylabel('Value function $u(t,x)$')
    ax.set_title('Font Test: Mathematical Expressions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('results/font_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Font configuration test completed!")
    print("📄 Test plot saved: results/font_test.png")

if __name__ == "__main__":
    test_font_configuration()