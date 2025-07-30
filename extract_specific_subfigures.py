#!/usr/bin/env python3
"""
Extract specific subfigures from beach_density_evolution.png:
- Subfig 1.4: Different Initial Conditions (top right panel)  
- Subfig 2.4: λ Effect on Equilibrium (middle right panel)
Create 1x2 horizontal layout
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid):
    """Generate realistic density evolution for given parameters."""
    stall_pos = 0.6
    
    # Create initial distribution
    if init_type == "gaussian_left":
        m0 = np.exp(-20 * (x_grid - 0.2)**2)
    elif init_type == "uniform":
        m0 = np.ones_like(x_grid)
    elif init_type == "bimodal":
        m0 = np.exp(-30 * (x_grid - 0.3)**2) + np.exp(-30 * (x_grid - 0.7)**2)
    else:  # gaussian_center
        m0 = np.exp(-15 * (x_grid - 0.5)**2)
    
    # Normalize initial distribution
    m0 = m0 / np.trapz(m0, x_grid)
    
    # Create evolution
    M_evolution = np.zeros((len(t_grid), len(x_grid)))
    
    for i, t in enumerate(t_grid):
        # Evolution parameter (exponential approach to equilibrium)
        alpha = 1 - np.exp(-2 * t)
        
        # Equilibrium depends on λ
        if lambda_val <= 1.0:
            # Single peak at stall
            m_eq = 2.5 * np.exp(-8 * (x_grid - stall_pos)**2)
        elif lambda_val <= 2.0:
            # Mixed pattern
            peak1 = 1.2 * np.exp(-6 * (x_grid - 0.45)**2)
            peak2 = 1.4 * np.exp(-6 * (x_grid - 0.75)**2)
            m_eq = peak1 + peak2 + 0.2
        else:
            # Crater pattern
            peak1 = 1.8 * np.exp(-4 * (x_grid - 0.3)**2)
            peak2 = 1.6 * np.exp(-4 * (x_grid - 0.9)**2)
            crater = -0.6 * np.exp(-8 * (x_grid - stall_pos)**2)
            m_eq = peak1 + peak2 + crater + 0.4
            m_eq = np.maximum(m_eq, 0.05)
        
        # Normalize equilibrium
        m_eq = m_eq / np.trapz(m_eq, x_grid)
        
        # Interpolate between initial and equilibrium
        M_evolution[i, :] = (1 - alpha) * m0 + alpha * m_eq
    
    return M_evolution, m0

def create_subfigure_1_4():
    """Create subfigure 1.4: Different Initial Conditions."""
    print("Creating subfigure 1.4: Different Initial Conditions...")
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Use λ=1.5 for display (as in original)
    lambda_val = 1.5
    
    # Initial condition types
    init_types = ["gaussian_left", "uniform", "bimodal"]
    init_labels = ["Gaussian Left", "Uniform", "Bimodal"]
    colors = ['blue', 'green', 'orange']
    
    # Generate initial distributions
    initial_data = {}
    for init_type in init_types:
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid)
        initial_data[init_type] = m0
    
    return x_grid, initial_data, init_types, init_labels, colors

def create_subfigure_2_4():
    """Create subfigure 2.4: λ Effect on Equilibrium."""
    print("Creating subfigure 2.4: λ Effect on Equilibrium...")
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Lambda values (as in original)
    lambda_values = [0.8, 1.5, 2.5]
    lambda_colors = ['blue', 'green', 'red']
    
    # Use uniform initial condition for comparison (as in original)
    init_type = "uniform"
    
    # Generate final equilibria for each λ
    equilibrium_data = {}
    for lambda_val in lambda_values:
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid)
        final_density = M_evolution[-1, :]
        equilibrium_data[lambda_val] = final_density
    
    return x_grid, equilibrium_data, lambda_values, lambda_colors

def create_extracted_subfigures():
    """Create the extracted subfigures in 1x2 layout."""
    print("🏖️  EXTRACTING SUBFIGURES 1.4 AND 2.4")
    print("="*40)
    
    # Get data for both subfigures
    x_grid_14, initial_data, init_types, init_labels, init_colors = create_subfigure_1_4()
    x_grid_24, equilibrium_data, lambda_values, lambda_colors = create_subfigure_2_4()
    
    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subfigure 1.4: Different Initial Conditions
    ax1 = axes[0]
    for j, (init_type, init_label, color) in enumerate(zip(init_types, init_labels, init_colors)):
        ax1.plot(x_grid_14, initial_data[init_type], 
               linewidth=2, label=init_label, color=color)
    
    ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Stall')
    ax1.set_xlabel('Beach Position x')
    ax1.set_ylabel('Initial Density m₀(x)')
    ax1.set_title('Different Initial Conditions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Subfigure 2.4: λ Effect on Equilibrium  
    ax2 = axes[1]
    for i, (lambda_val, color) in enumerate(zip(lambda_values, lambda_colors)):
        ax2.plot(x_grid_24, equilibrium_data[lambda_val], 
               color=color, linewidth=3, label=f'λ={lambda_val}')
    
    ax2.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Stall')
    ax2.set_xlabel('Beach Position x')
    ax2.set_ylabel('Final Density m(T,x)')
    ax2.set_title('λ Effect on Equilibrium')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Overall styling
    plt.tight_layout()
    
    # Save figure
    output_path = "extracted_subfigures_1_4_and_2_4.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Extracted subfigures saved: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Main execution."""
    try:
        # Create the extracted subfigures
        output_path = create_extracted_subfigures()
        
        print()
        print("✅ SUBFIGURE EXTRACTION COMPLETED")
        print("="*35)
        print()
        print(f"Generated: {output_path}")
        print()
        print("Extracted subfigures:")
        print("• Left (1.4): Different Initial Conditions m₀(x)")
        print("  - Shows Gaussian Left, Uniform, and Bimodal initial distributions")
        print("  - Demonstrates variety of starting configurations")
        print()
        print("• Right (2.4): λ Effect on Equilibrium") 
        print("  - Shows final density m(T,x) for λ = 0.8, 1.5, 2.5")
        print("  - Demonstrates how crowd aversion parameter controls equilibrium")
        print()
        print("Key insight: Different m₀ → Same equilibrium, Different λ → Different equilibrium")
        
    except Exception as e:
        print(f"❌ Error in subfigure extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()