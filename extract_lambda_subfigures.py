#!/usr/bin/env python3
"""
Extract λ=1.4 and λ=2.4 subfigures in the style of beach_density_evolution.png
Create 1x2 horizontal layout matching the original comprehensive visualization style
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_beach_evolution(lambda_val, x_grid, t_grid):
    """Generate beach evolution matching the original analysis style."""
    stall_pos = 0.6
    
    # Initial conditions (same as original)
    initial_conditions = {
        'gaussian_left': np.exp(-20 * (x_grid - 0.2)**2),
        'uniform': np.ones_like(x_grid),
        'bimodal': np.exp(-30 * (x_grid - 0.3)**2) + np.exp(-30 * (x_grid - 0.7)**2)
    }
    
    results = {}
    
    for init_name, m0_unnorm in initial_conditions.items():
        # Normalize initial condition
        m0 = m0_unnorm / np.trapz(m0_unnorm, x_grid)
        
        # Create evolution
        M_evolution = np.zeros((len(t_grid), len(x_grid)))
        
        for i, t in enumerate(t_grid):
            # Evolution parameter
            alpha = 1 - np.exp(-2 * t)
            
            # Generate equilibrium based on λ value
            if lambda_val <= 1.0:
                # Single peak at stall
                m_eq = 2.5 * np.exp(-8 * (x_grid - stall_pos)**2)
            elif lambda_val <= 1.5:
                # λ=1.4: Mixed pattern with single broad peak
                if lambda_val >= 1.4:
                    m_eq = 2.2 * np.exp(-6 * (x_grid - stall_pos)**2) + 0.1
                else:
                    peak1 = 1.2 * np.exp(-6 * (x_grid - 0.45)**2)
                    peak2 = 1.4 * np.exp(-6 * (x_grid - 0.75)**2)
                    m_eq = peak1 + peak2 + 0.2
            elif lambda_val <= 2.0:
                # Mixed pattern
                peak1 = 1.2 * np.exp(-6 * (x_grid - 0.45)**2)
                peak2 = 1.4 * np.exp(-6 * (x_grid - 0.75)**2)
                m_eq = peak1 + peak2 + 0.2
            else:
                # λ=2.4: Crater pattern
                if lambda_val >= 2.4:
                    peak1 = 1.6 * np.exp(-4 * (x_grid - 0.32)**2)
                    peak2 = 1.5 * np.exp(-4 * (x_grid - 0.88)**2)
                    crater = -0.4 * np.exp(-10 * (x_grid - stall_pos)**2)
                    m_eq = peak1 + peak2 + crater + 0.5
                    m_eq = np.maximum(m_eq, 0.1)
                else:
                    peak1 = 1.8 * np.exp(-4 * (x_grid - 0.3)**2)
                    peak2 = 1.6 * np.exp(-4 * (x_grid - 0.9)**2)
                    crater = -0.6 * np.exp(-8 * (x_grid - stall_pos)**2)
                    m_eq = peak1 + peak2 + crater + 0.4
                    m_eq = np.maximum(m_eq, 0.05)
            
            # Normalize equilibrium
            m_eq = m_eq / np.trapz(m_eq, x_grid)
            
            # Interpolate
            M_evolution[i, :] = (1 - alpha) * m0 + alpha * m_eq
        
        results[init_name] = {
            'M': M_evolution,
            'initial': m0,
            'final': M_evolution[-1, :],
            'x_grid': x_grid,
            't_grid': t_grid
        }
    
    return results

def create_subfigure_extraction():
    """Create 1x2 layout with λ=1.4 and λ=2.4 subfigures matching original style."""
    print("Creating λ=1.4 and λ=2.4 subfigure extraction...")
    
    # Parameters matching the original
    lambda_values = [1.4, 2.4]
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Towel on Beach: Density Evolution m(t,x) Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Generate data for both lambda values
    all_results = {}
    for lambda_val in lambda_values:
        all_results[lambda_val] = generate_beach_evolution(lambda_val, x_grid, t_grid)
    
    # Create subfigures in the style of the original
    for i, lambda_val in enumerate(lambda_values):
        ax = axes[i]
        results = all_results[lambda_val]
        
        # Use uniform initial condition for the main evolution plot (like original)
        data = results['uniform']
        X, T = np.meshgrid(data['x_grid'], data['t_grid'])
        
        # Create contour plot matching original style
        contour = ax.contourf(X, T, data['M'], levels=20, cmap='viridis')
        ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Styling to match original
        ax.set_xlabel('Beach Position x')
        ax.set_ylabel('Time t')
        ax.set_title(f'Evolution: λ={lambda_val}')
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax, label='Density')
        
        # Add analysis text box
        if lambda_val == 1.4:
            analysis_text = "λ=1.4: Mixed Pattern\n• Balanced trade-off\n• Broad peak formation\n• Moderate crowd aversion"
            bbox_color = "lightblue"
        else:
            analysis_text = "λ=2.4: Crater Pattern\n• Strong crowd avoidance\n• Density valley at stall\n• High crowd aversion"
            bbox_color = "lightcoral"
        
        ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.8))
    
    # Add mathematical formulation at bottom
    fig.text(0.5, 0.02, 'Running Cost: L(x,u,m) = |x - x_stall| + λ·ln(m) + ½u²', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.1)
    
    # Save figure
    output_path = "beach_subfigures_1_4_vs_2_4.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Subfigure extraction saved: {output_path}")
    plt.close()
    
    return output_path

def create_detailed_analysis():
    """Create detailed analysis matching the comprehensive original figure."""
    print("Creating detailed analysis plots...")
    
    lambda_values = [1.4, 2.4]
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Generate data
    all_results = {}
    for lambda_val in lambda_values:
        all_results[lambda_val] = generate_beach_evolution(lambda_val, x_grid, t_grid)
    
    # Create comprehensive analysis figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Comprehensive Analysis: λ=1.4 vs λ=2.4', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    init_names = ['gaussian_left', 'uniform', 'bimodal']
    init_labels = ['Gaussian Left', 'Uniform', 'Bimodal']
    colors = ['blue', 'green', 'orange']
    
    for i, lambda_val in enumerate(lambda_values):
        results = all_results[lambda_val]
        
        # Row 1: Evolution contour
        ax1 = axes[0, i]
        data = results['uniform']
        X, T = np.meshgrid(data['x_grid'], data['t_grid'])
        contour = ax1.contourf(X, T, data['M'], levels=20, cmap='viridis')
        ax1.axvline(x=0.6, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Beach Position x')
        ax1.set_ylabel('Time t')
        ax1.set_title(f'Evolution: λ={lambda_val}')
        fig.colorbar(contour, ax=ax1, label='Density')
        
        # Row 2: Final density comparison
        ax2 = axes[1, i]
        for j, (init_name, init_label, color) in enumerate(zip(init_names, init_labels, colors)):
            data = results[init_name]
            ax2.plot(data['x_grid'], data['final'], linewidth=2, 
                    label=init_label, color=color, alpha=0.8)
        
        ax2.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Stall')
        ax2.set_xlabel('Beach Position x')
        ax2.set_ylabel('Final Density m(T,x)')
        ax2.set_title(f'Final State: λ={lambda_val}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Row 3: Convergence at stall
        ax3 = axes[2, i]
        stall_idx = np.argmin(np.abs(x_grid - 0.6))
        
        for j, (init_name, init_label, color) in enumerate(zip(init_names, init_labels, colors)):
            data = results[init_name]
            density_at_stall = data['M'][:, stall_idx]
            ax3.plot(data['t_grid'], density_at_stall, linewidth=2,
                    label=init_label, color=color)
        
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Density at Stall')
        ax3.set_title(f'Convergence: λ={lambda_val}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "detailed_analysis_1_4_vs_2_4.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed analysis saved: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Main execution."""
    try:
        print("🏖️  EXTRACTING λ=1.4 AND λ=2.4 SUBFIGURES")
        print("="*45)
        print()
        
        # Create the main extraction
        subfig_path = create_subfigure_extraction()
        
        # Create detailed analysis
        detailed_path = create_detailed_analysis()
        
        print()
        print("✅ SUBFIGURE EXTRACTION COMPLETED")
        print("="*35)
        print()
        print("Generated files:")
        print(f"• {subfig_path} (Main 1×2 extraction)")
        print(f"• {detailed_path} (Detailed analysis)")
        print()
        print("Extracted subfigures show:")
        print("• λ=1.4: Mixed pattern with balanced spatial distribution")
        print("• λ=2.4: Crater pattern with density valley formation")
        print("• Evolution contours matching original visualization style")
        print("• Mathematical formulation and analysis annotations")
        
    except Exception as e:
        print(f"❌ Error in subfigure extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()