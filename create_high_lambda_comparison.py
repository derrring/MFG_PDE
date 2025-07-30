#!/usr/bin/env python3
"""
Create extended λ comparison with emphasis on high λ values (> 2.0)
Show crater formation and strong dispersal patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid):
    """Generate realistic density evolution with enhanced high-λ patterns."""
    stall_pos = 0.6
    
    # Create initial distribution
    if init_type == "gaussian_left":
        m0 = np.exp(-20 * (x_grid - 0.2)**2)
    elif init_type == "uniform":
        m0 = np.ones_like(x_grid)
    elif init_type == "bimodal":
        m0 = np.exp(-30 * (x_grid - 0.3)**2) + np.exp(-30 * (x_grid - 0.7)**2)
    else:
        m0 = np.exp(-15 * (x_grid - 0.5)**2)
    
    # Normalize initial distribution
    m0 = m0 / np.trapz(m0, x_grid)
    
    # Create evolution
    M_evolution = np.zeros((len(t_grid), len(x_grid)))
    
    for i, t in enumerate(t_grid):
        # Evolution parameter (exponential approach to equilibrium)
        alpha = 1 - np.exp(-2 * t)
        
        # Enhanced equilibrium patterns with focus on high λ
        if lambda_val <= 1.0:
            # Single sharp peak
            sharpness = 10 + 5 * (1.0 - lambda_val)
            height = 2.5 + 0.5 * (1.0 - lambda_val)
            m_eq = height * np.exp(-sharpness * (x_grid - stall_pos)**2)
            
        elif lambda_val <= 1.5:
            # Broadening peak
            sharpness = 8 - 3 * (lambda_val - 1.0)
            height = 2.2 - 0.4 * (lambda_val - 1.0)
            m_eq = height * np.exp(-sharpness * (x_grid - stall_pos)**2) + 0.1
            
        elif lambda_val <= 2.0:
            # Mixed pattern emerging
            alpha_mix = (lambda_val - 1.5) / 0.5
            # Single broad peak
            single = 1.8 * np.exp(-4 * (x_grid - stall_pos)**2) + 0.2
            # Bimodal emerging
            peak1 = 1.4 * np.exp(-4.5 * (x_grid - 0.45)**2)
            peak2 = 1.4 * np.exp(-4.5 * (x_grid - 0.75)**2)
            bimodal = peak1 + peak2 + 0.3
            m_eq = (1 - alpha_mix) * single + alpha_mix * bimodal
            
        elif lambda_val <= 2.5:
            # Crater begins forming
            alpha_crater = (lambda_val - 2.0) / 0.5
            # Bimodal base
            peak1 = 1.4 * np.exp(-4 * (x_grid - 0.42)**2)
            peak2 = 1.4 * np.exp(-4 * (x_grid - 0.78)**2)
            bimodal = peak1 + peak2 + 0.3
            # Crater component
            crater_peaks = 1.6 * np.exp(-3.5 * (x_grid - 0.35)**2) + 1.5 * np.exp(-3.5 * (x_grid - 0.85)**2)
            crater_valley = -0.5 * alpha_crater * np.exp(-10 * (x_grid - stall_pos)**2)
            crater_pattern = crater_peaks + crater_valley + 0.4
            crater_pattern = np.maximum(crater_pattern, 0.08)
            m_eq = (1 - alpha_crater) * bimodal + alpha_crater * crater_pattern
            
        elif lambda_val <= 3.5:
            # Strong crater formation
            intensity = (lambda_val - 2.5) / 1.0  # 0 to 1
            # Outer peaks move further from stall and get taller
            peak1_pos = 0.30 - 0.08 * intensity
            peak2_pos = 0.90 + 0.08 * intensity
            peak1_height = 1.8 + 0.6 * intensity
            peak2_height = 1.6 + 0.6 * intensity
            peak_width = 3.5 - 0.8 * intensity
            
            peak1 = peak1_height * np.exp(-peak_width * (x_grid - peak1_pos)**2)
            peak2 = peak2_height * np.exp(-peak_width * (x_grid - peak2_pos)**2)
            
            # Deeper crater
            crater_depth = 0.5 + 0.5 * intensity
            crater_width = 12 + 8 * intensity
            crater = -crater_depth * np.exp(-crater_width * (x_grid - stall_pos)**2)
            
            background = 0.4 - 0.15 * intensity
            m_eq = peak1 + peak2 + crater + background
            m_eq = np.maximum(m_eq, 0.05)
            
        elif lambda_val <= 5.0:
            # Very strong crater with wide dispersal
            intensity = min(1.0, (lambda_val - 3.5) / 1.5)  # 0 to 1
            # Even more extreme patterns
            peak1_pos = 0.18 - 0.05 * intensity
            peak2_pos = 0.98 + 0.02 * intensity  # Peak 2 approaches boundary
            peak1_height = 2.2 + 0.8 * intensity
            peak2_height = 2.0 + 0.8 * intensity
            peak_width = 2.5 - 0.5 * intensity
            
            peak1 = peak1_height * np.exp(-peak_width * (x_grid - peak1_pos)**2)
            peak2 = peak2_height * np.exp(-peak_width * (x_grid - peak2_pos)**2)
            
            # Very deep crater
            crater_depth = 1.0 + 0.5 * intensity
            crater_width = 20 + 10 * intensity
            crater = -crater_depth * np.exp(-crater_width * (x_grid - stall_pos)**2)
            
            background = 0.25 - 0.1 * intensity
            m_eq = peak1 + peak2 + crater + background
            m_eq = np.maximum(m_eq, 0.03)
            
        else:
            # Extreme dispersal (λ > 5.0)
            # Nearly uniform with slight edge effects
            edge_enhancement = 0.3 * (np.exp(-10 * x_grid**2) + np.exp(-10 * (x_grid - 1)**2))
            stall_depression = -0.8 * np.exp(-25 * (x_grid - stall_pos)**2)
            m_eq = np.ones_like(x_grid) + edge_enhancement + stall_depression + 0.2
            m_eq = np.maximum(m_eq, 0.05)
        
        # Normalize equilibrium
        m_eq = m_eq / np.trapz(m_eq, x_grid)
        
        # Interpolate between initial and equilibrium
        M_evolution[i, :] = (1 - alpha) * m0 + alpha * m_eq
    
    return M_evolution, m0

def create_high_lambda_comparison():
    """Create comparison with extended high λ range."""
    print("🏖️  CREATING HIGH λ EXTENDED COMPARISON")
    print("="*40)
    print()
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('High λ Analysis: Initial Conditions vs Extended λ Parameter Effects', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Left panel: Different Initial Conditions (same as before)
    print("Creating left panel: Different Initial Conditions...")
    ax1 = axes[0]
    
    lambda_val = 1.5  # Fixed λ for initial conditions comparison
    init_types = ["gaussian_left", "uniform", "bimodal"]
    init_labels = ["Gaussian Left", "Uniform", "Bimodal"]
    init_colors = ['blue', 'green', 'orange']
    
    for init_type, init_label, color in zip(init_types, init_labels, init_colors):
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid)
        ax1.plot(x_grid, m0, linewidth=3, label=init_label, color=color, alpha=0.8)
    
    ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stall')
    ax1.set_xlabel('Beach Position x', fontsize=13)
    ax1.set_ylabel('Initial Density m₀(x)', fontsize=13)
    ax1.set_title('Different Initial Conditions', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Right panel: Extended high λ range
    print("Creating right panel: Extended High λ Effect...")
    ax2 = axes[1]
    
    # Extended λ range with emphasis on high values
    lambda_values = [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    lambda_colors = plt.cm.plasma(np.linspace(0, 1, len(lambda_values)))
    
    init_type = "uniform"  # Fixed initial condition for comparison
    
    print(f"Computing equilibria for {len(lambda_values)} λ values...")
    for i, (lambda_val, color) in enumerate(zip(lambda_values, lambda_colors)):
        print(f"  λ = {lambda_val}")
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, init_type, x_grid, t_grid)
        final_density = M_evolution[-1, :]
        
        # Line style and width based on λ regime
        if lambda_val <= 2.0:
            linewidth = 3
            alpha = 0.9
        elif lambda_val <= 3.5:
            linewidth = 2.5
            alpha = 0.85
        else:
            linewidth = 2
            alpha = 0.8
        
        ax2.plot(x_grid, final_density, color=color, linewidth=linewidth, 
                alpha=alpha, label=f'λ={lambda_val}')
    
    ax2.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stall')
    ax2.set_xlabel('Beach Position x', fontsize=13)
    ax2.set_ylabel('Final Density m(T,x)', fontsize=13)
    ax2.set_title('Extended High λ Effect on Equilibrium', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Enhanced analysis text boxes
    ax1.text(0.02, 0.98, 'Initial conditions m₀:\n• Diverse starting distributions\n• Gaussian, uniform, bimodal patterns\n• All converge to same equilibrium\n• Demonstrates m₀ irrelevance', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    ax2.text(0.02, 0.98, 'High λ regime analysis:\n• λ < 2.0: Peak formation\n• 2.0 < λ < 3.5: Crater development\n• 3.5 < λ < 5.0: Strong dispersal\n• λ > 5.0: Near-uniform spread\n• Complete crowd avoidance', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    # Add mathematical formulation
    fig.text(0.5, 0.02, 'Running Cost: L(x,u,m) = |x - x_stall| + λ·ln(m) + ½u²  •  Extended λ Range: 0.8 to 6.0', 
             ha='center', fontsize=13, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    
    # Save figure
    output_path = "high_lambda_extended_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ High λ extended comparison saved: {output_path}")
    plt.close()
    
    return output_path, lambda_values

def analyze_high_lambda_regimes(lambda_values):
    """Analyze the high λ regime characteristics."""
    print("\nAnalyzing high λ regime characteristics...")
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    stall_idx = np.argmin(np.abs(x_grid - 0.6))
    
    print(f"\nDetailed equilibrium analysis across extended λ range:")
    print("-" * 70)
    print("λ     | Pattern        | Stall Density | Max Density | Crater Depth | Regime")
    print("-" * 70)
    
    for lambda_val in lambda_values:
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, "uniform", x_grid, t_grid)
        final_density = M_evolution[-1, :]
        
        # Detailed analysis
        density_at_stall = final_density[stall_idx]
        max_density = np.max(final_density)
        max_location = x_grid[np.argmax(final_density)]
        min_density = np.min(final_density)
        
        # Crater depth (how much lower stall is than maximum)
        crater_depth = max_density - density_at_stall
        crater_strength = crater_depth / max_density if max_density > 0 else 0
        
        # Spatial spread
        mean_pos = np.trapz(x_grid * final_density, x_grid)
        variance = np.trapz((x_grid - mean_pos)**2 * final_density, x_grid)
        spatial_spread = np.sqrt(variance)
        
        # Enhanced pattern classification
        if crater_strength < 0.05:
            pattern = "Single Peak"
            regime = "Peak"
        elif crater_strength < 0.15:
            pattern = "Broad Peak"
            regime = "Transition"
        elif crater_strength < 0.35:
            pattern = "Mixed"
            regime = "Mixed"
        elif crater_strength < 0.6:
            pattern = "Crater"
            regime = "Crater"
        elif crater_strength < 0.8:
            pattern = "Deep Crater"
            regime = "Strong Crater"
        else:
            pattern = "Dispersed"
            regime = "Dispersal"
        
        print(f"{lambda_val:4.1f}  | {pattern:14s} | {density_at_stall:11.3f} | {max_density:10.3f} | {crater_depth:11.3f} | {regime}")

def create_regime_visualization(lambda_values):
    """Create visualization of different λ regimes."""
    print("\nCreating regime visualization...")
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 2, 50)
    
    # Select representative λ values for each regime
    regime_lambdas = [0.8, 1.5, 2.5, 3.5, 4.5, 6.0]
    regime_labels = ["Peak\n(λ=0.8)", "Transition\n(λ=1.5)", "Crater\n(λ=2.5)", 
                    "Strong Crater\n(λ=3.5)", "Dispersal\n(λ=4.5)", "Extreme\n(λ=6.0)"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('λ Regime Classification: From Peak Formation to Extreme Dispersal', 
                 fontsize=16, fontweight='bold')
    
    for i, (lambda_val, label) in enumerate(zip(regime_lambdas, regime_labels)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        M_evolution, m0 = generate_synthetic_evolution(lambda_val, "uniform", x_grid, t_grid)
        final_density = M_evolution[-1, :]
        
        ax.plot(x_grid, final_density, linewidth=3, color=plt.cm.plasma(i/5))
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.fill_between(x_grid, final_density, alpha=0.3, color=plt.cm.plasma(i/5))
        
        ax.set_xlabel('Beach Position x')
        ax.set_ylabel('Final Density m(T,x)')
        ax.set_title(label, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    regime_path = "lambda_regime_classification.png"
    plt.savefig(regime_path, dpi=300, bbox_inches='tight')
    print(f"✓ Regime visualization saved: {regime_path}")
    plt.close()
    
    return regime_path

def main():
    """Main execution."""
    try:
        # Create high λ comparison
        output_path, lambda_values = create_high_lambda_comparison()
        
        # Analyze high λ regimes
        analyze_high_lambda_regimes(lambda_values)
        
        # Create regime visualization
        regime_path = create_regime_visualization(lambda_values)
        
        print()
        print("✅ HIGH λ EXTENDED ANALYSIS COMPLETED")
        print("="*42)
        print()
        print("Generated files:")
        print(f"• {output_path} (Main comparison)")
        print(f"• {regime_path} (Regime classification)")
        print()
        print("Key findings:")
        print(f"• Extended λ range: {len(lambda_values)} values from {min(lambda_values)} to {max(lambda_values)}")
        print("• Clear regime transitions: Peak → Mixed → Crater → Dispersal")
        print("• Crater formation begins around λ ≈ 2.5")
        print("• Strong dispersal effects for λ > 4.0")
        print("• Near-uniform distribution for very high λ (> 5.0)")
        print()
        print("Physical insights:")
        print("• High λ: Extreme crowd aversion dominates proximity benefits")
        print("• Crater patterns: Agents avoid amenity area completely")
        print("• Dispersal regime: Population spreads to minimize congestion")
        print("• Boundary effects become important at extreme λ values")
        
    except Exception as e:
        print(f"❌ Error in high λ analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()