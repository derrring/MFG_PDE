#!/usr/bin/env python3
"""
Demonstration of OmegaConf configuration management in MFG_PDE.

This example shows how to:
1. Load and compose configurations from YAML files
2. Perform parameter sweeps with interpolation
3. Integrate with existing Pydantic configs
4. Run experiments with structured configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("OmegaConf not available. Install with: pip install omegaconf")

from mfg_pde.config.omegaconf_manager import (
    OmegaConfManager, 
    create_omega_manager,
    load_beach_config,
    create_parameter_sweep_configs
)
from mfg_pde.config import create_fast_config


def demo_basic_config_loading():
    """Demonstrate basic configuration loading and composition."""
    print("🔧 BASIC CONFIGURATION LOADING")
    print("="*40)
    
    if not OMEGACONF_AVAILABLE:
        print("Skipping OmegaConf demos - package not available")
        return
    
    # Create configuration manager
    manager = create_omega_manager()
    
    # Load individual configurations
    print("\n1. Loading individual configurations:")
    beach_config = manager.load_config("beach_problem.yaml")
    solver_config = manager.load_config("solver.yaml")
    
    print(f"   Beach problem: {beach_config.problem.name}")
    print(f"   Grid size: {beach_config.problem.Nx} × {beach_config.problem.Nt}")
    print(f"   Solver type: {solver_config.solver.type}")
    print(f"   Max iterations: {solver_config.solver.max_iterations}")
    
    # Compose configurations
    print("\n2. Composing configurations:")
    composed = manager.compose_config(
        "base_mfg.yaml",
        "beach_problem.yaml", 
        "solver.yaml"
    )
    
    print(f"   Composed config has keys: {list(composed.keys())}")
    
    # Configuration with overrides
    print("\n3. Configuration with overrides:")
    beach_with_overrides = manager.load_config(
        "beach_problem.yaml",
        problem={"Nx": 100, "Nt": 50},
        lambda_=2.5,  # Parameter interpolation
        init_type="bimodal"
    )
    
    print(f"   Override grid: {beach_with_overrides.problem.Nx} × {beach_with_overrides.problem.Nt}")
    print(f"   Lambda value: {beach_with_overrides.problem.parameters.crowd_aversion}")


def demo_parameter_interpolation():
    """Demonstrate parameter interpolation and templating."""
    print("\n🔄 PARAMETER INTERPOLATION")
    print("="*30)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    manager = create_omega_manager()
    
    # Create configuration with interpolated parameters
    config_variants = []
    lambda_values = [0.8, 1.5, 2.5, 3.5]
    init_types = ["gaussian_left", "uniform", "bimodal"]
    
    print(f"\nCreating {len(lambda_values)} × {len(init_types)} = {len(lambda_values) * len(init_types)} configuration variants:")
    
    for lambda_val in lambda_values:
        for init_type in init_types:
            config = manager.load_config(
                "beach_problem.yaml",
                lambda_=lambda_val,
                init_type=init_type
            )
            config_variants.append({
                'lambda': lambda_val,
                'init_type': init_type,
                'config': config
            })
            
            print(f"   λ={lambda_val}, init={init_type}: "
                  f"crowd_aversion={config.problem.parameters.crowd_aversion}, "
                  f"init_type={config.problem.initial_condition.type}")


def demo_parameter_sweeps():
    """Demonstrate parameter sweep configuration generation."""
    print("\n📊 PARAMETER SWEEP GENERATION")
    print("="*32)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    # Create parameter sweep configurations
    lambda_values = [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
    init_types = ["gaussian_left", "uniform", "bimodal"]
    
    print(f"Generating parameter sweep: {len(lambda_values)} λ values × {len(init_types)} initial types")
    
    sweep_configs = create_parameter_sweep_configs(lambda_values, init_types)
    
    print(f"Generated {len(sweep_configs)} configurations")
    
    # Display sample configurations
    print("\nSample configurations:")
    for i, config in enumerate(sweep_configs[:5]):  # Show first 5
        current_params = config.experiment.current_params
        print(f"   Config {i+1}: λ={current_params['problem.parameters.crowd_aversion']:.1f}, "
              f"init={current_params['problem.initial_condition.type']}")
    
    if len(sweep_configs) > 5:
        print(f"   ... and {len(sweep_configs) - 5} more")
    
    return sweep_configs


def demo_pydantic_integration():
    """Demonstrate integration with existing Pydantic configurations."""
    print("\n🔗 PYDANTIC INTEGRATION")
    print("="*23)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    manager = create_omega_manager()
    
    # Load OmegaConf configuration
    omega_config = manager.compose_config("solver.yaml", "beach_problem.yaml")
    
    print("1. OmegaConf → Pydantic conversion:")
    try:
        pydantic_config = manager.create_pydantic_config(omega_config)
        print(f"   Created Pydantic config: {type(pydantic_config).__name__}")
        print(f"   Max iterations: {pydantic_config.max_iterations}")
        print(f"   Tolerance: {pydantic_config.tolerance}")
    except Exception as e:
        print(f"   Conversion warning: {e}")
        print("   Using fallback configuration")
    
    # Compare with direct Pydantic creation
    print("\n2. Direct Pydantic configuration:")
    direct_pydantic = create_fast_config()
    print(f"   Direct config: {type(direct_pydantic).__name__}")
    print(f"   Max iterations: {direct_pydantic.max_iterations}")
    print(f"   Tolerance: {direct_pydantic.tolerance}")


def demo_experiment_configuration():
    """Demonstrate experiment configuration and management."""
    print("\n🧪 EXPERIMENT CONFIGURATION")
    print("="*29)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    manager = create_omega_manager()
    
    # Load experiment configuration
    exp_config = manager.load_config("experiment.yaml")
    
    print("Experiment configuration loaded:")
    print(f"   Name: {exp_config.experiment.name}")
    print(f"   Description: {exp_config.experiment.description}")
    print(f"   Output dir: {exp_config.experiment.output.experiment_dir}")
    print(f"   Logging level: {exp_config.experiment.logging.level}")
    print(f"   Visualization enabled: {exp_config.experiment.visualization.enabled}")
    
    # Show available parameter sweeps
    print(f"\nAvailable parameter sweeps:")
    for sweep_name, sweep_config in exp_config.sweeps.items():
        print(f"   {sweep_name}: {sweep_config.description}")
        if hasattr(sweep_config, 'values'):
            print(f"      Values: {sweep_config.values}")
    
    # Show multi-dimensional sweeps
    print(f"\nMulti-dimensional sweeps:")
    for sweep_name, sweep_config in exp_config.multi_sweeps.items():
        print(f"   {sweep_name}: {sweep_config.description}")
        for param, values in sweep_config.parameters.items():
            print(f"      {param}: {values}")


def demo_config_validation_and_saving():
    """Demonstrate configuration validation and saving."""
    print("\n✅ CONFIGURATION VALIDATION & SAVING")
    print("="*38)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    manager = create_omega_manager()
    
    # Create and validate configuration
    config = manager.compose_config("beach_problem.yaml", "solver.yaml")
    
    print("1. Configuration validation:")
    is_valid = manager.validate_config(config)
    print(f"   Configuration is valid: {is_valid}")
    
    # Save configuration
    output_dir = Path("tmp_config_demo")
    output_dir.mkdir(exist_ok=True)
    
    print("\n2. Saving configurations:")
    
    # Save composed configuration
    output_path = output_dir / "composed_config.yaml"
    manager.save_config(config, output_path)
    print(f"   Composed config saved to: {output_path}")
    
    # Save experiment-specific configuration
    exp_config = manager.load_config(
        "experiment.yaml",
        experiment={
            "name": "demo_experiment",
            "description": "Configuration management demonstration"
        }
    )
    
    exp_output_path = output_dir / "demo_experiment.yaml"
    manager.save_config(exp_config, exp_output_path)
    print(f"   Experiment config saved to: {exp_output_path}")
    
    # Show saved file contents (first few lines)
    if output_path.exists():
        print(f"\n3. Preview of saved configuration:")
        with open(output_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:10]):
                print(f"   {line.rstrip()}")
            if len(lines) > 10:
                print(f"   ... ({len(lines) - 10} more lines)")


def create_visualization_demo():
    """Create visualization of configuration system."""
    print("\n📈 CONFIGURATION SYSTEM VISUALIZATION")
    print("="*38)
    
    if not OMEGACONF_AVAILABLE:
        return
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MFG_PDE Configuration Management with OmegaConf', 
                 fontsize=16, fontweight='bold')
    
    # 1. Configuration hierarchy
    ax1 = axes[0, 0]
    hierarchy_levels = ['Base MFG', 'Problem Specific', 'Solver Config', 'Experiment Setup']
    hierarchy_counts = [1, 3, 2, 4]  # Number of config files at each level
    
    bars = ax1.bar(hierarchy_levels, hierarchy_counts, 
                   color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax1.set_title('Configuration Hierarchy')
    ax1.set_ylabel('Number of Config Files')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, hierarchy_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # 2. Parameter sweep example
    ax2 = axes[0, 1]
    lambda_vals = np.array([0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
    n_configs = len(lambda_vals) * 3  # 3 initial condition types
    
    ax2.plot(lambda_vals, [3] * len(lambda_vals), 'o-', linewidth=2, markersize=8,
             label='Configs per λ value')
    ax2.fill_between(lambda_vals, 0, 3, alpha=0.3)
    ax2.set_title(f'Parameter Sweep ({n_configs} total configs)')
    ax2.set_xlabel('Crowd Aversion λ')
    ax2.set_ylabel('Initial Condition Types')
    ax2.set_ylim(0, 4)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Configuration composition flow
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    # Create flow diagram
    flow_text = """Configuration Composition Flow:
    
    1. base_mfg.yaml
       ↓ (inherit)
    2. beach_problem.yaml  
       ↓ (compose)
    3. solver.yaml
       ↓ (override)
    4. experiment.yaml
       ↓ (interpolate)
    5. Final Configuration"""
    
    ax3.text(0.1, 0.9, flow_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 4. Feature comparison
    ax4 = axes[1, 1]
    features = ['YAML Files', 'Interpolation', 'Validation', 'Composition', 'Pydantic\nIntegration']
    omega_features = [1, 1, 1, 1, 1]  # All supported
    direct_features = [0, 0, 1, 0, 1]  # Only validation and pydantic
    
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, omega_features, width, label='OmegaConf System', 
                    color='lightgreen', alpha=0.8)
    bars2 = ax4.bar(x + width/2, direct_features, width, label='Direct Config', 
                    color='lightcoral', alpha=0.8)
    
    ax4.set_title('Configuration System Features')
    ax4.set_ylabel('Feature Availability')
    ax4.set_xticks(x)
    ax4.set_xticklabels(features, rotation=45, ha='right')
    ax4.set_ylim(0, 1.2)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "config_system_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Configuration system overview saved: {output_path}")
    plt.close()


def main():
    """Run all configuration management demonstrations."""
    print("🔧 MFG_PDE CONFIGURATION MANAGEMENT DEMO")
    print("="*45)
    
    try:
        # Basic configuration operations
        demo_basic_config_loading()
        demo_parameter_interpolation()
        
        # Advanced features
        sweep_configs = demo_parameter_sweeps()
        demo_pydantic_integration()
        demo_experiment_configuration()
        demo_config_validation_and_saving()
        
        # Create visualization
        create_visualization_demo()
        
        print("\n✅ CONFIGURATION DEMO COMPLETED")
        print("="*32)
        print()
        print("Key capabilities demonstrated:")
        print("• YAML-based configuration files with hierarchical structure")
        print("• Parameter interpolation and templating")
        print("• Automatic parameter sweep generation")
        print("• Integration with existing Pydantic configurations")
        print("• Configuration validation and error checking")
        print("• Experiment management and metadata tracking")
        print()
        
        if OMEGACONF_AVAILABLE and sweep_configs:
            print(f"Generated {len(sweep_configs)} parameter sweep configurations")
            print("Ready for large-scale MFG experiments with structured configuration!")
        else:
            print("Install OmegaConf to enable full configuration management:")
            print("pip install omegaconf")
            
    except Exception as e:
        print(f"❌ Error in configuration demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()