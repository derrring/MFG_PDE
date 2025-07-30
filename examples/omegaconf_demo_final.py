#!/usr/bin/env python3
"""
Final demonstration of OmegaConf configuration management in MFG_PDE.

This script demonstrates the complete configuration management system with:
- YAML-based configurations
- Parameter overrides and composition
- Integration with existing Pydantic configs
- Parameter sweep generation
"""

import sys
from pathlib import Path
import logging

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mfg_pde.config import create_omega_manager, OMEGACONF_AVAILABLE
    from mfg_pde.config import create_fast_config  # Existing Pydantic config
except ImportError as e:
    print(f"Import error: {e}")
    OMEGACONF_AVAILABLE = False


def demo_basic_functionality():
    """Demonstrate basic OmegaConf functionality."""
    print("🔧 BASIC OMEGACONF FUNCTIONALITY")
    print("="*35)
    
    if not OMEGACONF_AVAILABLE:
        print("❌ OmegaConf not available")
        return
    
    # Create configuration manager
    manager = create_omega_manager()
    print("✓ Configuration manager created")
    print(f"  Config directory: {manager.config_dir}")
    
    # Load individual configurations
    print("\n1. Loading individual configurations:")
    base_config = manager.load_config('base_mfg.yaml')
    solver_config = manager.load_config('solver.yaml')
    
    print(f"   Base MFG: T={base_config.problem.T}, Nx={base_config.problem.Nx}")
    print(f"   Solver: {solver_config.solver.type}, max_iter={solver_config.solver.max_iterations}")
    
    # Configuration composition
    print("\n2. Configuration composition:")
    composed = manager.compose_config('base_mfg.yaml', 'solver.yaml')
    print(f"   Composed config keys: {list(composed.keys())}")
    
    # Parameter overrides
    print("\n3. Parameter overrides:")
    override_config = manager.load_config(
        'base_mfg.yaml',
        problem={'Nx': 100, 'Nt': 50, 'T': 2.0}
    )
    print(f"   Override: T={override_config.problem.T}, Nx={override_config.problem.Nx}")
    
    return manager


def demo_parameter_sweeps(manager):
    """Demonstrate parameter sweep generation."""
    print("\n📊 PARAMETER SWEEP GENERATION")
    print("="*32)
    
    # Create base configuration for sweeps
    base_config = manager.compose_config('base_mfg.yaml', 'solver.yaml')
    
    # Define parameter sweep
    sweep_params = {
        'problem.Nx': [40, 80, 160],
        'solver.max_iterations': [50, 100, 200],
        'solver.tolerance': [1e-5, 1e-6, 1e-7]
    }
    
    print(f"Creating parameter sweep with {len(sweep_params)} parameters:")
    for param, values in sweep_params.items():
        print(f"  {param}: {values}")
    
    # Generate sweep configurations
    sweep_configs = manager.create_parameter_sweep(base_config, sweep_params)
    
    total_configs = len(sweep_configs)
    print(f"\n✓ Generated {total_configs} parameter combinations")
    
    # Show sample configurations
    print("\nSample configurations:")
    for i, config in enumerate(sweep_configs[:3]):
        params = config.experiment.current_params
        print(f"  Config {i+1}: Nx={params['problem.Nx']}, "
              f"max_iter={params['solver.max_iterations']}, "
              f"tol={params['solver.tolerance']:.0e}")
    
    if total_configs > 3:
        print(f"  ... and {total_configs - 3} more configurations")
    
    return sweep_configs


def demo_configuration_validation(manager):
    """Demonstrate configuration validation."""
    print("\n✅ CONFIGURATION VALIDATION")
    print("="*29)
    
    # Valid configuration
    valid_config = manager.compose_config('base_mfg.yaml', 'solver.yaml')
    is_valid = manager.validate_config(valid_config)
    print(f"Valid configuration: {is_valid}")
    
    # Test with missing required field (simulate)
    try:
        incomplete_config = manager.load_config('base_mfg.yaml')
        # Remove required solver section
        del incomplete_config['problem']  # This should make it invalid
        is_valid = manager.validate_config(incomplete_config)
        print(f"Invalid configuration detected: {not is_valid}")
    except Exception as e:
        print(f"Configuration validation caught error: {type(e).__name__}")


def demo_config_saving(manager):
    """Demonstrate configuration saving and templates."""
    print("\n💾 CONFIGURATION SAVING & TEMPLATES")
    print("="*38)
    
    # Get configuration template
    template = manager.get_config_template('base_mfg')
    print("✓ Retrieved configuration template")
    
    # Create custom configuration
    custom_config = manager.compose_config(
        'base_mfg.yaml', 
        'solver.yaml',
        problem={'name': 'custom_experiment', 'T': 3.0, 'Nx': 200}
    )
    
    # Save configuration
    output_dir = Path('demo_configs')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'custom_experiment.yaml'
    manager.save_config(custom_config, output_path)
    print(f"✓ Custom configuration saved: {output_path}")
    
    # Verify saved configuration can be loaded (use absolute path)
    try:
        loaded_custom = manager.load_config(str(output_path.absolute()))
        print(f"✓ Saved configuration reloaded: {loaded_custom.problem.name}")
    except Exception as e:
        print(f"⚠ Note: Saved config verification skipped: {e}")


def demo_pydantic_integration(manager):
    """Demonstrate integration with existing Pydantic configurations."""
    print("\n🔗 PYDANTIC INTEGRATION")
    print("="*23)
    
    # Load OmegaConf configuration
    omega_config = manager.compose_config('solver.yaml')
    
    # Convert to Pydantic (with fallback)
    try:
        pydantic_config = manager.create_pydantic_config(omega_config)
        print("✓ OmegaConf → Pydantic conversion successful")
        print(f"  Pydantic config: {type(pydantic_config).__name__}")
        print(f"  Max iterations: {pydantic_config.max_iterations}")
    except Exception as e:
        print(f"⚠ Conversion note: {e}")
    
    # Compare with direct Pydantic creation
    direct_pydantic = create_fast_config()
    print("✓ Direct Pydantic configuration created")
    print(f"  Direct config type: {type(direct_pydantic).__name__}")
    
    print("\nBoth systems can coexist and complement each other!")
    print("OmegaConf provides YAML-based config management")
    print("Pydantic provides type validation and data modeling")


def create_summary_report(manager, sweep_configs):
    """Create a summary report of the configuration system."""
    print("\n📋 CONFIGURATION SYSTEM SUMMARY")
    print("="*35)
    
    import os
    config_files = os.listdir(manager.config_dir)
    
    print("Configuration Management Features:")
    print("  ✓ YAML-based configuration files")
    print("  ✓ Hierarchical configuration composition")
    print("  ✓ Parameter interpolation and overrides")
    print("  ✓ Automatic parameter sweep generation")
    print("  ✓ Configuration validation and error checking")
    print("  ✓ Integration with existing Pydantic configs")
    print("  ✓ Template-based configuration creation")
    print("  ✓ Configuration saving and loading")
    
    print(f"\nSystem Statistics:")
    print(f"  Available config files: {len(config_files)}")
    print(f"  Config files: {sorted(config_files)}")
    print(f"  Generated sweep configs: {len(sweep_configs) if sweep_configs else 0}")
    
    print(f"\nUsage Examples:")
    print("  # Load configuration with overrides:")
    print("  config = manager.load_config('solver.yaml', max_iterations=200)")
    print("  ")
    print("  # Compose multiple configurations:")
    print("  config = manager.compose_config('base_mfg.yaml', 'solver.yaml')")
    print("  ")
    print("  # Generate parameter sweep:")
    print("  sweeps = manager.create_parameter_sweep(base_config, param_dict)")


def main():
    """Run complete OmegaConf demonstration."""
    print("🏗️ MFG_PDE OMEGACONF CONFIGURATION SYSTEM")
    print("=" * 45)
    
    if not OMEGACONF_AVAILABLE:
        print("\n❌ OmegaConf not available. Install with:")
        print("pip install omegaconf")
        return
    
    try:
        print("Demonstrating comprehensive configuration management...")
        
        # Basic functionality
        manager = demo_basic_functionality()
        
        # Parameter sweeps
        sweep_configs = demo_parameter_sweeps(manager)
        
        # Validation
        demo_configuration_validation(manager)
        
        # Saving and templates
        demo_config_saving(manager)
        
        # Pydantic integration
        demo_pydantic_integration(manager)
        
        # Summary report
        create_summary_report(manager, sweep_configs)
        
        print("\n✅ OMEGACONF DEMONSTRATION COMPLETED")
        print("="*38)
        print("\nThe MFG_PDE configuration management system is ready!")
        print("Key benefits:")
        print("• Structured YAML-based configuration")
        print("• Easy parameter sweeps for experiments")
        print("• Seamless integration with existing code")
        print("• Robust validation and error handling")
        print("• Professional experiment management")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()