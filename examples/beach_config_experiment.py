#!/usr/bin/env python3
"""
Beach Problem Configuration Experiment using OmegaConf.

This example demonstrates practical usage of the OmegaConf configuration
system for running parameter sweeps on the Towel on Beach problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from mfg_pde.config import create_omega_manager, OMEGACONF_AVAILABLE


class BeachConfigExperiment:
    """
    Beach problem experiment using OmegaConf configuration management.
    """
    
    def __init__(self, config_dir: Path = None):
        """Initialize experiment with configuration directory."""
        if not OMEGACONF_AVAILABLE:
            raise ImportError("OmegaConf required for this experiment")
        
        self.config_manager = create_omega_manager(config_dir)
        self.results = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for experiment."""
        logger = logging.getLogger("BeachConfigExperiment")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_lambda_sweep_experiment(self) -> DictConfig:
        """Create experiment configuration for lambda parameter sweep."""
        
        # Load base experiment configuration
        exp_config = self.config_manager.load_config("experiment.yaml")
        
        # Customize for lambda sweep
        exp_config.experiment.name = "beach_lambda_sweep"
        exp_config.experiment.description = "Crowd aversion parameter sweep for beach problem"
        
        # Configure parameter sweep
        exp_config.sweeps.lambda_sweep = {
            "description": "Extended lambda sweep with high values",
            "parameter": "problem.parameters.crowd_aversion",
            "values": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        }
        
        # Configure analysis
        exp_config.analysis.metrics = {
            "equilibrium_classification": True,
            "crater_depth": True,
            "spatial_spread": True,
            "density_at_stall": True,
            "peak_locations": True
        }
        
        return exp_config
    
    def create_multi_param_experiment(self) -> DictConfig:
        """Create multi-parameter experiment configuration."""
        
        # Compose base configuration
        config = self.config_manager.compose_config(
            "base_mfg.yaml",
            "beach_problem.yaml",
            "solver.yaml",
            "experiment.yaml"
        )
        
        # Configure multi-dimensional sweep
        config.experiment.name = "beach_multi_param_sweep"
        config.experiment.description = "Multi-parameter sweep: lambda vs initial conditions"
        
        config.multi_sweeps.comprehensive_sweep = {
            "description": "Comprehensive parameter space exploration",
            "parameters": {
                "lambda": [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                "init_type": ["gaussian_left", "uniform", "bimodal", "gaussian_right"],
                "stall_position": [0.4, 0.5, 0.6, 0.7, 0.8]
            }
        }
        
        return config
    
    def simulate_beach_evolution(self, config: DictConfig) -> Dict[str, Any]:
        """
        Simulate beach problem evolution (synthetic for demonstration).
        
        In a real implementation, this would use the actual MFG solver.
        """
        # Extract parameters
        lambda_val = config.problem.parameters.crowd_aversion
        stall_pos = config.problem.parameters.stall_position
        init_type = config.problem.initial_condition.type
        
        # Create synthetic results (replace with actual solver)
        x_grid = np.linspace(0, 1, config.problem.Nx)
        t_grid = np.linspace(0, config.problem.T, config.problem.Nt)
        
        # Generate synthetic final density based on lambda
        stall_idx = np.argmin(np.abs(x_grid - stall_pos))
        
        if lambda_val <= 1.0:
            # Single peak
            final_density = 2.0 * np.exp(-8 * (x_grid - stall_pos)**2)
        elif lambda_val <= 2.5:
            # Mixed pattern
            peak1 = 1.2 * np.exp(-5 * (x_grid - (stall_pos - 0.15))**2)
            peak2 = 1.3 * np.exp(-5 * (x_grid - (stall_pos + 0.15))**2)
            final_density = peak1 + peak2 + 0.3
        else:
            # Crater pattern
            peak1 = 1.5 * np.exp(-3 * (x_grid - 0.3)**2)
            peak2 = 1.4 * np.exp(-3 * (x_grid - 0.9)**2)
            crater = -0.4 * np.exp(-10 * (x_grid - stall_pos)**2)
            final_density = peak1 + peak2 + crater + 0.4
            final_density = np.maximum(final_density, 0.05)
        
        # Normalize
        final_density = final_density / np.trapz(final_density, x_grid)
        
        # Compute metrics
        density_at_stall = final_density[stall_idx]
        max_density = np.max(final_density)
        crater_depth = max_density - density_at_stall
        
        mean_pos = np.trapz(x_grid * final_density, x_grid)
        variance = np.trapz((x_grid - mean_pos)**2 * final_density, x_grid)
        spatial_spread = np.sqrt(variance)
        
        # Classify equilibrium
        crater_strength = crater_depth / max_density if max_density > 0 else 0
        if crater_strength < 0.1:
            eq_type = "Single Peak"
        elif crater_strength < 0.4:
            eq_type = "Mixed"
        else:
            eq_type = "Crater"
        
        return {
            'x_grid': x_grid,
            'final_density': final_density,
            'density_at_stall': density_at_stall,
            'max_density': max_density,
            'crater_depth': crater_depth,
            'spatial_spread': spatial_spread,
            'equilibrium_type': eq_type,
            'parameters': {
                'lambda': lambda_val,
                'stall_position': stall_pos,
                'init_type': init_type
            }
        }
    
    def run_parameter_sweep(self, exp_config: DictConfig) -> List[Dict[str, Any]]:
        """Run parameter sweep experiment."""
        
        self.logger.info(f"Starting experiment: {exp_config.experiment.name}")
        self.logger.info(f"Description: {exp_config.experiment.description}")
        
        # Extract sweep parameters
        if 'lambda_sweep' in exp_config.sweeps:
            sweep_config = exp_config.sweeps.lambda_sweep
            param_values = sweep_config.values
            param_name = "lambda"
        else:
            raise ValueError("No lambda sweep configuration found")
        
        results = []
        
        for i, param_value in enumerate(param_values):
            self.logger.info(f"Running configuration {i+1}/{len(param_values)}: λ={param_value}")
            
            # Create configuration for this parameter value
            run_config = self.config_manager.compose_config(
                "beach_problem.yaml",
                "solver.yaml"
            )
            
            # Set parameter value
            run_config.problem.parameters.crowd_aversion = param_value
            run_config.problem.initial_condition.type = "uniform"  # Fixed for sweep
            
            # Run simulation
            result = self.simulate_beach_evolution(run_config)
            results.append(result)
            
            self.logger.info(f"   Result: {result['equilibrium_type']}, "
                           f"crater_depth={result['crater_depth']:.3f}")
        
        self.logger.info(f"Completed parameter sweep with {len(results)} runs")
        return results
    
    def create_analysis_plots(self, results: List[Dict[str, Any]], 
                            output_dir: Path = None) -> List[Path]:
        """Create analysis plots from experiment results."""
        
        if output_dir is None:
            output_dir = Path("beach_config_results")
        output_dir.mkdir(exist_ok=True)
        
        saved_plots = []
        
        # Extract data for plotting
        lambda_values = [r['parameters']['lambda'] for r in results]
        crater_depths = [r['crater_depth'] for r in results]
        spatial_spreads = [r['spatial_spread'] for r in results]
        eq_types = [r['equilibrium_type'] for r in results]
        
        # Plot 1: Final densities for selected lambda values
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        selected_indices = [0, 3, 6, 9, 12]  # Select subset for clarity
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
        
        for i, idx in enumerate(selected_indices):
            if idx < len(results):
                result = results[idx]
                x_grid = result['x_grid']
                density = result['final_density']
                lambda_val = result['parameters']['lambda']
                
                ax1.plot(x_grid, density, linewidth=2, color=colors[i], 
                        label=f'λ={lambda_val}', alpha=0.8)
        
        ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Stall')
        ax1.set_xlabel('Beach Position x')
        ax1.set_ylabel('Final Density m(T,x)')
        ax1.set_title('Final Density Evolution with λ Parameter')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plot1_path = output_dir / "final_densities.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot1_path)
        plt.close()
        
        # Plot 2: Parameter analysis
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('Parameter Sweep Analysis', fontsize=16, fontweight='bold')
        
        # Crater depth vs lambda
        ax = axes[0, 0]
        ax.plot(lambda_values, crater_depths, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Crowd Aversion λ')
        ax.set_ylabel('Crater Depth')
        ax.set_title('Crater Formation vs λ')
        ax.grid(True, alpha=0.3)
        
        # Spatial spread vs lambda
        ax = axes[0, 1]
        ax.plot(lambda_values, spatial_spreads, 's-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Crowd Aversion λ')
        ax.set_ylabel('Spatial Spread')
        ax.set_title('Spatial Distribution vs λ')
        ax.grid(True, alpha=0.3)
        
        # Equilibrium type classification
        ax = axes[1, 0]
        eq_type_counts = {eq_type: eq_types.count(eq_type) for eq_type in set(eq_types)}
        ax.bar(eq_type_counts.keys(), eq_type_counts.values(), 
               color=['lightblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('Number of Configurations')
        ax.set_title('Equilibrium Type Distribution')
        
        # Parameter space visualization
        ax = axes[1, 1]
        scatter = ax.scatter(lambda_values, crater_depths, c=spatial_spreads, 
                           s=60, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Crowd Aversion λ')
        ax.set_ylabel('Crater Depth')
        ax.set_title('Parameter Space (color = spatial spread)')
        plt.colorbar(scatter, ax=ax, label='Spatial Spread')
        
        plt.tight_layout()
        
        plot2_path = output_dir / "parameter_analysis.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot2_path)
        plt.close()
        
        self.logger.info(f"Analysis plots saved to: {output_dir}")
        return saved_plots
    
    def save_experiment_results(self, exp_config: DictConfig, results: List[Dict[str, Any]], 
                              output_dir: Path = None) -> Path:
        """Save experiment configuration and results."""
        
        if output_dir is None:
            output_dir = Path("beach_config_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config_path = output_dir / "experiment_config.yaml"
        self.config_manager.save_config(exp_config, config_path)
        
        # Save results summary
        summary_path = output_dir / "results_summary.yaml"
        
        summary_data = {
            'experiment': {
                'name': exp_config.experiment.name,
                'description': exp_config.experiment.description,
                'total_runs': len(results)
            },
            'parameter_range': {
                'lambda_min': min(r['parameters']['lambda'] for r in results),
                'lambda_max': max(r['parameters']['lambda'] for r in results),
                'lambda_values': [r['parameters']['lambda'] for r in results]
            },
            'equilibrium_summary': {},
            'metrics_summary': {
                'crater_depth': {
                    'min': min(r['crater_depth'] for r in results),
                    'max': max(r['crater_depth'] for r in results),
                    'mean': np.mean([r['crater_depth'] for r in results])
                },
                'spatial_spread': {
                    'min': min(r['spatial_spread'] for r in results),
                    'max': max(r['spatial_spread'] for r in results),
                    'mean': np.mean([r['spatial_spread'] for r in results])
                }
            }
        }
        
        # Count equilibrium types
        eq_types = [r['equilibrium_type'] for r in results]
        for eq_type in set(eq_types):
            summary_data['equilibrium_summary'][eq_type] = eq_types.count(eq_type)
        
        OmegaConf.save(OmegaConf.create(summary_data), summary_path)
        
        self.logger.info(f"Experiment results saved to: {output_dir}")
        return output_dir


def main():
    """Run beach configuration experiment."""
    print("🏖️  BEACH PROBLEM CONFIGURATION EXPERIMENT")
    print("="*45)
    
    if not OMEGACONF_AVAILABLE:
        print("❌ OmegaConf not available. Install with: pip install omegaconf")
        return
    
    try:
        # Create experiment
        experiment = BeachConfigExperiment()
        
        # Create lambda sweep experiment configuration
        print("\n1. Creating lambda sweep experiment configuration...")
        exp_config = experiment.create_lambda_sweep_experiment()
        print(f"   Experiment: {exp_config.experiment.name}")
        print(f"   Parameter values: {exp_config.sweeps.lambda_sweep.values}")
        
        # Run parameter sweep
        print("\n2. Running parameter sweep...")
        results = experiment.run_parameter_sweep(exp_config)
        
        # Create analysis
        print("\n3. Creating analysis plots...")
        plot_paths = experiment.create_analysis_plots(results)
        
        # Save results
        print("\n4. Saving experiment results...")
        output_dir = experiment.save_experiment_results(exp_config, results)
        
        print("\n✅ EXPERIMENT COMPLETED")
        print("="*22)
        print(f"Results saved to: {output_dir}")
        print(f"Generated plots: {[p.name for p in plot_paths]}")
        print()
        print("Configuration management features demonstrated:")
        print("• YAML-based experiment configuration")
        print("• Parameter sweep generation with interpolation")
        print("• Automatic result organization and saving")
        print("• Structured analysis and visualization")
        print("• Configuration validation and error handling")
        
    except Exception as e:
        print(f"❌ Error in beach configuration experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()