#!/usr/bin/env python3
"""
Command Line Interface Utilities for MFG_PDE

Provides argument parsing, configuration loading, and CLI tools for running
MFG solvers from the command line with professional argument handling.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sys
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def create_base_parser() -> argparse.ArgumentParser:
    """
    Create base argument parser with common MFG solver options.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="MFG_PDE: Mean Field Games Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For more information, visit: https://github.com/your-repo/MFG_PDE"
    )
    
    # Problem configuration
    problem_group = parser.add_argument_group('Problem Configuration')
    problem_group.add_argument(
        '--T', type=float, default=1.0,
        help='Terminal time for the MFG problem'
    )
    problem_group.add_argument(
        '--Nt', type=int, default=50,
        help='Number of time steps'
    )
    problem_group.add_argument(
        '--xmin', type=float, default=0.0,
        help='Minimum spatial domain value'
    )
    problem_group.add_argument(
        '--xmax', type=float, default=1.0,
        help='Maximum spatial domain value'
    )
    problem_group.add_argument(
        '--Nx', type=int, default=100,
        help='Number of spatial grid points'
    )
    
    # Solver configuration
    solver_group = parser.add_argument_group('Solver Configuration')
    solver_group.add_argument(
        '--solver-type', choices=['fixed_point', 'particle_collocation', 'monitored_particle', 'adaptive_particle'],
        default='monitored_particle',
        help='Type of solver to use'
    )
    solver_group.add_argument(
        '--preset', choices=['fast', 'balanced', 'accurate', 'research'],
        default='balanced',
        help='Configuration preset for the solver'
    )
    solver_group.add_argument(
        '--max-iterations', type=int, default=None,
        help='Maximum number of iterations (overrides preset)'
    )
    solver_group.add_argument(
        '--tolerance', type=float, default=None,
        help='Convergence tolerance (overrides preset)'
    )
    solver_group.add_argument(
        '--num-particles', type=int, default=None,
        help='Number of particles for particle methods'
    )
    
    # I/O options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file (JSON or YAML)'
    )
    io_group.add_argument(
        '--output', type=str, default=None,
        help='Output file path for results'
    )
    io_group.add_argument(
        '--save-config', type=str, default=None,
        help='Save current configuration to file'
    )
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    exec_group.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress all output except errors'
    )
    exec_group.add_argument(
        '--progress', action='store_true', default=True,
        help='Show progress bars (default: enabled)'
    )
    exec_group.add_argument(
        '--no-progress', dest='progress', action='store_false',
        help='Disable progress bars'
    )
    exec_group.add_argument(
        '--timing', action='store_true', default=True,
        help='Show timing information (default: enabled)'
    )
    exec_group.add_argument(
        '--no-timing', dest='timing', action='store_false',
        help='Disable timing information'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--warm-start', action='store_true',
        help='Enable warm start capability'
    )
    advanced_group.add_argument(
        '--return-structured', action='store_true', default=True,
        help='Return structured results (default: enabled)'
    )
    advanced_group.add_argument(
        '--profile', action='store_true',
        help='Enable performance profiling'
    )
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'r') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support requires PyYAML: pip install PyYAML")
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")


def save_config_file(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    
    try:
        with open(output_path, 'w') as f:
            if suffix == '.json':
                json.dump(config, f, indent=2, default=str)
            elif suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support requires PyYAML: pip install PyYAML")
                yaml.safe_dump(config, f, indent=2, default_flow_style=False)
            else:
                # Default to JSON
                json.dump(config, f, indent=2, default=str)
        
        print(f"✅ Configuration saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving config file: {e}")


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert argparse Namespace to configuration dictionary.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary suitable for MFG solver
    """
    config = {}
    
    # Problem parameters
    config['problem'] = {
        'T': args.T,
        'Nt': args.Nt,
        'xmin': args.xmin,
        'xmax': args.xmax,
        'Nx': args.Nx
    }
    
    # Solver parameters
    config['solver'] = {
        'type': args.solver_type,
        'preset': args.preset,
        'warm_start': args.warm_start,
        'return_structured': args.return_structured
    }
    
    # Override parameters if specified
    if args.max_iterations is not None:
        config['solver']['max_iterations'] = args.max_iterations
    if args.tolerance is not None:
        config['solver']['tolerance'] = args.tolerance
    if args.num_particles is not None:
        config['solver']['num_particles'] = args.num_particles
    
    # Execution parameters
    config['execution'] = {
        'verbose': args.verbose and not args.quiet,
        'progress': args.progress and not args.quiet,
        'timing': args.timing and not args.quiet,
        'profile': args.profile
    }
    
    # I/O parameters
    config['io'] = {
        'output': args.output,
        'save_config': args.save_config
    }
    
    return config


def create_solver_cli() -> argparse.ArgumentParser:
    """
    Create a comprehensive CLI for MFG solver execution.
    
    Returns:
        Configured ArgumentParser for solver CLI
    """
    parser = create_base_parser()
    
    # Add solver-specific subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve an MFG problem')
    solve_parser.add_argument(
        'problem_file', type=str, default=None, nargs='?',
        help='Python file defining the MFG problem class'
    )
    solve_parser.add_argument(
        '--problem-class', type=str, default='MFGProblem',
        help='Name of the problem class to instantiate'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command')
    
    # Generate config
    gen_config = config_subparsers.add_parser('generate', help='Generate sample configuration')
    gen_config.add_argument('output_file', help='Output configuration file')
    gen_config.add_argument('--format', choices=['json', 'yaml'], default='json')
    
    # Validate config
    val_config = config_subparsers.add_parser('validate', help='Validate configuration file')
    val_config.add_argument('config_file', help='Configuration file to validate')
    
    return parser


def run_solver_from_cli(args: argparse.Namespace) -> None:
    """
    Run MFG solver based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    # Convert args to configuration
    config = args_to_config(args)
    
    # Load and merge config file if provided
    if args.config:
        try:
            file_config = load_config_file(args.config)
            config = merge_configs(file_config, config)
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)
    
    # Set up logging level
    if config['execution']['verbose']:
        print("🚀 Starting MFG solver with configuration:")
        print(json.dumps(config, indent=2, default=str))
    
    try:
        # Import here to avoid circular imports
        from ..factory import create_solver
        from .. import MFGProblem
        
        # Create problem instance
        # This is a simplified example - in practice, you'd load from problem_file
        class CLIMFGProblem(MFGProblem):
            def __init__(self, T, Nt, xmin, xmax, Nx):
                super().__init__(T=T, Nt=Nt, xmin=xmin, xmax=xmax, Nx=Nx)
            
            def g(self, x):
                return 0.5 * (x - 0.5)**2
            
            def rho0(self, x):
                import numpy as np
                return np.exp(-10 * (x - 0.3)**2)
            
            def f(self, x, u, m):
                return 0.1 * u**2 + 0.05 * m
            
            def sigma(self, x):
                return 0.1
            
            def H(self, x, p, m):
                return 0.5 * p**2
            
            def dH_dm(self, x, p, m):
                return 0.0
        
        # Create problem
        problem_config = config['problem']
        problem = CLIMFGProblem(**problem_config)
        
        # Create solver
        solver_config = config['solver']
        import numpy as np
        collocation_points = np.linspace(problem.xmin, problem.xmax, problem.Nx).reshape(-1, 1)
        
        solver = create_solver(
            problem=problem,
            solver_type=solver_config['type'],
            preset=solver_config['preset'],
            collocation_points=collocation_points,
            **{k: v for k, v in solver_config.items() if k not in ['type', 'preset']}
        )
        
        # Solve with appropriate settings
        exec_config = config['execution']
        solve_kwargs = {
            'verbose': exec_config['verbose'],
            # Add other solver-specific parameters
        }
        
        # Add progress/timing control
        if hasattr(solver, 'enable_progress'):
            solver.enable_progress(exec_config['progress'])
        if hasattr(solver, 'enable_timing'):
            solver.enable_timing(exec_config['timing'])
        
        if exec_config['verbose']:
            print("🔧 Solving MFG problem...")
        
        # Solve the problem
        result = solver.solve(**solve_kwargs)
        
        if exec_config['verbose']:
            print("✅ MFG problem solved successfully!")
            if hasattr(result, 'converged'):
                print(f"   Converged: {result.converged}")
            if hasattr(result, 'iterations'):
                print(f"   Iterations: {result.iterations}")
        
        # Save results if requested
        if config['io']['output']:
            output_path = config['io']['output']
            try:
                if isinstance(result, tuple):
                    # Convert tuple result to dictionary
                    result_dict = {
                        'U': result[0].tolist(),
                        'M': result[1].tolist(),
                        'iterations': result[2] if len(result) > 2 else None
                    }
                else:
                    # Structured result
                    result_dict = {
                        'U': result.U.tolist(),
                        'M': result.M.tolist(),
                        'iterations': result.iterations,
                        'converged': result.converged,
                        'metadata': result.metadata
                    }
                
                with open(output_path, 'w') as f:
                    json.dump(result_dict, f, indent=2, default=str)
                
                print(f"💾 Results saved to: {output_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving results: {e}")
        
        # Save configuration if requested
        if config['io']['save_config']:
            save_config_file(config, config['io']['save_config'])
        
    except Exception as e:
        print(f"❌ Error running solver: {e}")
        if config['execution']['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_solver_cli()
    args = parser.parse_args()
    
    if args.command == 'solve':
        run_solver_from_cli(args)
    elif args.command == 'config':
        if args.config_command == 'generate':
            # Generate sample configuration
            sample_config = args_to_config(argparse.Namespace(
                T=1.0, Nt=50, xmin=0.0, xmax=1.0, Nx=100,
                solver_type='monitored_particle', preset='balanced',
                max_iterations=None, tolerance=None, num_particles=None,
                warm_start=False, return_structured=True,
                verbose=True, quiet=False, progress=True, timing=True,
                profile=False, output=None, save_config=None, config=None
            ))
            save_config_file(sample_config, args.output_file)
        elif args.config_command == 'validate':
            try:
                config = load_config_file(args.config_file)
                print(f"✅ Configuration file is valid: {args.config_file}")
                print(json.dumps(config, indent=2, default=str))
            except Exception as e:
                print(f"❌ Invalid configuration file: {e}")
                sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()