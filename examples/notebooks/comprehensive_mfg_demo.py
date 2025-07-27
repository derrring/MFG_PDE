#!/usr/bin/env python3
"""
Comprehensive MFG Demo using Modern MFG_PDE Framework

This example demonstrates the complete modern workflow:
- Pydantic configuration with validation
- Factory pattern solver creation
- Professional logging system
- Interactive notebook report generation
- Array validation with physical constraints
"""

import numpy as np
import time
from pathlib import Path

# Core MFG framework
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.factory.pydantic_solver_factory import create_validated_solver
from mfg_pde.config.pydantic_config import MFGSolverConfig, create_research_config
from mfg_pde.config.array_validation import MFGGridConfig, ExperimentConfig

# Utilities
from mfg_pde.utils.logging import configure_research_logging, log_convergence_analysis
from mfg_pde.utils.notebook_reporting import create_mfg_research_report
from mfg_pde.utils.log_analysis import LogAnalyzer

def create_standard_mfg_problem():
    """Create our usual MFG problem setup."""
    print("🔧 Creating standard MFG problem...")
    
    # Grid configuration with validation
    grid_config = MFGGridConfig(
        Nx=50,           # Spatial points
        Nt=30,           # Time points  
        xmin=0.0,
        xmax=1.0,
        T=1.0,           # Final time
        sigma=0.2        # Diffusion coefficient
    )
    
    print(f"   Grid: {grid_config.Nx}×{grid_config.Nt}, CFL={grid_config.cfl_number:.3f}")
    
    # Create MFG problem using validated configuration
    problem = ExampleMFGProblem(
        xmin=grid_config.xmin,
        xmax=grid_config.xmax, 
        T=grid_config.T,
        Nx=grid_config.Nx,
        Nt=grid_config.Nt,
        sigma=grid_config.sigma
    )
    
    return problem, grid_config

def solve_mfg_with_modern_framework():
    """Solve MFG using complete modern framework."""
    
    # Setup logging for research
    log_file = configure_research_logging(
        experiment_name="comprehensive_mfg_demo",
        level="INFO",
        include_debug=True
    )
    
    # Get the configured logger
    from mfg_pde.utils.logging import get_logger
    logger = get_logger("mfg_pde.research")
    
    logger.info("🚀 Starting comprehensive MFG demonstration")
    
    # Create problem
    problem, grid_config = create_standard_mfg_problem()
    
    # Create research-grade configuration with Pydantic validation
    config = create_research_config()
    logger.info(f"📋 Configuration: Picard max_iter={config.picard.max_iterations}, "
                f"Newton max_iter={config.newton.max_iterations}")
    
    # Test multiple solver types with factory pattern
    solver_results = {}
    solver_types = [
        "particle_collocation",
        "adaptive_particle", 
        "hjb_gfdm_qp"
    ]
    
    for solver_type in solver_types:
        logger.info(f"🔬 Testing solver: {solver_type}")
        
        try:
            start_time = time.time()
            
            # Create solver using validated factory pattern
            solver = create_validated_solver(
                problem=problem,
                solver_type=solver_type,
                config=config
            )
            
            # Solve with comprehensive logging
            result = solver.solve(
                max_picard_iterations=config.picard.max_iterations,
                picard_tolerance=config.picard.tolerance,
                verbose=True
            )
            
            solve_time = time.time() - start_time
            
            # Extract solutions (handle both tuple and structured returns)
            if hasattr(result, 'solution'):
                U, M = result.solution, result.density
                convergence_info = result.convergence_info
            else:
                U, M = result[0], result[1]
                convergence_info = result[2] if len(result) > 2 else {}
            
            # Log convergence analysis
            error_history = convergence_info.get('error_history', [])
            if error_history:
                log_convergence_analysis(
                    logger=logger,
                    error_history=error_history,
                    final_iterations=len(error_history),
                    tolerance=config.picard.tolerance,
                    converged=convergence_info.get('converged', True)
                )
            
            solver_results[solver_type] = {
                'U': U,
                'M': M,
                'convergence_info': convergence_info,
                'solve_time': solve_time,
                'solver_instance': solver,
                'grid_config': grid_config
            }
            
            logger.info(f"✅ {solver_type} completed in {solve_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ {solver_type} failed: {str(e)}")
            solver_results[solver_type] = {'error': str(e)}
    
    return solver_results, problem, config, logger

def validate_solutions_with_pydantic(solver_results):
    """Validate solutions using Pydantic array validation."""
    print("\n🔍 Validating solutions with Pydantic...")
    
    validation_results = {}
    
    for solver_type, result in solver_results.items():
        if 'error' in result:
            continue
            
        try:
            from mfg_pde.config.array_validation import MFGArrays
            
            # Create validated array container
            arrays = MFGArrays(
                U_solution=result['U'],
                M_solution=result['M'], 
                grid_config=result['grid_config']
            )
            
            # Get comprehensive statistics
            stats = arrays.get_solution_statistics()
            validation_results[solver_type] = {
                'validation_passed': True,
                'statistics': stats
            }
            
            print(f"   ✅ {solver_type}: Mass conservation = {stats['mass_conservation']['mass_drift']:.2e}")
            
        except Exception as e:
            validation_results[solver_type] = {
                'validation_passed': False,
                'error': str(e)
            }
            print(f"   ❌ {solver_type}: Validation failed - {str(e)}")
    
    return validation_results

def generate_comprehensive_reports(solver_results, problem, config, logger):
    """Generate notebook reports and perform log analysis."""
    
    logger.info("📊 Generating comprehensive reports...")
    
    # Create output directory
    output_dir = Path("./results/comprehensive_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reports_generated = []
    
    for solver_type, result in solver_results.items():
        if 'error' in result:
            continue
            
        try:
            # Prepare solver results for reporting
            solver_results_dict = {
                'solution_U': result['U'],
                'solution_M': result['M'],
                'convergence_info': result['convergence_info'],
                'timing': {
                    'solve_time': result['solve_time'],
                    'solver_type': solver_type
                }
            }
            
            # Prepare problem configuration  
            problem_config = {
                'grid_config': result['grid_config'].dict(),
                'solver_config': config.model_dump(),
                'solver_type': solver_type,
                'problem_parameters': {
                    'xmin': problem.xmin,
                    'xmax': problem.xmax,
                    'T': problem.T,
                    'sigma': problem.sigma
                }
            }
            
            # Generate notebook report
            report_paths = create_mfg_research_report(
                title=f"MFG Analysis: {solver_type.replace('_', ' ').title()}",
                solver_results=solver_results_dict,
                problem_config=problem_config,
                output_dir=str(output_dir),
                export_html=True
            )
            
            reports_generated.append({
                'solver_type': solver_type,
                'notebook_path': report_paths.get('notebook_path'),
                'html_path': report_paths.get('html_path')
            })
            
            logger.info(f"📄 Report generated for {solver_type}")
            
        except Exception as e:
            logger.error(f"❌ Report generation failed for {solver_type}: {str(e)}")
    
    return reports_generated

def perform_log_analysis(logger):
    """Demonstrate log analysis capabilities."""
    
    logger.info("🔍 Performing log analysis...")
    
    try:
        # Find the log file from our current session
        import logging
        log_files = []
        
        # Get log file paths from handlers
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                log_files.append(handler.baseFilename)
        
        if log_files:
            analyzer = LogAnalyzer()
            
            for log_file in log_files:
                logger.info(f"📈 Analyzing log: {log_file}")
                
                # Analyze solver performance
                performance_analysis = analyzer.analyze_solver_performance()
                logger.info(f"Performance analysis completed")
                
                # Find performance bottlenecks
                bottlenecks = analyzer.find_performance_bottlenecks(
                    log_file_path=log_file,
                    threshold_seconds=0.1
                )
                
                if bottlenecks:
                    logger.info(f"Found {len(bottlenecks)} performance bottlenecks")
                    for bottleneck in bottlenecks[:3]:  # Show top 3
                        logger.info(f"  - {bottleneck['operation']}: {bottleneck['duration']:.3f}s")
                else:
                    logger.info("No significant performance bottlenecks detected")
        
    except Exception as e:
        logger.error(f"Log analysis failed: {str(e)}")

def main():
    """Main demonstration function."""
    print("=" * 60)
    print("🎯 MFG_PDE Comprehensive Framework Demonstration")
    print("=" * 60)
    
    start_time = time.time()
    
    # Solve MFG with modern framework
    solver_results, problem, config, logger = solve_mfg_with_modern_framework()
    
    # Validate solutions with Pydantic
    validation_results = validate_solutions_with_pydantic(solver_results)
    
    # Generate comprehensive reports
    reports = generate_comprehensive_reports(solver_results, problem, config, logger)
    
    # Perform log analysis
    perform_log_analysis(logger)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    successful_solvers = len([r for r in solver_results.values() if 'error' not in r])
    print(f"✅ Solvers completed: {successful_solvers}/{len(solver_results)}")
    
    validation_passed = len([v for v in validation_results.values() if v.get('validation_passed', False)])
    print(f"🔍 Validations passed: {validation_passed}/{len(validation_results)}")
    
    print(f"📊 Reports generated: {len(reports)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    # Show generated files
    if reports:
        print("\n📄 Generated Reports:")
        for report in reports:
            if report['notebook_path']:
                print(f"   - {report['solver_type']}: {report['notebook_path']}")
    
    logger.info(f"🎉 Comprehensive demonstration completed in {total_time:.2f}s")
    
    return solver_results, validation_results, reports

if __name__ == "__main__":
    main()