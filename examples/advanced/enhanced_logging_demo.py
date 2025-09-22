#!/usr/bin/env python3
"""
Enhanced MFG_PDE Logging System Demonstration
============================================

This script demonstrates the enhanced logging capabilities including:
- Configuration presets (research, development, production, performance)
- Advanced logging functions for solver analysis
- Log analysis utilities with performance bottleneck detection
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfg_pde.utils.log_analysis import LogAnalyzer, analyze_log_file, find_performance_bottlenecks
from mfg_pde.utils.logging import (
    LoggedOperation,
    configure_development_logging,
    configure_performance_logging,
    configure_research_logging,
    get_logger,
    log_convergence_analysis,
    log_mass_conservation,
    log_performance_metric,
    log_solver_configuration,
)


def demonstrate_configuration_presets():
    """Demonstrate different logging configuration presets."""
    print("=== CONFIGURATION PRESETS DEMONSTRATION ===")

    # 1. Research logging
    print("\n1. Research Logging Configuration:")
    log_file = configure_research_logging(experiment_name="enhanced_logging_demo", level="INFO", include_debug=False)
    print(f"   Research log file: {log_file}")

    logger = get_logger("research_demo")
    logger.info("This is a research session log entry")
    logger.info("Experiment parameters configured successfully")

    # 2. Development logging
    print("\n2. Development Logging Configuration:")
    configure_development_logging(include_location=True)
    dev_logger = get_logger("development_demo")
    dev_logger.debug("Debug information with file location")
    dev_logger.info("Development logging shows detailed information")

    # 3. Performance logging
    print("\n3. Performance Logging Configuration:")
    perf_log = configure_performance_logging()
    perf_logger = get_logger("performance_demo")
    perf_logger.info("Performance logging focuses on timing and metrics")
    print(f"   Performance log file: {perf_log}")


def demonstrate_advanced_logging_functions():
    """Demonstrate advanced logging functions for solver analysis."""
    print("\n=== ADVANCED LOGGING FUNCTIONS ===")

    logger = get_logger("solver_analysis")

    # 1. Solver configuration logging
    print("\n1. Solver Configuration Logging:")
    solver_config = {
        "max_iterations": 100,
        "tolerance": 1e-8,
        "newton_damping": 0.7,
        "adaptive_time_step": True,
        "mesh_refinement": {"levels": 3, "criterion": "error_based"},
    }

    problem_info = {"problem_type": "Mean Field Game", "domain": "[0,1] x [0,T]", "agents": 1000, "time_horizon": 2.0}

    log_solver_configuration(logger, "GFDM-Tuned-QP", solver_config, problem_info)

    # 2. Convergence analysis logging
    print("\n2. Convergence Analysis:")
    # Simulate convergence history
    error_history = [1e-2, 5e-3, 2e-3, 8e-4, 3e-4, 1e-4, 4e-5, 1.5e-5, 6e-6, 2e-6, 8e-7]
    log_convergence_analysis(logger, error_history, len(error_history), 1e-6, True)

    # 3. Mass conservation analysis
    print("\n3. Mass Conservation Analysis:")
    # Simulate mass conservation over time
    mass_history = [1.0, 0.9999, 0.99995, 1.00001, 0.99998, 1.00002, 0.99999, 1.00001]
    log_mass_conservation(logger, mass_history, tolerance=1e-5)

    # 4. Performance metrics
    print("\n4. Performance Metrics:")
    with LoggedOperation(logger, "Matrix assembly", log_level=20):  # INFO level
        time.sleep(0.2)  # Simulate work

    log_performance_metric(logger, "HJB equation solve", 1.234, {"matrix_size": "500x500", "sparsity": "0.02%"})
    log_performance_metric(logger, "Particle update", 0.456, {"num_particles": 10000, "method": "Euler"})


def demonstrate_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("\n=== LOG ANALYSIS DEMONSTRATION ===")

    # First, create some sample log data
    print("\n1. Creating Sample Log Data...")
    configure_performance_logging(log_file="sample_analysis.log")
    logger = get_logger("analysis_sample")

    # Generate various log entries for analysis
    for i in range(5):
        logger.info("Initializing HJBSolver")
        logger.info(f"Performance - Matrix assembly: {0.1 + i*0.05:.3f}s")
        logger.info(f"Iteration {i+1}/10 (10.0%) - Error: {1e-3/(i+1):.2e}")
        if i == 2:
            logger.error("Convergence issue detected - reducing time step")
        logger.info("HJBSolver completed - Status: CONVERGED")
        time.sleep(0.01)  # Small delay for timestamp variation

    # Now analyze the log
    print("\n2. Analyzing Log File...")
    log_file = Path("sample_analysis.log")

    if log_file.exists():
        # Use the comprehensive analysis function
        analysis_results = analyze_log_file(str(log_file), generate_report=True)

        print("\n3. Analysis Summary:")
        print(f"   Total entries: {analysis_results['summary'].get('total_entries', 0)}")
        print(f"   Error count: {analysis_results['errors'].get('total_errors', 0)}")
        print(f"   Solver sessions: {len(analysis_results['performance'].get('solver_sessions', []))}")

        # Find performance bottlenecks
        print("\n4. Performance Bottlenecks:")
        bottlenecks = find_performance_bottlenecks(str(log_file), threshold_seconds=0.1)
        if bottlenecks:
            for bottleneck in bottlenecks[:3]:  # Show top 3
                print(f"   {bottleneck['operation']}: {bottleneck['duration']:.3f}s")
        else:
            print("   No significant bottlenecks found (>0.1s)")

        # Export analysis as JSON
        print("\n5. Exporting Analysis:")
        analyzer = LogAnalyzer(str(log_file))
        json_file = analyzer.export_analysis_json()
        print(f"   Analysis exported to: {json_file}")
    else:
        print("   Sample log file not created - skipping analysis")


def demonstrate_log_locations():
    """Show where different types of logs are stored."""
    print("\n=== LOG FILE LOCATIONS ===")

    # Check for different log directories
    log_dirs = {
        "Default logs": Path("logs"),
        "Research logs": Path("research_logs"),
        "Performance logs": Path("performance_logs"),
        "Production logs": Path("production_logs"),
    }

    for name, path in log_dirs.items():
        if path.exists():
            log_files = list(path.glob("*.log"))
            print(f"\n📁 {name}: {path}")
            print(f"   Files found: {len(log_files)}")
            if log_files:
                latest = max(log_files, key=lambda p: p.stat().st_mtime)
                print(f"   Latest: {latest.name} ({latest.stat().st_size} bytes)")
        else:
            print(f"\n📁 {name}: {path} (not created yet)")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ENHANCED MFG_PDE LOGGING SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Demonstrate all capabilities
    demonstrate_configuration_presets()
    demonstrate_advanced_logging_functions()
    demonstrate_log_analysis()
    demonstrate_log_locations()

    print("\n" + "=" * 80)
    print("ENHANCED LOGGING FEATURES SUMMARY")
    print("=" * 80)

    print("\nNEW CONFIGURATION PRESETS:")
    print("   • configure_research_logging() - Optimized for research sessions")
    print("   • configure_development_logging() - Debug with location info")
    print("   • configure_production_logging() - Warnings/errors only")
    print("   • configure_performance_logging() - Focus on timing/metrics")

    print("\nENHANCED LOGGING FUNCTIONS:")
    print("   • log_solver_configuration() - Detailed solver setup")
    print("   • log_convergence_analysis() - Convergence rate analysis")
    print("   • log_mass_conservation() - Mass conservation tracking")
    print("   • Enhanced LoggedOperation context manager")

    print("\nLOG ANALYSIS UTILITIES:")
    print("   • LogAnalyzer class for comprehensive analysis")
    print("   • analyze_log_file() - Quick analysis with reports")
    print("   • find_performance_bottlenecks() - Performance optimization")
    print("   • JSON export for automated processing")

    print("\n📁 ORGANIZED LOG STORAGE:")
    print("   • research_logs/ - Research session logs")
    print("   • performance_logs/ - Performance analysis logs")
    print("   • production_logs/ - Production environment logs")
    print("   • logs/ - General purpose logs")

    print("\n✨ The enhanced logging system is ready for advanced research workflows!")


if __name__ == "__main__":
    main()
