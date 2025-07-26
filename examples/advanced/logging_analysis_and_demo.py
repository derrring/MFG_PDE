#!/usr/bin/env python3
"""
MFG_PDE Logging System Analysis and Demonstration
===============================================

This script analyzes the current logging system and demonstrates where
logs are stored and how to use the logging infrastructure effectively.
"""

import sys
import os
import logging
from pathlib import Path
import time

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde.utils.logging import (
    configure_logging, get_logger, MFGLogger,
    log_solver_start, log_solver_progress, log_solver_completion,
    LoggedOperation
)


def analyze_current_logging_system():
    """Analyze the current logging system capabilities and configuration."""
    print("=" * 70)
    print("MFG_PDE LOGGING SYSTEM ANALYSIS")
    print("=" * 70)
    
    print("\n1. CURRENT LOGGING CAPABILITIES:")
    print("   ✅ Professional logging infrastructure with MFGLogger singleton")
    print("   ✅ Colored console output (with optional colorlog dependency)")
    print("   ✅ File logging capability with configurable paths")
    print("   ✅ Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    print("   ✅ Custom formatters with timestamps and module names")
    print("   ✅ Context managers for timed operations")
    print("   ✅ Specialized logging functions for solver operations")
    print("   ✅ External library noise suppression")
    print("   ✅ Location information (file:line) option")
    
    print("\n2. DEFAULT CONFIGURATION:")
    print("   - Console Output: ✅ Enabled by default")
    print("   - File Output: ❌ Disabled by default")
    print("   - Colors: ✅ Enabled if colorlog available")
    print("   - Log Level: INFO")
    print("   - External Suppression: ✅ Enabled")
    
    print("\n3. LOG STORAGE LOCATIONS:")
    print("   📍 Console: All logs printed to stdout")
    print("   📍 File (when enabled): ./logs/mfg_pde_{timestamp}.log")
    print("   📍 Custom Path: Configurable via log_file_path parameter")
    
    print("\n4. CURRENT USAGE ANALYSIS:")
    # Check where logging is currently used
    usage_locations = [
        "✅ Mathematical visualization modules",
        "✅ Advanced visualization modules", 
        "✅ Notebook reporting system",
        "✅ Examples and demonstrations",
        "❓ Core solver classes (limited usage)",
        "❓ Configuration system (limited)",
        "❓ Factory patterns (limited)"
    ]
    
    for location in usage_locations:
        print(f"   {location}")


def demonstrate_console_logging():
    """Demonstrate console logging with different levels."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: CONSOLE LOGGING")
    print("=" * 70)
    
    print("\nConfiguring logging for console output only...")
    configure_logging(
        level="DEBUG",
        log_to_file=False,
        use_colors=True,
        include_location=False
    )
    
    # Get loggers for different components
    solver_logger = get_logger("mfg_pde.solvers.demo")
    config_logger = get_logger("mfg_pde.config.demo")
    main_logger = get_logger("demo.main")
    
    print("\nDemonstrating different log levels:")
    print("(Note: Colors may appear in terminal but not in this output)")
    
    # Demonstrate all log levels
    solver_logger.debug("🔍 DEBUG: Detailed solver internal information")
    solver_logger.info("ℹ️  INFO: Solver initialized successfully")
    solver_logger.warning("⚠️  WARNING: Using default parameters, consider optimization")
    solver_logger.error("❌ ERROR: Convergence issue detected in iteration 15")
    solver_logger.critical("🚨 CRITICAL: Solver failed completely")
    
    # Demonstrate structured logging functions
    print("\nDemonstrating structured solver logging:")
    
    config = {
        "max_newton_iterations": 30,
        "newton_tolerance": 1e-6,
        "max_picard_iterations": 20,
        "picard_tolerance": 1e-4
    }
    
    log_solver_start(solver_logger, "ParticleCollocationSolver", config)
    
    # Simulate solver progress
    for i in [1, 5, 10, 15, 20]:
        error = 1e-2 * (0.8 ** i)
        additional_info = {"phase": "Picard" if i < 15 else "Newton", "damping": 0.5}
        log_solver_progress(solver_logger, i, error, 20, additional_info)
        time.sleep(0.1)  # Brief pause for demonstration
    
    log_solver_completion(solver_logger, "ParticleCollocationSolver", 
                         18, 3.2e-7, 2.45, True)
    
    # Demonstrate context manager
    print("\nDemonstrating timed operation logging:")
    with LoggedOperation(main_logger, "Matrix factorization", level=logging.INFO):
        time.sleep(0.5)  # Simulate computation
        # Operation automatically logs completion time


def demonstrate_file_logging():
    """Demonstrate file logging capabilities."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: FILE LOGGING")
    print("=" * 70)
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    print(f"Configuring logging with file output to: {log_dir}/")
    
    configure_logging(
        level="DEBUG",
        log_to_file=True,
        log_file_path=None,  # Use default timestamped filename
        use_colors=True,     # Colors for console, plain text for file
        include_location=True,
        suppress_external=True
    )
    
    logger = get_logger("mfg_pde.file_demo")
    
    # Generate various log messages
    logger.info("File logging demonstration started")
    logger.debug("This debug message will appear in both console and file")
    logger.warning("This warning demonstrates file logging capability")
    
    # Simulate some solver operations
    with LoggedOperation(logger, "File logging test operation"):
        logger.info("Performing file logging test operations...")
        for i in range(3):
            logger.debug(f"Test operation step {i+1}/3")
            time.sleep(0.2)
    
    logger.info("File logging demonstration completed")
    
    # Show where the log file was created
    log_files = list(log_dir.glob("mfg_pde_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"\n📁 Log file created: {latest_log}")
        print(f"📊 Log file size: {latest_log.stat().st_size} bytes")
        
        print("\n📄 Log file contents (last 10 lines):")
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("❌ No log file was created")


def demonstrate_custom_log_location():
    """Demonstrate custom log file location."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: CUSTOM LOG LOCATION")
    print("=" * 70)
    
    # Create custom log directory
    custom_log_dir = Path("my_research_logs")
    custom_log_dir.mkdir(exist_ok=True)
    
    custom_log_file = custom_log_dir / "mfg_research_session.log"
    
    print(f"Configuring custom log location: {custom_log_file}")
    
    configure_logging(
        level="INFO",
        log_to_file=True,
        log_file_path=custom_log_file,
        use_colors=False,  # Disable colors for cleaner file output
        include_location=True
    )
    
    research_logger = get_logger("mfg_pde.research")
    
    # Log some research activities
    research_logger.info("Research session started")
    research_logger.info("Problem configuration: T=1.0, Nx=50, sigma=0.2")
    research_logger.info("Using Particle-Collocation method with 1000 particles")
    
    # Simulate research progress
    with LoggedOperation(research_logger, "MFG problem solving"):
        research_logger.info("Initializing solver...")
        time.sleep(0.3)
        research_logger.info("Running Picard iterations...")
        time.sleep(0.4)
        research_logger.info("Convergence achieved")
    
    research_logger.info("Research session completed successfully")
    
    # Show custom log file
    if custom_log_file.exists():
        print(f"\n📁 Custom log file: {custom_log_file}")
        print(f"📊 File size: {custom_log_file.stat().st_size} bytes")
        
        print("\n📄 Custom log contents:")
        with open(custom_log_file, 'r') as f:
            for line in f:
                print(f"   {line.rstrip()}")
    else:
        print("❌ Custom log file was not created")


def analyze_logging_strengths_and_gaps():
    """Analyze the strengths and identify gaps in the current logging system."""
    print("\n" + "=" * 70)
    print("LOGGING SYSTEM ANALYSIS: STRENGTHS & GAPS")
    print("=" * 70)
    
    print("\n🎯 STRENGTHS:")
    strengths = [
        "Professional architecture with singleton pattern",
        "Flexible configuration (console + file, colors, levels)",
        "Structured logging functions for solver operations",
        "Context managers for automatic timing",
        "External library noise suppression",
        "Good integration with existing codebase",
        "Thread-safe implementation",
        "Memory efficient (singleton pattern)"
    ]
    
    for strength in strengths:
        print(f"   ✅ {strength}")
    
    print("\n⚠️  GAPS & IMPROVEMENT OPPORTUNITIES:")
    gaps = [
        "Limited integration in core solver classes",
        "No log rotation (files can grow indefinitely)",
        "No structured logging (JSON format) for automated analysis", 
        "No centralized log aggregation for distributed runs",
        "No built-in log analysis tools",
        "Default configuration may not be obvious to new users",
        "No performance impact measurement",
        "No log level runtime adjustment without reconfiguration"
    ]
    
    for gap in gaps:
        print(f"   ❌ {gap}")
    
    print("\n💡 RECOMMENDED IMPROVEMENTS:")
    improvements = [
        "Add logging to core solver classes by default",
        "Implement log rotation (daily/size-based)",
        "Add JSON logging format option for automated analysis",
        "Create log analysis utilities (error summaries, performance metrics)",
        "Add runtime log level adjustment API",
        "Implement solver performance logging by default",
        "Add configuration presets (research, production, debug)",
        "Create log viewer utility for better log inspection"
    ]
    
    for improvement in improvements:
        print(f"   🚀 {improvement}")


def provide_usage_recommendations():
    """Provide recommendations for effective logging usage."""
    print("\n" + "=" * 70)
    print("LOGGING USAGE RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n📋 FOR RESEARCHERS:")
    print("""
   1. Enable file logging for research sessions:
      configure_logging(level="INFO", log_to_file=True, 
                       log_file_path="research_session.log")
   
   2. Use DEBUG level for detailed solver analysis:
      configure_logging(level="DEBUG", include_location=True)
   
   3. Use structured logging for experiment tracking:
      logger.info(f"Experiment: particles={n_particles}, tolerance={tol}")
   
   4. Use LoggedOperation for timing critical sections:
      with LoggedOperation(logger, "Convergence analysis"):
          # Your analysis code here
   """)
    
    print("\n🔧 FOR DEVELOPERS:")
    print("""
   1. Add logging to all new solver classes:
      self.logger = get_logger(__name__)
      self.logger.info("Solver initialized")
   
   2. Use appropriate log levels:
      DEBUG: Detailed internal state
      INFO: Key operations and results  
      WARNING: Suboptimal conditions
      ERROR: Recoverable failures
      CRITICAL: Unrecoverable failures
   
   3. Log solver progress in iterations:
      log_solver_progress(logger, iter, error, max_iter, extra_info)
   """)
    
    print("\n⚙️  FOR PRODUCTION USE:")
    print("""
   1. Use WARNING or ERROR level to reduce noise:
      configure_logging(level="WARNING", log_to_file=True)
   
   2. Implement log rotation for long-running processes:
      # (Feature to be implemented)
   
   3. Monitor log files for error patterns:
      grep "ERROR\\|CRITICAL" logs/mfg_pde_*.log
   """)


def main():
    """Main demonstration function."""
    print("MFG_PDE Logging System Analysis and Demonstration")
    print("This script analyzes the logging infrastructure and shows where logs are stored.")
    
    # Step 1: Analyze current system
    analyze_current_logging_system()
    
    # Step 2: Demonstrate console logging
    demonstrate_console_logging()
    
    # Step 3: Demonstrate file logging
    demonstrate_file_logging()
    
    # Step 4: Demonstrate custom log location
    demonstrate_custom_log_location()
    
    # Step 5: Analyze strengths and gaps
    analyze_logging_strengths_and_gaps()
    
    # Step 6: Provide usage recommendations
    provide_usage_recommendations()
    
    print("\n" + "=" * 70)
    print("SUMMARY: WHERE TO FIND YOUR LOGS")
    print("=" * 70)
    
    print("\n📍 LOG LOCATIONS:")
    print("   1. Console Output: Always visible when running Python scripts")
    print("   2. Default File Logs: ./logs/mfg_pde_{timestamp}.log")
    print("   3. Custom File Logs: Any path you specify in configure_logging()")
    print("   4. Research Logs: ./my_research_logs/mfg_research_session.log (from demo)")
    
    print("\n🔍 HOW TO VIEW LOGS:")
    print("   - Console: Watch terminal output while running")
    print("   - File: tail -f logs/mfg_pde_*.log (live monitoring)")
    print("   - File: cat logs/mfg_pde_*.log (full contents)")
    print("   - File: grep ERROR logs/mfg_pde_*.log (filter errors)")
    
    print("\n⚙️  QUICK START LOGGING:")
    print("""   from mfg_pde.utils.logging import configure_logging, get_logger
   
   # Enable file logging
   configure_logging(level="INFO", log_to_file=True)
   
   # Get logger and use it
   logger = get_logger(__name__)
   logger.info("My research session started")""")
    
    print("\n🎯 The logging system is comprehensive and ready for use!")
    print("   Check the created log files to see the output.")


if __name__ == "__main__":
    main()