#!/usr/bin/env python3
"""
Simple MFG_PDE Logging Demonstration
===================================

This script demonstrates the logging system and shows where logs are stored.
"""

import sys
import os
import logging
from pathlib import Path
import time

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde.utils.logging import configure_logging, get_logger, LoggedOperation


def main():
    """Demonstrate logging system and show where logs are stored."""
    print("=" * 60)
    print("MFG_PDE LOGGING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. CONSOLE LOGGING (Default):")
    print("-" * 40)
    
    # Configure console logging
    configure_logging(level="INFO", log_to_file=False, use_colors=True)
    logger = get_logger("demo")
    
    logger.info("This appears on console only")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\n2. FILE LOGGING:")
    print("-" * 40)
    
    # Enable file logging
    configure_logging(
        level="DEBUG", 
        log_to_file=True,
        use_colors=True,
        include_location=True
    )
    
    logger = get_logger("file_demo")
    logger.info("File logging enabled - this goes to both console and file")
    logger.debug("Debug message - only visible with DEBUG level")
    
    # Show where the log file is created
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("mfg_pde_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📁 Log file created: {latest_log}")
            print(f"📊 File size: {latest_log.stat().st_size} bytes")
    
    print("\n3. CUSTOM LOG LOCATION:")
    print("-" * 40)
    
    # Custom log location
    custom_log = Path("my_experiment.log")
    configure_logging(
        level="INFO",
        log_to_file=True,
        log_file_path=custom_log,
        use_colors=False
    )
    
    research_logger = get_logger("research")
    research_logger.info("Research session started")
    research_logger.info("Testing custom log location")
    
    if custom_log.exists():
        print(f"📁 Custom log created: {custom_log}")
        print(f"📊 Content:")
        with open(custom_log, 'r') as f:
            for line in f:
                print(f"   {line.rstrip()}")
    
    print("\n4. TIMED OPERATIONS:")
    print("-" * 40)
    
    # Demonstrate timed operations
    logger = get_logger("timing_demo") 
    with LoggedOperation(logger, "Demo computation"):
        logger.info("Performing demo computation...")
        time.sleep(1)  # Simulate work
    
    print("\n" + "=" * 60)
    print("SUMMARY: WHERE TO FIND LOGS")
    print("=" * 60)
    
    print("\n📍 LOG LOCATIONS:")
    print("  • Console: Always visible during execution")
    print("  • Default files: ./logs/mfg_pde_YYYYMMDD_HHMMSS.log")
    print("  • Custom files: Any path you specify")
    
    print("\n🔧 CONFIGURATION:")
    print("  from mfg_pde.utils.logging import configure_logging, get_logger")
    print("  ")
    print("  # Enable file logging")
    print("  configure_logging(level='INFO', log_to_file=True)")
    print("  ")
    print("  # Get logger and use it")
    print("  logger = get_logger(__name__)")
    print("  logger.info('My message')")
    
    print("\n🎯 The logging system is ready for use!")
    print("   Check the created log files in ./logs/ directory")


if __name__ == "__main__":
    main()