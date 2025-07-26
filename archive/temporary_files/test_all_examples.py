#!/usr/bin/env python3
"""
Test all examples for import issues and basic functionality.
"""

import os
import sys
import importlib.util
import traceback

def test_example_imports():
    """Test that all examples can be imported without errors."""
    examples_dir = "examples"
    examples = [f for f in os.listdir(examples_dir) if f.endswith('.py')]
    
    print("=" * 60)
    print("TESTING ALL EXAMPLES FOR IMPORT ISSUES")
    print("=" * 60)
    
    results = {}
    
    for example in sorted(examples):
        example_name = example[:-3]  # Remove .py extension
        print(f"\nTesting: {example}")
        print("-" * 40)
        
        try:
            # Import the example module
            spec = importlib.util.spec_from_file_location(
                example_name, 
                os.path.join(examples_dir, example)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print("✓ Import successful")
            
            # Check for main functions
            main_functions = [name for name in dir(module) 
                            if (name.startswith('run_') or 
                                name.startswith('main') or 
                                name.startswith('demonstrate') or
                                name == 'test')]
            
            if main_functions:
                print(f"✓ Main functions found: {', '.join(main_functions)}")
            else:
                print("? No obvious main function found")
                
            results[example] = {"status": "success", "functions": main_functions}
            
        except Exception as e:
            print(f"✗ Import failed: {e}")
            print("Error details:")
            traceback.print_exc()
            results[example] = {"status": "failed", "error": str(e)}
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [k for k, v in results.items() if v["status"] == "success"]
    failed = [k for k, v in results.items() if v["status"] == "failed"]
    
    print(f"✓ Successful: {len(successful)}/{len(examples)}")
    for example in successful:
        print(f"  - {example}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(examples)}")
        for example in failed:
            print(f"  - {example}: {results[example]['error']}")
    
    return results

if __name__ == "__main__":
    test_example_imports()