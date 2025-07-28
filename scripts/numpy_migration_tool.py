#!/usr/bin/env python3
"""
NumPy 2.0+ Migration Tool for MFG_PDE

Command-line utility for NumPy compatibility checking, migration assistance,
and performance benchmarking.
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MFG_PDE NumPy 2.0+ Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python numpy_migration_tool.py check                    # Check NumPy compatibility
  python numpy_migration_tool.py migrate                  # Interactive migration assistant  
  python numpy_migration_tool.py validate                 # Validate installation
  python numpy_migration_tool.py benchmark                # Performance benchmarking
  python numpy_migration_tool.py benchmark --save results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check NumPy compatibility')
    check_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Show detailed information')
    
    # Migration command
    migrate_parser = subparsers.add_parser('migrate',
                                          help='Interactive NumPy 2.0+ migration assistant')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate',
                                           help='Validate MFG_PDE installation')
    validate_parser.add_argument('--fix', action='store_true',
                                help='Attempt to fix issues automatically')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark',
                                           help='Performance benchmarking')
    benchmark_parser.add_argument('--save', '-s', type=str,
                                 help='Save results to file')
    benchmark_parser.add_argument('--sizes', nargs='+', type=int,
                                 default=[1000, 10000, 100000],
                                 help='Array sizes to benchmark')
    
    # Version command
    version_parser = subparsers.add_parser('version',
                                          help='Show version information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        return execute_command(args)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def execute_command(args):
    """Execute the specified command."""
    
    # Add the parent directory to Python path to import mfg_pde
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    if args.command == 'check':
        from mfg_pde.utils.numpy_compat import check_numpy_compatibility, get_numpy_version_info
        
        if args.verbose:
            info = get_numpy_version_info()
            print("📊 Detailed NumPy Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()
        
        check_numpy_compatibility()
        return 0
    
    elif args.command == 'migrate':
        from mfg_pde.utils.numpy_compat import migration_assistant
        migration_assistant()
        return 0
    
    elif args.command == 'validate':
        from mfg_pde.utils.numpy_compat import validate_installation
        results = validate_installation()
        
        if args.fix and not results['overall_status']:
            print("\n🔧 Attempting automatic fixes...")
            attempt_fixes(results)
        
        return 0 if results['overall_status'] else 1
    
    elif args.command == 'benchmark':
        from mfg_pde.utils.numpy_compat import benchmark_performance
        import time
        import numpy as np
        
        # Custom benchmark with user-specified sizes
        print("🏃 Custom Performance Benchmarking")
        print("=" * 40)
        
        results = {}
        
        for size in args.sizes:
            x = np.linspace(0, 1, size)
            y = np.sin(x) * np.exp(-x)
            
            print(f"\nArray size: {size:,}")
            
            # Benchmark available functions
            times = {}
            
            if hasattr(np, 'trapezoid'):
                start = time.perf_counter()
                for _ in range(100):
                    result_trapezoid = np.trapezoid(y, x)
                times['trapezoid'] = (time.perf_counter() - start) / 100
                print(f"  trapezoid: {times['trapezoid']*1000:.3f} ms")
            
            if hasattr(np, 'trapz'):
                start = time.perf_counter()
                for _ in range(100):
                    result_trapz = np.trapz(y, x)  
                times['trapz'] = (time.perf_counter() - start) / 100
                print(f"  trapz:     {times['trapz']*1000:.3f} ms")
            
            # Calculate speedup if both available
            if 'trapezoid' in times and 'trapz' in times:
                speedup = times['trapz'] / times['trapezoid']
                print(f"  speedup:   {speedup:.2f}× (trapezoid vs trapz)")
                
            results[size] = times
        
        if args.save:
            save_path = Path(args.save)
            # Add metadata
            benchmark_data = {
                'results': results,
                'numpy_version': np.__version__,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'array_sizes': args.sizes,
                'function_info': {
                    'has_trapezoid': hasattr(np, 'trapezoid'),
                    'has_trapz': hasattr(np, 'trapz')
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(benchmark_data, f, indent=2, default=str)
            print(f"\n💾 Benchmark results saved to: {save_path}")
        
        return 0
    
    elif args.command == 'version':
        show_version_info()
        return 0
    
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


def attempt_fixes(validation_results):
    """Attempt to automatically fix installation issues."""
    import subprocess
    import sys
    
    fixes_attempted = []
    
    if not validation_results['numpy_version']:
        print("   🔄 Upgrading NumPy...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'numpy>=2.0'
            ])
            fixes_attempted.append("NumPy upgrade")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to upgrade NumPy")
    
    if not validation_results['mfg_imports']:
        print("   🔄 Reinstalling MFG_PDE...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', '.'
            ])
            fixes_attempted.append("MFG_PDE reinstall")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to reinstall MFG_PDE")
    
    if fixes_attempted:
        print(f"\n✅ Attempted fixes: {', '.join(fixes_attempted)}")
        print("🔄 Please restart Python and run validation again")
    else:
        print("   ℹ️  No automatic fixes available")


def show_version_info():
    """Show comprehensive version information."""
    print("📦 MFG_PDE Version Information")
    print("=" * 40)
    
    try:
        import mfg_pde
        print(f"MFG_PDE: {mfg_pde.__version__}")
    except:
        print("MFG_PDE: Not installed or version unavailable")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        
        # NumPy-specific information
        from mfg_pde.utils.numpy_compat import get_numpy_version_info
        info = get_numpy_version_info()
        print(f"  • Supports NumPy 2.0+: {info['supports_numpy_2']}")
        print(f"  • Has trapezoid: {info['has_trapezoid']}")
        print(f"  • Has trapz: {info['has_trapz']}")
        
    except:
        print("NumPy: Not available")
    
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except:
        print("SciPy: Not available")
    
    try:
        import matplotlib
        print(f"Matplotlib: {matplotlib.__version__}")
    except:
        print("Matplotlib: Not available")
    
    print(f"Python: {sys.version}")


if __name__ == '__main__':
    sys.exit(main())