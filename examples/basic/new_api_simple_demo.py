#!/usr/bin/env python3
"""
New API Simple Demo - Dead Simple MFG Solving

This example demonstrates the new Layer 1 API that makes MFG solving
as simple as calling a single function.

Perfect for:
- Research prototyping
- Quick experiments
- Teaching and demonstrations
- Getting started with MFG
"""

from mfg_pde import get_available_problems, load_example, solve_mfg


def main():
    print("🚀 MFG_PDE New API Demo - Simple Interface")
    print("=" * 50)

    # 1. SIMPLEST POSSIBLE USAGE
    print("\n1. Simplest crowd dynamics simulation:")
    result = solve_mfg("crowd_dynamics")
    print(f"✅ Solved! Converged in {result.iterations} iterations")

    # Built-in visualization
    result.plot()

    # 2. CUSTOM PARAMETERS
    print("\n2. Custom parameters:")
    result = solve_mfg(
        "crowd_dynamics",
        domain_size=5.0,  # 5-meter corridor
        crowd_size=300,  # 300 people
        time_horizon=2.0,  # 2 seconds
        accuracy="high",  # High accuracy
        verbose=True,
    )  # Show progress

    print(f"📊 Evacuation efficiency: {result.evacuation_efficiency:.1%}")
    print(f"🎯 Final mass: {result.total_mass:.3f}")

    # 3. DIFFERENT PROBLEM TYPES
    print("\n3. Different problem types:")

    # Portfolio optimization
    portfolio_result = solve_mfg("portfolio_optimization", risk_aversion=0.3, time_horizon=1.5)
    print(f"📈 Portfolio optimization converged in {portfolio_result.iterations} iterations")

    # Traffic flow
    traffic_result = solve_mfg("traffic_flow", domain_size=10.0, speed_limit=2.0)
    print(f"🚗 Traffic simulation converged in {traffic_result.iterations} iterations")

    # 4. ACCURACY LEVELS
    print("\n4. Accuracy comparison:")

    # Fast (for prototyping)
    fast_result = solve_mfg("crowd_dynamics", accuracy="fast")
    print(f"⚡ Fast: {fast_result.iterations} iterations, {fast_result.solve_time:.2f}s")

    # Balanced (default)
    balanced_result = solve_mfg("crowd_dynamics", accuracy="balanced")
    print(f"⚖️ Balanced: {balanced_result.iterations} iterations, {balanced_result.solve_time:.2f}s")

    # High accuracy (for research)
    high_result = solve_mfg("crowd_dynamics", accuracy="high")
    print(f"🎯 High: {high_result.iterations} iterations, {high_result.solve_time:.2f}s")

    # 5. PREDEFINED EXAMPLES
    print("\n5. Load predefined examples:")

    # Built-in examples
    example_result = load_example("simple_crowd")
    print(f"📚 Example loaded and solved: {example_result.iterations} iterations")

    # 6. PARAMETER VALIDATION AND HELP
    print("\n6. Getting help with parameters:")

    # See available problems
    problems = get_available_problems()
    print("Available problem types:")
    for name, info in problems.items():
        print(f"  • {name}: {info['description']}")

    # Parameter validation
    from mfg_pde import suggest_problem_setup, validate_problem_parameters

    validation = validate_problem_parameters("crowd_dynamics", crowd_size=-10)
    if not validation["valid"]:
        print("\n❌ Invalid parameters detected:")
        for issue in validation["issues"]:
            print(f"   - {issue}")

    # Get suggestions
    setup = suggest_problem_setup("crowd_dynamics")
    print(f"\n💡 Suggested setup: {setup['parameters']}")

    # 7. SMART SOLVING WITH AUTO-OPTIMIZATION
    print("\n7. Smart solving with automatic optimization:")

    smart_result = solve_mfg("epidemic", infection_rate=0.7, time_horizon=5.0, validate=True)  # Automatic validation

    print(f"🧠 Smart solve completed: {smart_result.iterations} iterations")

    print("\n🎉 Demo completed! The new API makes MFG solving incredibly simple.")
    print("📖 Next steps:")
    print("   • Try different problem types and parameters")
    print("   • Check out core_objects.md for more control")
    print("   • Explore advanced_hooks.md for algorithm customization")


if __name__ == "__main__":
    main()
