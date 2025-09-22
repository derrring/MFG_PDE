#!/usr/bin/env python3
"""
Real-World MFG Benchmark Problems

This module provides a collection of realistic MFG problems for benchmarking
AMR-enhanced solvers. These problems are designed to test AMR effectiveness
on practical applications with known characteristics.

Problem Collection:
- Traffic flow with bottlenecks and varying road capacities
- Financial markets with volatility clustering
- Crowd dynamics with obstacles and exit strategies
- Energy trading with demand/supply shocks
- Urban planning with zoning and infrastructure
- Epidemic spreading with heterogeneous population density
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MFG_PDE imports
from mfg_pde import ExampleMFGProblem
from mfg_pde.geometry import Domain1D, dirichlet_bc, periodic_bc


@dataclass
class ProblemSpecification:
    """Specification for a real-world MFG benchmark problem."""

    name: str
    description: str
    dimension: int
    expected_features: list[str]  # e.g., ["sharp_gradients", "localized_density"]
    amr_advantage: str  # Why AMR should help
    baseline_grid_size: tuple[int, ...]
    reference_solution_method: str
    physical_interpretation: str


class TrafficFlowProblem(ExampleMFGProblem):
    """
    Traffic flow with varying road capacity and bottlenecks.

    This problem models vehicle flow on a highway with:
    - Variable speed limits creating bottlenecks
    - Entrance/exit ramps affecting density
    - Rush hour demand patterns

    AMR advantage: Sharp density changes at bottlenecks and ramps.
    """

    def __init__(self, highway_length: float = 10.0, rush_hour_intensity: float = 2.0):
        # Highway from 0 to highway_length km
        domain = Domain1D(0.0, highway_length, dirichlet_bc())

        super().__init__(
            T=2.0,  # 2 hour simulation
            xmin=0.0,
            xmax=highway_length,
            Nx=128,
            Nt=60,
            sigma=0.02,  # Low diffusion (vehicles follow traffic flow)
            coefCT=rush_hour_intensity,  # Congestion cost
        )

        self.domain = domain
        self.dimension = 1
        self.highway_length = highway_length
        self.rush_hour_intensity = rush_hour_intensity

    def capacity_function(self, x: np.ndarray) -> np.ndarray:
        """Variable highway capacity with bottlenecks."""
        capacity = np.ones_like(x)

        # Bottleneck at x = 3 km (bridge)
        bridge_location = 3.0
        bridge_width = 0.2
        capacity *= 0.5 + 0.5 * (1 + np.tanh((np.abs(x - bridge_location) - bridge_width) / 0.05))

        # Construction zone at x = 7 km
        construction_start, construction_end = 6.5, 7.5
        construction_mask = (x >= construction_start) & (x <= construction_end)
        capacity[construction_mask] *= 0.3  # Severe capacity reduction

        return capacity

    def entrance_ramp_flow(self, x: np.ndarray, t: float) -> np.ndarray:
        """Traffic entering from on-ramps during rush hour."""
        inflow = np.zeros_like(x)

        # Major on-ramp at x = 2 km
        ramp_location = 2.0
        ramp_width = 0.1
        ramp_intensity = 0.5 * self.rush_hour_intensity

        # Rush hour pattern (morning: t < 1, evening: t > 1.5)
        if t < 1.0:  # Morning rush
            time_factor = np.sin(np.pi * t)
        elif t > 1.5:  # Evening rush
            time_factor = np.sin(np.pi * (t - 1.5))
        else:  # Midday
            time_factor = 0.1

        ramp_flow = ramp_intensity * time_factor * np.exp(-(((x - ramp_location) / ramp_width) ** 2))
        inflow += ramp_flow

        return inflow


class FinancialMarketProblem(ExampleMFGProblem):
    """
    Financial market with volatility clustering and liquidity shocks.

    Models trader behavior in a market with:
    - Heterogeneous risk aversion
    - Volatility clustering around market events
    - Liquidity drying up during stress

    AMR advantage: Sharp price movements and density clustering around equilibrium.
    """

    def __init__(self, price_range: float = 4.0, volatility_factor: float = 1.5):
        # Asset price domain (log-scaled)
        domain = Domain1D(-price_range / 2, price_range / 2, periodic_bc())

        super().__init__(
            T=1.0,  # 1 trading day
            xmin=-price_range / 2,
            xmax=price_range / 2,
            Nx=96,
            Nt=80,
            sigma=0.15,  # Market volatility
            coefCT=volatility_factor,  # Price impact intensity
        )

        self.domain = domain
        self.dimension = 1
        self.price_range = price_range
        self.volatility_factor = volatility_factor

    def volatility_surface(self, x: np.ndarray) -> np.ndarray:
        """Market volatility with clustering around stress points."""
        base_volatility = 0.2

        # Volatility smile effect
        moneyness_effect = 0.1 * x**2

        # Stress clustering around x = 0 (at-the-money)
        stress_effect = 0.3 * np.exp(-(x**2) / 0.1)

        return base_volatility + moneyness_effect + stress_effect

    def liquidity_function(self, x: np.ndarray, t: float) -> np.ndarray:
        """Time-varying liquidity with market stress events."""
        base_liquidity = np.ones_like(x)

        # Market stress event at t = 0.6 (3pm market close effect)
        if t > 0.5:
            stress_intensity = 2.0 * np.exp(-((t - 0.6) ** 2) / 0.02)
            liquidity_reduction = stress_intensity * np.exp(-(x**2) / 0.5)
            base_liquidity *= 1 - 0.7 * liquidity_reduction

        return np.maximum(base_liquidity, 0.1)  # Minimum liquidity floor


class CrowdDynamicsProblem(ExampleMFGProblem):
    """
    Crowd evacuation with obstacles and multiple exits.

    Models pedestrian movement in an emergency evacuation:
    - Fixed obstacles (pillars, furniture)
    - Multiple exit points with varying attractiveness
    - Panic-driven behavior with congestion aversion

    AMR advantage: Sharp density gradients around obstacles and exits.
    """

    def __init__(self, corridor_length: float = 20.0, panic_factor: float = 3.0):
        # Corridor from 0 to corridor_length meters
        domain = Domain1D(0.0, corridor_length, dirichlet_bc())

        super().__init__(
            T=0.5,  # 30 seconds evacuation
            xmin=0.0,
            xmax=corridor_length,
            Nx=80,
            Nt=40,
            sigma=0.5,  # Higher diffusion (people spreading)
            coefCT=panic_factor,  # Panic-driven congestion cost
        )

        self.domain = domain
        self.dimension = 1
        self.corridor_length = corridor_length
        self.panic_factor = panic_factor

    def obstacle_configuration(self, x: np.ndarray) -> np.ndarray:
        """Fixed obstacles that impede crowd flow."""
        mobility = np.ones_like(x)

        # Pillar at x = 8m
        pillar_location = 8.0
        pillar_width = 0.5
        pillar_effect = np.exp(-(((x - pillar_location) / pillar_width) ** 2))
        mobility *= 1 - 0.8 * pillar_effect

        # Furniture cluster at x = 15m
        furniture_start, furniture_end = 14.5, 15.5
        furniture_mask = (x >= furniture_start) & (x <= furniture_end)
        mobility[furniture_mask] *= 0.3

        return mobility

    def exit_attractiveness(self, x: np.ndarray) -> np.ndarray:
        """Multiple exits with different attractiveness."""
        attraction = np.zeros_like(x)

        # Main exit at x = 20m (end of corridor)
        main_exit_attraction = 2.0 * np.exp(-(((x - self.corridor_length) / 1.0) ** 2))
        attraction += main_exit_attraction

        # Emergency exit at x = 5m (less attractive but closer)
        emergency_exit_attraction = 0.8 * np.exp(-(((x - 5.0) / 0.8) ** 2))
        attraction += emergency_exit_attraction

        return attraction


class EnergyTradingProblem(ExampleMFGProblem):
    """
    Electricity market with renewable intermittency and demand shocks.

    Models energy trader strategies with:
    - Renewable generation uncertainty (wind/solar)
    - Peak demand periods
    - Storage arbitrage opportunities

    AMR advantage: Sharp price spikes during supply/demand imbalances.
    """

    def __init__(self, price_volatility: float = 2.0):
        # Energy price domain ($/MWh, centered around typical market price)
        domain = Domain1D(-100.0, 200.0, dirichlet_bc())

        super().__init__(
            T=1.0,  # 24-hour trading day
            xmin=-100.0,
            xmax=200.0,
            Nx=120,
            Nt=96,
            sigma=0.3,  # Price volatility
            coefCT=price_volatility,  # Market impact
        )

        self.domain = domain
        self.dimension = 1
        self.price_volatility = price_volatility

    def renewable_intermittency(self, t: float) -> float:
        """Renewable generation variability throughout the day."""
        # Solar generation pattern (peaks at midday)
        solar_pattern = np.maximum(0, np.sin(np.pi * t))

        # Wind generation (more random, but correlated with weather patterns)
        wind_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * t + 0.5)

        # Combined renewable factor
        renewable_factor = 0.6 * solar_pattern + 0.4 * wind_pattern

        return renewable_factor

    def demand_pattern(self, t: float) -> float:
        """Electricity demand with peak/off-peak periods."""
        # Base demand
        base_demand = 1.0

        # Peak demand periods (morning: t~0.3, evening: t~0.8)
        morning_peak = 0.4 * np.exp(-(((t - 0.3) / 0.1) ** 2))
        evening_peak = 0.6 * np.exp(-(((t - 0.8) / 0.1) ** 2))

        total_demand = base_demand + morning_peak + evening_peak

        return total_demand


class EpidemicSpreadProblem(ExampleMFGProblem):
    """
    Epidemic spreading with population heterogeneity and control measures.

    Models individual behavior during epidemic with:
    - Heterogeneous population density
    - Social distancing measures
    - Economic vs health trade-offs

    AMR advantage: Sharp transitions at policy boundaries and density hotspots.
    """

    def __init__(self, social_distancing_strength: float = 1.5):
        # Spatial domain representing city/region
        domain = Domain1D(0.0, 10.0, periodic_bc())

        super().__init__(
            T=1.0,  # Epidemic time horizon
            xmin=0.0,
            xmax=10.0,
            Nx=100,
            Nt=50,
            sigma=0.1,  # Population mobility
            coefCT=social_distancing_strength,  # Social distancing cost
        )

        self.domain = domain
        self.dimension = 1
        self.social_distancing_strength = social_distancing_strength

    def population_density_profile(self, x: np.ndarray) -> np.ndarray:
        """Heterogeneous population density (urban vs suburban)."""
        density = np.ones_like(x)

        # Urban center at x = 3-5
        urban_center = (x >= 3.0) & (x <= 5.0)
        density[urban_center] *= 3.0

        # Suburban areas
        suburban = ((x >= 1.0) & (x < 3.0)) | ((x > 5.0) & (x <= 8.0))
        density[suburban] *= 1.5

        # Rural areas (default density = 1.0)

        return density

    def infection_risk_function(self, x: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Location-dependent infection risk."""
        base_risk = 0.1 * density  # Risk proportional to density

        # High-risk locations (hospitals, transport hubs)
        hospital_location = 4.0
        hospital_risk = 0.3 * np.exp(-(((x - hospital_location) / 0.3) ** 2))

        transport_hub = 6.5
        transport_risk = 0.2 * np.exp(-(((x - transport_hub) / 0.4) ** 2))

        total_risk = base_risk + hospital_risk + transport_risk

        return total_risk


class RealWorldMFGProblems:
    """Collection of real-world MFG benchmark problems."""

    def __init__(self):
        self.problems = {
            "traffic_flow": ProblemSpecification(
                name="Highway Traffic Flow",
                description="Vehicle flow with bottlenecks and variable capacity",
                dimension=1,
                expected_features=["sharp_gradients", "localized_density", "bottlenecks"],
                amr_advantage="Sharp density changes at bottlenecks require fine resolution",
                baseline_grid_size=(128,),
                reference_solution_method="high_resolution_uniform",
                physical_interpretation="Vehicles/km density and optimal speed policy",
            ),
            "financial_market": ProblemSpecification(
                name="Financial Market Trading",
                description="Asset pricing with volatility clustering and liquidity shocks",
                dimension=1,
                expected_features=["volatility_clustering", "sharp_transitions", "price_jumps"],
                amr_advantage="Volatility clustering creates localized high-gradient regions",
                baseline_grid_size=(96,),
                reference_solution_method="fine_grid_accurate",
                physical_interpretation="Trader density and optimal trading strategy",
            ),
            "crowd_dynamics": ProblemSpecification(
                name="Emergency Evacuation",
                description="Crowd evacuation with obstacles and multiple exits",
                dimension=1,
                expected_features=["obstacles", "exit_clustering", "panic_behavior"],
                amr_advantage="Sharp density gradients around obstacles and exits",
                baseline_grid_size=(80,),
                reference_solution_method="obstacle_aware_refinement",
                physical_interpretation="People density and optimal evacuation paths",
            ),
            "energy_trading": ProblemSpecification(
                name="Electricity Market",
                description="Energy trading with renewable intermittency",
                dimension=1,
                expected_features=["price_spikes", "intermittent_supply", "demand_peaks"],
                amr_advantage="Price spikes require high resolution for accurate capture",
                baseline_grid_size=(120,),
                reference_solution_method="high_temporal_resolution",
                physical_interpretation="Energy price distribution and trading strategies",
            ),
            "epidemic_spread": ProblemSpecification(
                name="Epidemic Control",
                description="Individual behavior during epidemic with control measures",
                dimension=1,
                expected_features=["policy_boundaries", "density_hotspots", "behavioral_transitions"],
                amr_advantage="Sharp transitions at policy boundaries and population centers",
                baseline_grid_size=(100,),
                reference_solution_method="multi_scale_accurate",
                physical_interpretation="Population spatial distribution and mobility choices",
            ),
        }

    def create_problem(self, problem_type: str, **kwargs) -> ExampleMFGProblem:
        """Create a specific real-world MFG problem."""

        if problem_type == "traffic_flow":
            return TrafficFlowProblem(**kwargs)
        elif problem_type == "financial_market":
            return FinancialMarketProblem(**kwargs)
        elif problem_type == "crowd_dynamics":
            return CrowdDynamicsProblem(**kwargs)
        elif problem_type == "energy_trading":
            return EnergyTradingProblem(**kwargs)
        elif problem_type == "epidemic_spread":
            return EpidemicSpreadProblem(**kwargs)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def get_problem_specification(self, problem_type: str) -> ProblemSpecification:
        """Get specification for a problem type."""
        if problem_type not in self.problems:
            raise ValueError(f"Unknown problem type: {problem_type}")
        return self.problems[problem_type]

    def list_available_problems(self) -> list[str]:
        """List all available problem types."""
        return list(self.problems.keys())

    def create_benchmark_suite(self) -> dict[str, ExampleMFGProblem]:
        """Create the complete benchmark suite with default parameters."""
        suite = {}

        for problem_type in self.problems.keys():
            try:
                problem = self.create_problem(problem_type)
                suite[problem_type] = problem
                print(f"✓ Created {problem_type} problem")
            except Exception as e:
                print(f"✗ Failed to create {problem_type}: {e}")

        return suite

    def generate_problem_report(self, output_dir: str = "benchmark_problems"):
        """Generate detailed report of all benchmark problems."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "real_world_problems_report.md"

        with open(report_file, "w") as f:
            f.write("# Real-World MFG Benchmark Problems Report\n\n")
            f.write("This document describes the collection of real-world MFG problems\n")
            f.write("designed for benchmarking AMR-enhanced solvers.\n\n")

            for problem_type, spec in self.problems.items():
                f.write(f"## {spec.name}\n\n")
                f.write(f"**Problem Type**: `{problem_type}`  \n")
                f.write(f"**Description**: {spec.description}  \n")
                f.write(f"**Dimension**: {spec.dimension}D  \n")
                f.write(f"**Baseline Grid**: {spec.baseline_grid_size}  \n\n")

                f.write("**Expected Features**:\n")
                for feature in spec.expected_features:
                    f.write(f"- {feature.replace('_', ' ').title()}\n")
                f.write("\n")

                f.write(f"**AMR Advantage**: {spec.amr_advantage}  \n")
                f.write(f"**Physical Interpretation**: {spec.physical_interpretation}  \n\n")

                # Create and analyze problem
                try:
                    problem = self.create_problem(problem_type)
                    f.write("**Problem Parameters**:\n")
                    f.write(f"- Time horizon: {problem.T}\n")
                    f.write(f"- Spatial domain: [{problem.xmin}, {problem.xmax}]\n")
                    f.write(f"- Grid size: {problem.Nx} × {problem.Nt}\n")
                    f.write(f"- Diffusion: σ = {problem.sigma}\n")
                    f.write(f"- Congestion: λ = {problem.coefCT}\n\n")

                except Exception as e:
                    f.write(f"**Error creating problem**: {e}\n\n")

                f.write("---\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"**Total Problems**: {len(self.problems)}  \n")
            f.write(f"**Dimensions Covered**: {set(spec.dimension for spec in self.problems.values())}  \n")
            f.write(
                f"**Feature Categories**: {len(set(feature for spec in self.problems.values() for feature in spec.expected_features))}  \n\n"
            )

            # Usage examples
            f.write("## Usage Examples\n\n")
            f.write("```python\n")
            f.write("from mfg_pde.benchmarks.real_world_problems import RealWorldMFGProblems\n\n")
            f.write("# Create problem collection\n")
            f.write("problems = RealWorldMFGProblems()\n\n")
            f.write("# Create specific problem\n")
            f.write("traffic_problem = problems.create_problem('traffic_flow')\n\n")
            f.write("# Create complete benchmark suite\n")
            f.write("benchmark_suite = problems.create_benchmark_suite()\n")
            f.write("```\n")

        print(f"Problem report generated: {report_file}")


def main():
    """Generate the real-world problem collection and report."""
    problems = RealWorldMFGProblems()

    print("Real-World MFG Benchmark Problems")
    print("=" * 50)

    # List available problems
    available_problems = problems.list_available_problems()
    print(f"Available problems: {', '.join(available_problems)}")

    # Create benchmark suite
    print("\nCreating benchmark suite...")
    suite = problems.create_benchmark_suite()

    print(f"\nSuccessfully created {len(suite)} benchmark problems.")

    # Generate report
    problems.generate_problem_report()

    print("\n✅ Real-world benchmark problems ready!")


if __name__ == "__main__":
    main()
