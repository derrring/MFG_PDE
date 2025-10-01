#!/usr/bin/env python3
"""
Santa Fe Bar Problem - Discrete State Mean Field Game Implementation

This implements the mathematically correct discrete-state MFG formulation where:
- State space is discrete: {0: Stay Home, 1: Go to Bar}
- System reduces to coupled ODEs (no spatial derivatives)
- Uses logit choice model for transition probabilities

Mathematical Formulation:
========================

States:
- State 0: Stay home
- State 1: Go to bar
- m(t): Proportion of population at bar
- u₀(t), u₁(t): Value functions for each state
- ν > 0: Noise parameter (preference uncertainty)

HJB Equations (Backward ODEs):
    -du₁/dt = F(m(t)) - ν log(1 + exp((u₀ - u₁)/ν))
    -du₀/dt = U_home - ν log(1 + exp((u₁ - u₀)/ν))

FPK Equation (Forward ODE):
    dm/dt = (1-m(t))·P₀→₁ - m(t)·P₁→₀

Transition Probabilities (Logit Model):
    P₀→₁ = exp(u₁/ν) / (exp(u₀/ν) + exp(u₁/ν))
    P₁→₀ = exp(u₀/ν) / (exp(u₀/ν) + exp(u₁/ν))

Payoff Function:
    F(m) = G if m < m_threshold (Good time)
         = B if m ≥ m_threshold (Bad time)
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure matplotlib to avoid font warnings
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class DiscreteSantaFeBarMFG:
    """
    Discrete-state Mean Field Game for the Santa Fe Bar Problem.

    This class implements the mathematically correct formulation where the state
    space is discrete (home vs. bar) and the dynamics are governed by ODEs.
    """

    def __init__(
        self,
        T: float = 10.0,
        m_threshold: float = 0.6,
        payoff_good: float = 10.0,
        payoff_bad: float = -5.0,
        payoff_home: float = 0.0,
        noise_level: float = 1.0,
        initial_m: float = 0.5,
        initial_u0: float = 0.0,
        initial_u1: float = 0.0,
    ):
        """
        Initialize the discrete Santa Fe Bar MFG.

        Parameters:
        -----------
        T : float
            Final time horizon
        m_threshold : float
            Attendance threshold (bar becomes overcrowded above this)
        payoff_good : float
            Utility when bar is not overcrowded
        payoff_bad : float
            Utility when bar is overcrowded
        payoff_home : float
            Utility of staying home (baseline)
        noise_level : float
            Noise parameter ν (higher = more random decisions)
        initial_m : float
            Initial proportion at bar
        initial_u0 : float
            Initial value of staying home
        initial_u1 : float
            Initial value of going to bar
        """
        self.T = T
        self.m_threshold = m_threshold
        self.payoff_good = payoff_good
        self.payoff_bad = payoff_bad
        self.payoff_home = payoff_home
        self.noise_level = noise_level

        # Initial conditions
        self.initial_m = initial_m
        self.initial_u0 = initial_u0
        self.initial_u1 = initial_u1

        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized Discrete Santa Fe Bar MFG: threshold={m_threshold}, noise={noise_level}, T={T}")

    def payoff_function(self, m: float) -> float:
        """
        Payoff function F(m) for being at the bar.

        Returns good payoff if attendance below threshold, bad payoff otherwise.
        """
        return self.payoff_good if m < self.m_threshold else self.payoff_bad

    def transition_probabilities(self, u0: float, u1: float) -> tuple[float, float]:
        """
        Calculate transition probabilities using logit model.

        Parameters:
        -----------
        u0 : float
            Value of staying home
        u1 : float
            Value of going to bar

        Returns:
        --------
        Tuple[float, float]
            (P₀→₁, P₁→₀) - probabilities of switching from home to bar and vice versa
        """
        # Logit probabilities (softmax)
        exp_u0 = np.exp(u0 / self.noise_level)
        exp_u1 = np.exp(u1 / self.noise_level)
        denominator = exp_u0 + exp_u1

        P_0_to_1 = exp_u1 / denominator  # Prob of choosing bar
        P_1_to_0 = exp_u0 / denominator  # Prob of choosing home

        return P_0_to_1, P_1_to_0

    def mfg_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        The coupled ODE system for the discrete MFG.

        State vector: y = [u0, u1, m]

        Returns time derivatives: [du0/dt, du1/dt, dm/dt]
        """
        u0, u1, m = y

        # Ensure m stays in [0,1]
        m = np.clip(m, 0.0, 1.0)

        # Current payoffs
        F_m = self.payoff_function(m)

        # Transition probabilities
        P_0_to_1, P_1_to_0 = self.transition_probabilities(u0, u1)

        # HJB equations (backward time, so negative derivatives)
        # -du1/dt = F(m) - ν log(1 + exp((u0 - u1)/ν))
        log_term_1 = self.noise_level * np.log(1 + np.exp((u0 - u1) / self.noise_level))
        du1_dt = -(F_m - log_term_1)

        # -du0/dt = U_home - ν log(1 + exp((u1 - u0)/ν))
        log_term_0 = self.noise_level * np.log(1 + np.exp((u1 - u0) / self.noise_level))
        du0_dt = -(self.payoff_home - log_term_0)

        # FPK equation (forward time)
        # dm/dt = (1-m)·P₀→₁ - m·P₁→₀
        dm_dt = (1 - m) * P_0_to_1 - m * P_1_to_0

        return np.array([du0_dt, du1_dt, dm_dt])

    def solve(self, nt: int = 1000) -> dict[str, Any]:
        """
        Solve the discrete MFG system.

        Parameters:
        -----------
        nt : int
            Number of time points

        Returns:
        --------
        Dict[str, Any]
            Solution dictionary with time, values, and analysis
        """
        self.logger.info(" Solving Discrete Santa Fe Bar MFG...")

        start_time = time.time()

        # Time grid
        t_span = (0, self.T)
        t_eval = np.linspace(0, self.T, nt)

        # Initial conditions: [u0(0), u1(0), m(0)]
        y0 = np.array([self.initial_u0, self.initial_u1, self.initial_m])

        # Solve the ODE system
        try:
            solution = solve_ivp(
                self.mfg_system, t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10, max_step=0.01
            )

            if not solution.success:
                raise RuntimeError(f"ODE solver failed: {solution.message}")

        except Exception as e:
            self.logger.error(f"ERROR: Solver failed: {e}")
            raise

        solve_time = time.time() - start_time

        # Extract solutions
        t = solution.t
        u0 = solution.y[0]
        u1 = solution.y[1]
        m = solution.y[2]

        # Calculate derived quantities
        transition_probs = np.array([self.transition_probabilities(u0[i], u1[i]) for i in range(len(t))])
        P_0_to_1 = transition_probs[:, 0]
        P_1_to_0 = transition_probs[:, 1]

        # Payoffs over time
        payoffs = np.array([self.payoff_function(m[i]) for i in range(len(t))])

        self.logger.info(f"SUCCESS: Solution completed in {solve_time:.3f} seconds")
        self.logger.info(f"Final attendance: {m[-1]:.1%}, Final values: u0={u0[-1]:.3f}, u1={u1[-1]:.3f}")

        return {
            "time": t,
            "u0": u0,
            "u1": u1,
            "attendance": m,
            "P_0_to_1": P_0_to_1,
            "P_1_to_0": P_1_to_0,
            "payoffs": payoffs,
            "solve_time": solve_time,
            "converged": solution.success,
            "final_attendance": m[-1],
            "final_u0": u0[-1],
            "final_u1": u1[-1],
        }

    def analyze_equilibrium(self, solution: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze the equilibrium properties of the solution.
        """
        t = solution["time"]
        m = solution["attendance"]
        u0 = solution["u0"]
        u1 = solution["u1"]

        # Steady-state analysis (last 10% of simulation)
        steady_idx = int(0.9 * len(t))
        steady_m = m[steady_idx:]
        steady_u0 = u0[steady_idx:]
        steady_u1 = u1[steady_idx:]

        analysis = {
            "steady_state_attendance": np.mean(steady_m),
            "attendance_variance": np.var(steady_m),
            "steady_state_u0": np.mean(steady_u0),
            "steady_state_u1": np.mean(steady_u1),
            "value_difference": np.mean(steady_u1 - steady_u0),
            "overcrowding_fraction": np.mean(steady_m > self.m_threshold),
            "efficiency": 1.0 - abs(np.mean(steady_m) - self.m_threshold) / self.m_threshold,
            "convergence_achieved": np.var(steady_m) < 1e-6,
            "oscillation_amplitude": np.max(steady_m) - np.min(steady_m),
        }

        # Economic interpretation
        if analysis["steady_state_attendance"] > self.m_threshold * 1.1:
            analysis["regime"] = "overcrowded"
            analysis["interpretation"] = "Bar is consistently overcrowded"
        elif analysis["steady_state_attendance"] < self.m_threshold * 0.9:
            analysis["regime"] = "underutilized"
            analysis["interpretation"] = "Bar is underutilized due to over-caution"
        else:
            analysis["regime"] = "optimal"
            analysis["interpretation"] = "Near-optimal attendance achieved"

        return analysis


def create_santa_fe_visualizations(
    solution: dict[str, Any], analysis: dict[str, Any], problem: DiscreteSantaFeBarMFG
) -> None:
    """
    Create comprehensive visualizations for the discrete Santa Fe Bar MFG.
    """

    logger = get_logger(__name__)
    logger.info(" Creating Santa Fe Bar visualizations...")

    t = solution["time"]
    m = solution["attendance"]
    u0 = solution["u0"]
    u1 = solution["u1"]
    P_0_to_1 = solution["P_0_to_1"]
    P_1_to_0 = solution["P_1_to_0"]
    payoffs = solution["payoffs"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Santa Fe Bar Problem - Discrete State Mean Field Game", fontsize=16, fontweight="bold")

    # 1. Attendance Evolution
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t, m, "b-", linewidth=3, label="Bar Attendance")
    ax1.axhline(
        y=problem.m_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({problem.m_threshold:.0%})"
    )
    ax1.fill_between(t, m, alpha=0.3, color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Proportion at Bar")
    ax1.set_title("Bar Attendance Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Add efficiency annotation
    efficiency = analysis["efficiency"]
    ax1.text(
        0.05,
        0.95,
        f"Efficiency: {efficiency:.1%}",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.8},
    )

    # 2. Value Functions
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t, u0, "g-", linewidth=2, label="u0 (Stay Home)")
    ax2.plot(t, u1, "r-", linewidth=2, label="u1 (Go to Bar)")
    ax2.plot(t, u1 - u0, "k--", linewidth=2, label="u1 - u0 (Value Diff)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value Function")
    ax2.set_title("Value Functions Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Transition Probabilities
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t, P_0_to_1, "b-", linewidth=2, label="P(Home -> Bar)")
    ax3.plot(t, P_1_to_0, "r-", linewidth=2, label="P(Bar -> Home)")
    ax3.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label="Neutral")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Transition Probability")
    ax3.set_title("Transition Probabilities")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # 4. Payoff Function
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t, payoffs, "purple", linewidth=3, label="Bar Payoff F(m)")
    ax4.axhline(
        y=problem.payoff_good, color="green", linestyle="--", alpha=0.7, label=f"Good Payoff ({problem.payoff_good})"
    )
    ax4.axhline(
        y=problem.payoff_bad, color="red", linestyle="--", alpha=0.7, label=f"Bad Payoff ({problem.payoff_bad})"
    )
    ax4.axhline(
        y=problem.payoff_home, color="gray", linestyle=":", alpha=0.7, label=f"Home Payoff ({problem.payoff_home})"
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Payoff")
    ax4.set_title("Payoff Function Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Phase Portrait (Attendance vs Value Difference)
    ax5 = plt.subplot(2, 3, 5)
    plt.cm.viridis(np.linspace(0, 1, len(t)))
    scatter = ax5.scatter(m, u1 - u0, c=t, cmap="viridis", s=30, alpha=0.7)
    ax5.set_xlabel("Bar Attendance m(t)")
    ax5.set_ylabel("Value Difference u₁ - u₀")
    ax5.set_title("Phase Portrait: Attendance vs Value Difference")
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label="Time")

    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = f"""
Santa Fe Bar Problem Analysis

Problem Parameters:
• Threshold: {problem.m_threshold:.0%}
• Good payoff: {problem.payoff_good}
• Bad payoff: {problem.payoff_bad}
• Noise level: {problem.noise_level}

Equilibrium Results:
• Final attendance: {analysis["steady_state_attendance"]:.1%}
• Attendance variance: {analysis["attendance_variance"]:.2e}
• Value difference: {analysis["value_difference"]:.3f}
• Economic efficiency: {analysis["efficiency"]:.1%}

Behavioral Insights:
• Regime: {analysis["regime"].title()}
• Overcrowding fraction: {analysis["overcrowding_fraction"]:.1%}
• Oscillation amplitude: {analysis["oscillation_amplitude"]:.3f}
• Converged: {"Yes" if analysis["convergence_achieved"] else "No"}

Economic Interpretation:
{analysis["interpretation"]}

Mathematical Framework:
• Discrete state space (Home vs Bar)
• Coupled ODE system (no spatial derivatives)
• Logit choice model for transitions
• Noise parameter ν = {problem.noise_level}
    """

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save the figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"santa_fe_discrete_mfg_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"📁 Visualization saved as {filename}")

    plt.show()


def compare_parameter_scenarios():
    """
    Compare different parameter scenarios to understand behavioral effects.
    """

    logger = get_logger(__name__)
    logger.info("🔬 Comparing Santa Fe Bar parameter scenarios...")

    scenarios = [
        {"noise_level": 0.5, "threshold": 0.6, "name": "Low Noise"},
        {"noise_level": 2.0, "threshold": 0.6, "name": "High Noise"},
        {"noise_level": 1.0, "threshold": 0.4, "name": "Small Bar"},
        {"noise_level": 1.0, "threshold": 0.8, "name": "Large Bar"},
    ]

    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Santa Fe Bar Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")

    for i, scenario in enumerate(scenarios):
        logger.info(f"  Solving: {scenario['name']}...")

        # Create and solve problem
        problem = DiscreteSantaFeBarMFG(
            T=20.0,
            noise_level=scenario.get("noise_level", 1.0),
            m_threshold=scenario.get("threshold", 0.6),
            payoff_good=10.0,
            payoff_bad=-5.0,
        )

        solution = problem.solve(nt=2000)
        analysis = problem.analyze_equilibrium(solution)

        results[scenario["name"]] = {"solution": solution, "analysis": analysis, "problem": problem}

        # Plot results
        ax = axes[i // 2, i % 2]
        t = solution["time"]
        m = solution["attendance"]

        ax.plot(t, m, linewidth=2, label="Attendance")
        ax.axhline(
            y=problem.m_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({problem.m_threshold:.0%})",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Bar Attendance")
        ax.set_title(f"{scenario['name']}\n(ν={problem.noise_level}, threshold={problem.m_threshold:.0%})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add efficiency text
        efficiency = analysis["efficiency"]
        ax.text(
            0.05,
            0.95,
            f"Efficiency: {efficiency:.1%}",
            transform=ax.transAxes,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "yellow", "alpha": 0.7},
        )

    plt.tight_layout()
    plt.show()

    # Print comparison table
    logger.info("\n Parameter Comparison Results:")
    print("Scenario        | Noise | Threshold | Final Attend. | Efficiency | Regime")
    print("-" * 75)
    for name, result in results.items():
        analysis = result["analysis"]
        problem = result["problem"]
        print(
            f"{name:<15} | {problem.noise_level:>5.1f} | {problem.m_threshold:>9.0%} | "
            f"{analysis['steady_state_attendance']:>12.1%} | {analysis['efficiency']:>9.1%} | "
            f"{analysis['regime']}"
        )

    return results


def main():
    """
    Main function to demonstrate the discrete Santa Fe Bar MFG.
    """

    configure_research_logging("santa_fe_discrete_mfg", level="INFO")
    logger = get_logger(__name__)

    logger.info("🍺 Santa Fe Bar Problem - Discrete State Mean Field Game")
    logger.info("=" * 70)

    try:
        # Basic problem solution
        logger.info("1. Solving basic Santa Fe Bar Problem...")

        problem = DiscreteSantaFeBarMFG(
            T=20.0, m_threshold=0.6, payoff_good=10.0, payoff_bad=-5.0, payoff_home=0.0, noise_level=1.0, initial_m=0.3
        )

        solution = problem.solve(nt=2000)
        analysis = problem.analyze_equilibrium(solution)

        logger.info("2. Creating visualizations...")
        create_santa_fe_visualizations(solution, analysis, problem)

        logger.info("3. Running parameter comparison...")
        compare_parameter_scenarios()

        logger.info("SUCCESS: Santa Fe Bar analysis completed successfully!")

        # Print key insights
        logger.info("\n Key Economic Insights:")
        logger.info(f"  • Final attendance rate: {analysis['steady_state_attendance']:.1%}")
        logger.info(f"  • Economic efficiency: {analysis['efficiency']:.1%}")
        logger.info(f"  • Regime: {analysis['regime']}")
        logger.info(f"  • Convergence: {'Yes' if analysis['convergence_achieved'] else 'No'}")
        logger.info(f"  • Value difference: {analysis['value_difference']:.3f}")

        logger.info("\n📚 This discrete MFG formulation correctly captures:")
        logger.info("   → Binary choice structure (home vs. bar)")
        logger.info("   → Logit choice probabilities with noise")
        logger.info("   → Coupled ODE dynamics (no spatial derivatives)")
        logger.info("   → Threshold effects in payoff structure")

    except Exception as e:
        logger.error(f"ERROR: Error in Santa Fe Bar analysis: {e}")
        raise


if __name__ == "__main__":
    main()
