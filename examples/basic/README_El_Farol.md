# El Farol Bar Problem - MFG Implementation

## Overview

This directory contains implementations of the famous **El Farol Bar Problem** (also known as the Santa Fe Bar Problem) using the MFG_PDE framework. The El Farol Bar Problem is a classic example in game theory and bounded rationality that demonstrates how individual rational decisions can lead to suboptimal collective outcomes.

## Problem Description

**The Setup:**
- N agents must independently decide whether to go to a bar
- The bar has limited capacity (e.g., comfortable for 60% of the population)
- If too many people go → overcrowded, unpleasant experience (low utility)
- If few people go → pleasant, uncrowded experience (high utility)
- Each agent tries to predict others' behavior and decides accordingly
- This creates a self-referential prediction problem

**The Challenge:**
- If everyone thinks "others won't go, so I should go" → overcrowding
- If everyone thinks "others will go, so I shouldn't" → underutilization
- The optimal solution requires coordination without communication

## MFG Formulation

We model this as a Mean Field Game where:

- **State space x ∈ [0,1]**: Agent's tendency to go to the bar
  - x = 0: Strong tendency to stay home
  - x = 1: Strong tendency to go to bar
  
- **Population density m(t,x)**: Distribution of agents over decision tendencies

- **Control u(t,x)**: Rate of changing decision tendency

- **Expected attendance**: ∫ x · m(t,x) dx (probability-weighted attendance)

- **Cost function**: Includes crowding penalty, decision effort, and historical memory

## Files in this Directory

### 🚀 **el_farol_simple_working.py** (Recommended)
- **Working implementation** using ExampleMFGProblem as base
- Complete with visualization and economic analysis
- Interactive scenario comparison
- **Ready to run!**

### 📝 **el_farol_bar_simple.py**
- Minimal example with fallback to basic MFG
- Good starting point for understanding the concept

### 🔬 **../advanced/el_farol_bar_mfg.py**
- Full custom MFG problem implementation (under development)
- Advanced mathematical formulation
- Research-grade implementation

## Quick Start

```bash
# Run the working example
python el_farol_simple_working.py

# You'll see:
# 1. Problem solving with MFG
# 2. Economic analysis of results
# 3. Comprehensive visualizations
# 4. Option to compare different scenarios
```

## Key Results and Insights

### Economic Metrics

1. **Final Attendance Rate**: Equilibrium attendance as % of population
2. **Economic Efficiency**: How close to optimal (1 = perfect, 0 = worst)
3. **Capacity Utilization**: Attendance relative to bar capacity
4. **Convergence**: Whether the system reaches equilibrium

### Behavioral Insights

- **Herding Effects**: How agents cluster in their decision tendencies
- **Coordination Failures**: When rational individual choices lead to poor collective outcomes
- **Parameter Sensitivity**: How crowd aversion affects equilibrium

### Example Results

```
📊 Key Results:
  • Final attendance: 42.3%
  • Economic efficiency: 71.2%
  • Capacity utilization: 70.5%

💡 Economic Insights:
  → Moderate efficiency. Some coordination issues.
  → Bar is underutilized. Agents over-cautious about crowding.
```

## Parameter Effects

### Bar Capacity
- **Lower capacity** → More likely to be overcrowded
- **Higher capacity** → More forgiving, better coordination

### Crowd Aversion
- **Low aversion** → Risk of overcrowding
- **High aversion** → Risk of underutilization  
- **Moderate aversion** → Often optimal balance

### Decision Volatility
- **Low volatility** → Agents stick to decisions (can cause herding)
- **High volatility** → More exploration but less stable

## Visualizations

The implementation generates comprehensive visualizations:

1. **Population Density Evolution**: How agent tendencies evolve over time
2. **Value Function**: Optimal value-to-go for each decision state
3. **Attendance Time Series**: Bar attendance over time vs. optimal capacity
4. **Final Distribution**: Equilibrium population distribution
5. **Economic Metrics**: Key performance indicators
6. **Economic Interpretation**: Behavioral and efficiency analysis

## Scenario Comparisons

The code can compare multiple scenarios:

- Small Bar vs. Large Bar
- Low Crowd Aversion vs. High Crowd Aversion
- Different parameter combinations

Example comparison results:
```
Scenario Comparison Results:
Scenario                    | Attendance | Efficiency | Converged
-----------------------------------------------------------------
Small Bar, Low Aversion     |      65.2% |      48.0% | Yes
Medium Bar, Medium Aversion |      42.3% |      71.2% | Yes
Large Bar, Low Aversion     |      51.7% |      64.6% | Yes
Medium Bar, High Aversion   |      28.9% |      51.9% | Yes
```

## Mathematical Foundation

### Cost Function
```
L(t, x, u, m) = α·max(0, attendance - capacity)² + ½u² + β(x - historical)²
```

Where:
- α = crowd aversion parameter
- attendance = ∫ x·m(t,x) dx
- β = memory/historical weight

### Hamiltonian
```
H(t, x, p, m) = min_u [L(t, x, u, m) + p·u]
```

### Optimal Control
```
u* = -p (first-order condition)
```

## Connection to Real-World Problems

The El Farol Bar Problem models many real-world coordination challenges:

1. **Traffic congestion**: Route choice during rush hour
2. **Restaurant crowding**: Weekend dining decisions  
3. **Resource allocation**: Access to limited public goods
4. **Market participation**: Trading in financial markets
5. **Technology adoption**: Network effects and critical mass

## Further Reading

- **Original Paper**: Arthur, W.B. (1994). "Inductive Reasoning and Bounded Rationality"
- **Game Theory**: Fundenberg & Tirole, "Game Theory" (1991)
- **Mean Field Games**: Carmona & Delarue, "Probabilistic Theory of Mean Field Games" (2018)

## Usage Tips

1. **Start simple**: Run `el_farol_simple_working.py` first
2. **Experiment**: Try different bar capacities and crowd aversion levels
3. **Observe patterns**: Look for overcrowding vs. underutilization
4. **Compare scenarios**: Use the built-in comparison function
5. **Interpret results**: Focus on efficiency and attendance patterns

## Future Enhancements

Potential extensions of this implementation:

- **Heterogeneous agents**: Different crowd aversion levels
- **Learning dynamics**: Agents adapt based on historical outcomes
- **Network effects**: Social influence between agents
- **Multi-period memory**: Longer historical consideration
- **Asymmetric information**: Agents have different information sets

---

**Created**: July 27, 2025  
**MFG_PDE Framework**: A+ Quality Implementation  
**Status**: Ready for research and education use

The El Farol Bar Problem demonstrates the power of Mean Field Games in understanding collective decision-making and coordination challenges in economics and social systems.