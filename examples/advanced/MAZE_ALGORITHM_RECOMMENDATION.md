# Maze Algorithm Assessment for MFG-RL Experiments

**Decision:** Recursive Backtracking (DFS) vs Wilson's Algorithm

**Date:** October 2025
**Author:** MFG_PDE Team
**Reference:** Jamis Buck, "Mazes for Programmers" (2015)

---

## Executive Summary

Based on rigorous empirical analysis of 10 trials per algorithm on 20x20 grids:

**RECOMMENDATION: Use BOTH algorithms for complementary research objectives**

- **Recursive Backtracking (DFS)**: For high-difficulty exploration challenges
- **Wilson's Algorithm**: For structural diversity and decision-making experiments

---

## Empirical Results (20x20 Grid, 10 Trials)

### Performance Metrics

| Metric | Recursive Backtracking | Wilson's | Winner |
|:-------|:----------------------|:---------|:-------|
| **Generation Speed** | 0.0007s | 0.0076s | **RB (11x faster)** |
| **Dead Ends** | 42.3 | 116.7 | Wilson's (2.8x more) |
| **Junctions** | 39.9 | 100.2 | **Wilson's (2.5x more)** |
| **Longest Path** | 234.7 cells | 93.7 cells | **RB (2.5x longer)** |
| **Avg Path Length** | 81.6 cells | 33.1 cells | **RB (2.5x longer)** |
| **Branching Entropy** | 0.942 | 1.682 | **Wilson's (1.79x higher)** |
| **Structural Bias** | 0.002 | 0.002 | **Tied (both unbiased)** |
| **Exploration Difficulty** | 2.041 | 0.828 | **RB (2.5x harder)** |

### Statistical Significance

All differences statistically significant (p < 0.0001) **except structural bias** (p = 0.977).

Both algorithms produce **perfectly unbiased mazes** - no directional preferences.

---

## Algorithm Characteristics

### Recursive Backtracking (DFS)

**Visual Pattern:**
- Long, winding corridors
- Few dead ends (10.6% of cells)
- Low junction count (10% of cells)
- High corridor density (79.5% of cells)

**Maze Topology:**
- **High exploration difficulty** (2.04x grid dimension)
- **Long detour paths** between any two points
- **Sequential structure** - resembles depth-first traversal
- **Low branching** - few decision points

**Best For:**
- Testing agent persistence in long sequences
- Reward delay tolerance experiments
- Credit assignment over long horizons
- Memory-based navigation strategies

**Generation:**
- **Very fast** (0.7ms for 20x20)
- Deterministic from seed
- Simple implementation

---

### Wilson's Algorithm

**Visual Pattern:**
- More balanced structure
- Many dead ends (29.2% of cells)
- High junction count (25% of cells)
- Moderate corridor density (45.8% of cells)

**Maze Topology:**
- **Lower exploration difficulty** (0.83x grid dimension)
- **Shorter average paths** between points
- **High branching entropy** (1.68 bits)
- **Many decision points** - frequent choices

**Best For:**
- Multi-agent coordination (more paths)
- Decision-making under uncertainty
- Exploration-exploitation tradeoffs
- Strategic planning experiments

**Generation:**
- **Slower** (7.6ms for 20x20, still fast)
- **Truly unbiased** - all possible mazes equally likely
- Loop-erased random walk algorithm

---

## Detailed Analysis for MFG-RL

### 1. Exploration Difficulty

**Recursive Backtracking:**
- Average path = 81.6 cells (2.04x grid size 40)
- Agents must commit to long corridors
- High cost of wrong decisions
- **Tests:** Long-term planning, patience

**Wilson's:**
- Average path = 33.1 cells (0.83x grid size)
- More alternative routes
- Lower penalty for mistakes
- **Tests:** Quick adaptation, local optimization

---

### 2. Decision Points and Learning

**Recursive Backtracking:**
- 39.9 junctions (10% of cells)
- Sparse decision points
- **RL Challenge:** Long sequences between choices
- **Suitable for:** LSTM/GRU agents, temporal credit assignment

**Wilson's:**
- 100.2 junctions (25% of cells)
- Frequent decision points
- **RL Challenge:** Many local optima
- **Suitable for:** Value-based methods, policy gradients

---

### 3. Structural Diversity

**Branching Entropy Comparison:**

- **Recursive Backtracking:** 0.942 bits
  - More predictable structure
  - Passage distribution: mostly corridors
  - Lower structural variety

- **Wilson's:** 1.682 bits (1.79x higher)
  - Higher structural complexity
  - More balanced passage distribution
  - Greater variety across maze instances

**Implication for RL:**
- RB: Tests generalization across similar topologies
- Wilson's: Tests generalization across diverse structures

---

### 4. Multi-Agent Mean Field Games

**Recursive Backtracking:**
- Long corridors create **bottlenecks**
- Strong mean-field effects (congestion)
- Few alternative paths
- **Good for:** Studying coordination in constraints

**Wilson's:**
- More paths reduce bottlenecks
- Weaker mean-field coupling
- Alternative routes available
- **Good for:** Studying strategic path selection

---

### 5. Computational Performance

**Recursive Backtracking:**
- **11x faster** generation (0.7ms vs 7.6ms)
- Scales to larger grids efficiently
- **Use for:** Large-scale experiments, grid sweeps

**Wilson's:**
- Still very fast (7.6ms for 20x20)
- Computational cost justified by unbiased sampling
- **Use for:** Controlled experiments, benchmarks

---

## Recommendations by Research Objective

### 1. **For Page 45 Figure Reproduction:**

**Primary Choice:** **Recursive Backtracking**

**Rationale:**
- Creates challenging exploration scenario
- Long paths test RL convergence
- Fast generation for repeated trials
- Clear "hard exploration" benchmark

**Alternative:** Wilson's for diversity baseline

---

### 2. **For Comprehensive RL Study:**

**Use BOTH** in comparative framework:

```python
# Complementary experiment design
experiments = [
    {
        "maze_algorithm": "recursive_backtracking",
        "objective": "Long-horizon credit assignment",
        "expected_difficulty": "HIGH"
    },
    {
        "maze_algorithm": "wilsons",
        "objective": "Decision-making diversity",
        "expected_difficulty": "MEDIUM"
    }
]
```

**Analysis:**
- Compare convergence rates across algorithms
- Identify algorithm-specific challenges
- Test generalization across topologies

---

### 3. **For Mean Field Game Experiments:**

**Primary Choice:** **Recursive Backtracking**

**Rationale:**
- Strong bottleneck effects
- Clear mean-field coupling
- Congestion-driven dynamics
- Better tests MFG theory predictions

**Secondary:** Wilson's for comparison with weak coupling

---

### 4. **For Methodological Comparisons:**

**Use Wilson's** as **canonical baseline**

**Rationale:**
- Truly unbiased sampling
- No algorithmic artifacts
- Every possible maze equally likely
- Gold standard for fairness

---

## Implementation Guidance

### Quick Start (Single Algorithm)

```python
from perfect_maze_generator import PerfectMazeGenerator, MazeAlgorithm

# For challenging exploration
generator = PerfectMazeGenerator(
    rows=20,
    cols=20,
    algorithm=MazeAlgorithm.RECURSIVE_BACKTRACKING
)
grid = generator.generate(seed=42)
maze_array = generator.to_numpy_array()

# For structural diversity
generator = PerfectMazeGenerator(
    rows=20,
    cols=20,
    algorithm=MazeAlgorithm.WILSONS
)
grid = generator.generate(seed=42)
```

### Comparative Study

```python
from page45_perfect_maze_demo import Page45MazeExperiment, ExperimentConfig

for algorithm in [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS]:
    config = ExperimentConfig(
        maze_rows=20,
        maze_cols=20,
        num_agents=50,
        algorithm=algorithm,
        seed=42
    )

    experiment = Page45MazeExperiment(config)
    results = experiment.run_experiment()

    print(f"{algorithm.value}: Success rate = {results['success_rate']:.2%}")
```

---

## Statistical Validity

All comparisons based on:
- **Sample size:** 10 independent mazes per algorithm
- **Grid size:** 20x20 (400 cells)
- **Statistical test:** Independent t-tests
- **Significance level:** α = 0.05

**Confidence:** All performance differences are highly significant (p < 0.0001).

---

## Final Verdict

### Single Algorithm Choice

**Choose Recursive Backtracking** if you need:
- ✅ One algorithm only
- ✅ Challenging exploration
- ✅ Fast generation
- ✅ Clear difficulty benchmark

### Dual Algorithm Approach

**Use BOTH** for:
- ✅ Comprehensive study
- ✅ Topology-agnostic conclusions
- ✅ Algorithm robustness testing
- ✅ Methodological rigor

---

## Key Insight

**Perfect mazes guarantee solvability, but structural properties matter:**

- **Recursive Backtracking:** Tests agent *persistence* and *long-term planning*
- **Wilson's Algorithm:** Tests agent *decision-making* and *adaptability*

Both are scientifically valid. Your choice should align with your research question.

---

## References

1. Jamis Buck. *Mazes for Programmers*. Pragmatic Bookshelf, 2015.
2. Implementation: `examples/advanced/perfect_maze_generator.py`
3. Assessment code: `examples/advanced/maze_algorithm_assessment.py`
4. Quick comparison: `examples/advanced/quick_maze_assessment.py`

---

**Generated:** October 2025
**Status:** Production-ready recommendation based on empirical analysis
