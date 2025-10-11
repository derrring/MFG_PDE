# Hybrid Maze Generation for MFG Research

**Document Type**: Technical Design Document
**Status**: Proposed Implementation
**Created**: October 2025
**Author**: MFG_PDE Team

---

## 🎯 **Executive Summary**

Hybrid maze generation combines multiple maze algorithms to create realistic, heterogeneous environments that better model real-world MFG scenarios. This document outlines the design, benefits, and implementation strategy for hybrid maze generation.

**Key Innovation**: Enable multi-zone environments where different spatial structures coexist (e.g., structured office areas + organic courtyards).

---

## 💡 **Motivation & Research Value**

### **Real-World Buildings Are Hybrid**

Real architectural spaces combine multiple spatial patterns:
- **Office buildings**: Structured rooms (Recursive Division) + service corridors (Perfect Maze)
- **Museums**: Irregular galleries (Voronoi) + garden courtyards (Cellular Automata)
- **Shopping malls**: Store spaces (Voronoi) + connecting hallways (Recursive Division)
- **Hospitals**: Large wards (Voronoi) + narrow corridors (Perfect Maze)
- **Campuses**: Multiple buildings with different interior structures

### **MFG Research Benefits**

**1. Zone-Specific Behavior Analysis**
```
Research Question: How do agents adapt strategies across different spatial structures?

Scenario: Museum with galleries (Voronoi) + gardens (CA)
- Gallery behavior: Navigate around exhibits (structured)
- Garden behavior: Free exploration (organic)
- Transition dynamics: Strategy adaptation at boundaries
```

**2. Heterogeneous Nash Equilibria**
```
Research Question: Do different zones lead to different equilibrium strategies?

Scenario: Multi-building campus
- North wing (Office, RD): Corridor-following strategies
- South wing (Lab, Voronoi): Room-hopping strategies
- Connecting areas (Perfect Maze): Exploration strategies

Enables study of spatially-varying equilibria
```

**3. Multi-Scale Planning**
```
Research Question: How do agents plan across multiple spatial scales?

Scenario: Large complex
- Macro-level: Which building/zone to enter?
- Micro-level: How to navigate within chosen zone?

Hybrid mazes naturally create hierarchical planning problems
```

**4. Realistic Evacuation Modeling**
```
Research Question: How do heterogeneous spaces affect evacuation dynamics?

Scenario: Conference center
- Large halls (Voronoi): Gathering areas during evacuation
- Narrow exits (Recursive Division): Bottlenecks
- Service corridors (Perfect Maze): Alternative routes

Captures realistic building evacuation complexity
```

---

## 🏗️ **Hybrid Combination Strategies**

### **Strategy 1: Spatial Partitioning** ⭐⭐⭐
**Concept**: Divide grid into spatial regions, each using a different algorithm

**Advantages**:
- Simplest to implement
- Clear zone boundaries
- Easy to visualize and understand

**Applications**:
- Museum: Left half = Voronoi galleries, Right half = CA garden
- Building: North wing = offices, South wing = labs
- Campus: Different algorithms per building

**Implementation Complexity**: Low

### **Strategy 2: Hierarchical Generation** ⭐⭐⭐⭐⭐
**Concept**: Use one algorithm to define major zones, then different algorithms within each zone

**Advantages**:
- Most powerful and flexible
- Natural multi-scale structure
- Realistic building layouts

**Applications**:
- Level 1: Voronoi defines 8 major zones
- Level 2: Zones 1-3 use Recursive Division (offices)
- Level 2: Zones 4-6 use Perfect Maze (corridors)
- Level 2: Zones 7-8 use Cellular Automata (gardens)

**Implementation Complexity**: Medium-High

### **Strategy 3: Checkerboard Pattern** ⭐⭐
**Concept**: Alternating algorithm selection in a grid pattern

**Advantages**:
- Fine-grained mixing
- High spatial diversity
- Interesting visual patterns

**Applications**:
- Alternating room types in large facilities
- Mixed-use spaces

**Implementation Complexity**: Low

### **Strategy 4: Radial/Concentric** ⭐⭐⭐
**Concept**: Different algorithms based on distance from center

**Advantages**:
- Natural center-periphery distinction
- Models buildings with central atriums
- Smooth transition zones

**Applications**:
- Center: Large Voronoi atrium
- Middle ring: Recursive Division offices
- Outer ring: Cellular Automata gardens

**Implementation Complexity**: Medium

### **Strategy 5: Feature Blending** ⭐⭐⭐⭐
**Concept**: Smooth interpolation between two maze types using gradient masks

**Advantages**:
- Smoothest transitions
- Most realistic boundaries
- Sophisticated appearance

**Applications**:
- Indoor-outdoor transitions
- Gradual change from structured to organic

**Implementation Complexity**: High (requires post-processing)

---

## 📊 **Recommended Hybrid Combinations**

### **Tier 1: High Impact** (Implement First)

| Combination | Strategy | MFG Value | Use Case |
|-------------|----------|-----------|----------|
| **Voronoi + Cellular Automata** | Spatial | ⭐⭐⭐⭐⭐ | Museum galleries + gardens |
| **Recursive Division + Perfect Maze** | Spatial | ⭐⭐⭐⭐⭐ | Offices + service corridors |
| **Hierarchical (Voronoi → Multiple)** | Hierarchical | ⭐⭐⭐⭐⭐ | Multi-zone complex |

### **Tier 2: Specialized**

| Combination | Strategy | MFG Value | Use Case |
|-------------|----------|-----------|----------|
| **Voronoi + Recursive Division** | Spatial | ⭐⭐⭐⭐ | Conference halls + meeting rooms |
| **Radial (Center: Voronoi)** | Radial | ⭐⭐⭐⭐ | Atrium-centered building |
| **CA + Perfect Maze** | Spatial | ⭐⭐⭐ | Natural caverns + tunnels |

### **Tier 3: Advanced**

| Combination | Strategy | MFG Value | Use Case |
|-------------|----------|-----------|----------|
| **Blended (Voronoi ↔ CA)** | Blending | ⭐⭐⭐⭐ | Indoor-outdoor transition |
| **Checkerboard** | Checkerboard | ⭐⭐ | Mixed-use facility |

---

## 🔧 **Implementation Design**

### **Core Architecture**

```python
from dataclasses import dataclass
from enum import Enum
from typing import Literal

class HybridStrategy(Enum):
    """Strategy for combining maze algorithms."""
    SPATIAL_SPLIT = "spatial_split"        # Divide grid into regions
    HIERARCHICAL = "hierarchical"           # Zones within zones
    CHECKERBOARD = "checkerboard"          # Alternating pattern
    RADIAL = "radial"                      # Center vs periphery
    BLENDING = "blending"                  # Smooth interpolation

@dataclass
class AlgorithmSpec:
    """Specification for one algorithm in hybrid."""
    algorithm: Literal["perfect", "recursive_division", "cellular_automata", "voronoi"]
    config: dict  # Algorithm-specific configuration
    region: str | None = None  # Region identifier (for hierarchical)

@dataclass
class HybridMazeConfig:
    """Configuration for hybrid maze generation."""
    rows: int
    cols: int
    strategy: HybridStrategy
    algorithms: list[AlgorithmSpec]

    # Strategy-specific parameters
    blend_ratio: float = 0.5       # For SPATIAL_SPLIT
    split_axis: Literal["horizontal", "vertical", "both"] = "vertical"
    num_zones: int = 4             # For HIERARCHICAL
    radial_center: tuple[int, int] | None = None  # For RADIAL

    seed: int | None = None
    ensure_connectivity: bool = True  # Connect all regions

class HybridMazeGenerator:
    """
    Generate hybrid mazes combining multiple algorithms.

    Enables realistic, heterogeneous MFG environments.
    """

    def __init__(self, config: HybridMazeConfig):
        self.config = config
        self.maze: np.ndarray | None = None
        self.zone_map: np.ndarray | None = None  # Which zone each cell belongs to

    def generate(self, seed: int | None = None) -> np.ndarray:
        """Generate hybrid maze."""
        if self.config.strategy == HybridStrategy.SPATIAL_SPLIT:
            return self._spatial_split()
        elif self.config.strategy == HybridStrategy.HIERARCHICAL:
            return self._hierarchical()
        elif self.config.strategy == HybridStrategy.CHECKERBOARD:
            return self._checkerboard()
        elif self.config.strategy == HybridStrategy.RADIAL:
            return self._radial()
        elif self.config.strategy == HybridStrategy.BLENDING:
            return self._blending()

    def _spatial_split(self) -> np.ndarray:
        """Divide grid spatially and apply different algorithms."""
        # Implementation details...

    def _hierarchical(self) -> np.ndarray:
        """Use one algorithm for zones, others within zones."""
        # 1. Generate zone structure (e.g., Voronoi with large rooms)
        # 2. For each zone, apply specified algorithm
        # 3. Connect zones at boundaries

    def _ensure_inter_zone_connectivity(self):
        """Add doors between zones to ensure global connectivity."""
        # Find zone boundaries
        # Add connecting passages
```

### **Example Usage**

```python
# Example 1: Museum (Spatial Split)
museum_config = HybridMazeConfig(
    rows=80, cols=100,
    strategy=HybridStrategy.SPATIAL_SPLIT,
    algorithms=[
        AlgorithmSpec("voronoi", {"num_points": 12, "relaxation_iterations": 2}),
        AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.42, "num_iterations": 6})
    ],
    blend_ratio=0.6,  # 60% Voronoi (galleries), 40% CA (garden)
    split_axis="vertical",
    seed=42
)

generator = HybridMazeGenerator(museum_config)
maze = generator.generate()
zone_map = generator.zone_map  # Which algorithm generated each cell

# Example 2: Multi-Zone Campus (Hierarchical)
campus_config = HybridMazeConfig(
    rows=120, cols=120,
    strategy=HybridStrategy.HIERARCHICAL,
    algorithms=[
        AlgorithmSpec("voronoi", {"num_points": 9}, region="zones"),  # Level 1: 9 major zones
        AlgorithmSpec("recursive_division", {"min_room_width": 5}, region="zone_0,zone_1,zone_2"),  # Offices
        AlgorithmSpec("perfect", {}, region="zone_3,zone_4"),  # Labs with corridors
        AlgorithmSpec("cellular_automata", {"num_iterations": 5}, region="zone_5,zone_6"),  # Gardens
        AlgorithmSpec("voronoi", {"num_points": 8}, region="zone_7,zone_8")  # Conference areas
    ],
    num_zones=9,
    seed=42
)

# Example 3: Atrium Building (Radial)
atrium_config = HybridMazeConfig(
    rows=80, cols=80,
    strategy=HybridStrategy.RADIAL,
    algorithms=[
        AlgorithmSpec("voronoi", {"num_points": 1}),  # Center: Large atrium
        AlgorithmSpec("recursive_division", {"min_room_width": 4}),  # Middle: Offices
        AlgorithmSpec("cellular_automata", {}),  # Outer: Gardens
    ],
    radial_center=(40, 40),
    seed=42
)
```

---

## 🧪 **Testing Strategy**

**Unit Tests**:
- Configuration validation
- Connectivity verification (all zones reachable)
- Zone boundary detection
- Algorithm application correctness

**Integration Tests**:
- Complete hybrid generation pipeline
- Inter-zone door placement
- Various strategy combinations

**Visual Tests**:
- Generate example hybrids
- Verify clear zone boundaries
- Check aesthetic quality

**MFG Tests**:
- Agent can navigate entire maze
- Path length reasonable across zones
- No isolated regions

---

## 📈 **Expected Impact**

### **Research Publications**
- **First** hybrid MFG environment framework in literature
- Enables new class of research questions
- More realistic than current single-algorithm approaches

### **Applications**
- Building evacuation with realistic layouts
- Multi-zone crowd management
- Campus navigation
- Complex facility planning

### **Benchmark Value**
- Provides challenging test environments
- Heterogeneous structure tests algorithm robustness
- More realistic than uniform mazes

---

## 🗓️ **Implementation Roadmap**

### **Phase 1: Foundation** (Week 1)
- [ ] Create `HybridMazeConfig` and `AlgorithmSpec` dataclasses
- [ ] Implement `HybridMazeGenerator` base class
- [ ] Implement SPATIAL_SPLIT strategy (easiest)
- [ ] Add connectivity verification
- [ ] Unit tests

### **Phase 2: Advanced Strategies** (Week 2)
- [ ] Implement HIERARCHICAL strategy
- [ ] Implement RADIAL strategy
- [ ] Implement CHECKERBOARD strategy
- [ ] Integration tests

### **Phase 3: Refinement** (Week 3)
- [ ] Implement BLENDING strategy (optional, advanced)
- [ ] Add inter-zone door optimization
- [ ] Create comprehensive examples
- [ ] Visualization demos

### **Phase 4: Documentation** (Week 4)
- [ ] API documentation
- [ ] Tutorial: "Creating Realistic MFG Environments"
- [ ] Example gallery
- [ ] Benchmark hybrid mazes

---

## 📚 **References**

### **Maze Generation**
- Jamis Buck, "Mazes for Programmers" (2015)
- Recursive division, cellular automata, Voronoi diagrams

### **MFG Applications**
- Building evacuation modeling
- Multi-zone crowd dynamics
- Heterogeneous spatial structures

### **Spatial Algorithms**
- Voronoi partitioning (Fortune's algorithm)
- Cellular automata (Conway's Life, smoothing rules)
- Spanning trees (Kruskal's, Prim's)

---

## ✅ **Success Criteria**

1. **Functionality**: Generate valid hybrid mazes with all strategies
2. **Connectivity**: 100% global connectivity across zones
3. **Flexibility**: Support arbitrary algorithm combinations
4. **Quality**: Visually distinct zones with clear transitions
5. **Performance**: Generate 100×100 hybrid maze in <5 seconds
6. **Documentation**: Complete API docs + tutorial
7. **Tests**: >90% code coverage with hybrid-specific tests

---

**Status**: Ready for implementation
**Priority**: High (novel research contribution)
**Estimated Effort**: 3-4 weeks for complete implementation
