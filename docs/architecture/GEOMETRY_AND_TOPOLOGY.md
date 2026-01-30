# [SPEC] MFG_PDE Geometry & Topology System Architecture

**Document ID**: MFG-SPEC-GEO-1.0
**Status**: **FINAL RELEASE**
**Date**: 2026-01-30
**Target**: Core Architecture Team & Lead Developers

---

## 1. Executive Summary: The "Zero-Cost" & "Complete" Philosophy

This specification defines the architecture for the Geometry and Topology module of the `MFG_PDE` library. Addressing the eternal conflict between **"High-Level Abstraction"** and **"Low-Level Performance"** in scientific computing, this architecture establishes the following core principles:

1.  **Trait-Based Static Polymorphism**: Abandoning traditional runtime virtual function inheritance in favor of compile-time Trait composition. This allows the compiler to generate highly optimized kernel code for specific geometric characteristics, achieving **Zero-Cost Abstraction**.
2.  **Atomicity & Orthogonality**: Deconstructing all physical domains into 8 orthogonal atomic properties (Atomic Traits).
3.  **The 8+1 Strategy**: Covering 98% of mainstream scenarios via 8 core Traits, and handling edge cases (e.g., IGA, mixed dimensions) via 1 **Auxiliary Configuration** interface, ensuring architectural closure and completeness.
4.  **Recursive Composition**: Handling system-level requirements like multiphysics and overset grids via Aggregate Domains.

---

## 2. The Type System: 8 Atomic Traits (The Core)

A `Domain` in `MFG_PDE` is not a class hierarchy but a **Signature**. This signature is uniquely determined by the following 8 compile-time characteristics:

### 2.1 Connectivity (Topology)
Defines how neighbor relationships are determined, directly dictating the memory access pattern of computational kernels.
* **`ImplicitConnectivity`**: Neighbors are calculated via arithmetic rules (e.g., $idx \pm stride$).
    * *Advantage*: Zero memory overhead, extremely favorable for SIMD/Vectorization.
* **`ExplicitConnectivity`**: Neighbors are stored in memory (e.g., CSR, Adjacency List).
    * *Cost*: Memory bandwidth bound, suitable for arbitrary topology.
* **`DynamicConnectivity`**: Neighbors are calculated at runtime via spatial search (e.g., Radius Search).
    * *Cost*: High compute cost, suitable for Meshfree/SPH methods.

### 2.2 Structure (Regularity)
Defines the regularity of the graph, influencing indexing strategies.
* **`Structured`**: Nodes form a regular lattice, implying logical coordinates $(i, j, k)$.
* **`Unstructured`**: Arbitrary graph topology, no concept of logical coordinates.

### 2.3 Embedding (Geometry)
Defines the mapping from logical nodes to physical space $\mathbb{R}^d$.
* **`GridEmbedding`**: Coordinate is a function of the index $x = F(i)$ (e.g., $x_i = i \cdot dx$).
* **`FreeEmbedding`**: Coordinates are explicitly stored as a field $x \in \text{Array}$.
* **`AbstractEmbedding`**: No concept of space (Pure Graph).

### 2.4 Metric (Measure)
Defines the manifold structure and the form of differential operators.
* **`EuclideanMetric`**: Flat space, metric tensor is $\delta_{ij}$.
* **`ManifoldMetric`**: Curved space, defined by metric tensor $g_{ij}(x)$ (e.g., PDE on a Sphere).
* **`GraphMetric`**: Distance defined by edge weights or hop counts.

### 2.5 Boundary (Definition)
Defines the closure of the computational domain.
* **`BoxBoundary`**: Hyper-rectangular bounds (AABB).
* **`MeshBoundary`**: Explicit collection of boundary elements (Facets).
* **`ImplicitBoundary`**: Defined by a Level-Set function $\phi(x)=0$.
* **`NoBoundary`**: Periodic or infinite domain.

### 2.6 Temporality (Evolution)
Guides memory management and solver update strategies.
* **`Static`**: Geometry and topology are immutable.
* **`GeometricDynamic`**: Coordinates $x(t)$ change over time, topology is fixed (e.g., ALE, Lagrangian).
* **`TopologicalDynamic`**: Connectivity changes over time (e.g., AMR, Remeshing).

### 2.7 Data Layout (Memory)
The key switch for low-level performance.
* **`AoS` (Array of Structs)**: `[(x,y,u), ...]`. Suitable for particle interaction logic.
* **`SoA` (Struct of Arrays)**: `([x...], [y...], [u...])`. Mandatory for GPU and CPU SIMD.

### 2.8 Distribution (Parallelism)
Defines the parallel computation state.
* **`Local`**: Single-process view.
* **`Distributed`**: Multi-process view, implying the existence of Halo/Ghost regions.

---

## 3. The "+1" Extension: Auxiliary Configuration

To handle the 2% of edge cases not captured by the 8 Traits above, the **`AuxiliaryConfig`** protocol is introduced.

**Definition**: An optional metadata container used to carry extra geometric information required by specific solvers without altering the compilation path of the core Traits.

**Typical Use Cases**:
* **Isogeometric Analysis (IGA)**: Needs to store `KnotVectors` and `ControlPointWeights`.
* **High-Order FEM**: Needs to store `PolynomialDegree` or `BasisFunctionType`.
* **Mixed Dimensions**: Needs to explicitly distinguish `TopologicalDim` (e.g., 2D Sheet) and `EmbeddingDim` (e.g., 3D Space).

---

## 4. Generative Grammar: The Periodic Table

By orthogonally combining the 8 Traits, we can generate all physical domains required for scientific computing.

| Target Class | Connect. | Structure | Embed. | Metric | Temporal. | Bound. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cartesian Grid** | `Implicit` | `Structured` | `Grid` | `Euclidean` | `Static` | `Box` |
| **Curvilinear Grid** | `Implicit` | `Structured` | `Grid` | `Manifold` | `Static` | `Box` |
| **Unstructured FEM** | `Explicit` | `Unstructured` | `Free` | `Euclidean` | `Static` | `Mesh` |
| **Lagrangian Mesh** | `Explicit` | `Unstructured` | `Free` | `Euclidean` | **`GeoDyn`** | `Mesh` |
| **SPH Particles** | **`Dynamic`** | `Unstructured` | `Free` | `Euclidean` | **`GeoDyn`** | `None` |
| **AMR Grid** | `Explicit`[^1] | `Unstructured` | `Grid` | `Euclidean` | **`TopoDyn`**| `Box` |
| **Graph Network** | `Explicit` | `Unstructured` | **`Abstract`**| `Graph` | `Static` | `N/A` |

---

## 5. Constraint Logic & Validation

The `DomainSpec` must enforce physical constraints before object creation to prevent illegal combinations.

**Core Constraint Rules**:

| Rule ID | Conflicting Traits | Explanation |
| :--- | :--- | :--- |
| **C-01** | `Implicit` + `Unstructured` | Implicit index arithmetic relies on structural regularity. |
| **C-02** | `Implicit` + `TopologicalDynamic` | Adding/removing nodes breaks the fixed stride arithmetic. |
| **C-03** | `GridEmbedding` + `MovingNodes` | `Grid` embedding implies coordinates are locked to indices; moving nodes must be `FreeEmbedding`. |
| **C-04** | `AbstractEmbedding` + `EuclideanMetric` | Abstract graphs lack coordinates, making Euclidean distance impossible. |
| **C-05** | `ImplicitBoundary` + `NoMetric` | Implicit boundaries usually rely on a Signed Distance Function (SDF), requiring a metric space[^2]. |

---

## 6. System Level: Recursive Composition

To handle complex systems (Blind Spots), **Aggregate Domains** are introduced. These are containers, not new atoms.

### 6.1 CompositeDomain (Overset/Chimera)
Used for systems with multiple overlapping grids.
* **Structure**: Contains `List[Domain]`.
* **Key Component**: **`InterpolationMatrix`**. Responsible for transferring data between the boundary of Domain A and the interior of Domain B.
* **Application**: Helicopter rotor (rotating grid) + Fuselage (background grid).

### 6.2 MultiphysicsDomain (Coupling)
Used for coupling between different physical manifolds.
* **Structure**: Contains `Dict[PhysicsID, Domain]`.
* **Key Component**: **`InterfaceCoupler`**. Responsible for dimensionality reduction mapping (e.g., 3D Fluid -> 2D Structure Surface) and conservative flux transfer.
* **Application**: Fluid-Structure Interaction (FSI), Conjugate Heat Transfer (CHT).

---

## 7. Implementation Specifications

This section defines the standard patterns for code implementation.

### 7.1 The Domain Specification (Spec)
```python
@dataclasses.dataclass(frozen=True)
class DomainSpec:
    # --- 8 Core Traits ---
    connectivity: Type[ConnectivityTrait]
    structure: Type[StructureTrait]
    embedding: Type[EmbeddingTrait]
    metric: Type[MetricTrait]
    boundary: Type[BoundaryTrait]
    temporality: Type[TemporalityTrait]
    layout: Type[DataLayoutTrait]
    distribution: Type[DistributionTrait]
    
    # --- The +1 Aux Config ---
    auxiliary_config: Optional[object] = None

    def __post_init__(self):
        # Implementation of Constraint Logic (Section 5)
        validate_spec(self)
```



### 7.2 The Kernel Factory (Strategy Pattern)

```python
def create_domain(spec: DomainSpec, data):
    # 1. Trait-based Dispatch
    # Select the optimal computation kernel based on Connectivity & Layout
    if issubclass(spec.connectivity, Implicit):
        if issubclass(spec.layout, SoA):
            kernel = ImplicitSoAKernel() # Highly vectorized
        else:
            kernel = ImplicitAoSKernel()
    elif issubclass(spec.connectivity, Explicit):
        kernel = ExplicitCSRKernel() # Memory optimized
        
    # 2. Composition
    return Domain(spec, kernel, data)

```



## 8. Interaction Protocols

Defines how a generic Solver interacts with specific geometric features.

### 8.1 Protocol: Implicit Geometry (`DiscretizationPolicy`)

Resolves the integration problem between `ImplicitDomain` (SDF) and the background grid.

* **Principle**: Separation of Data and Algorithm.
* **Components**:
* `ImplicitDomain`: Holds only geometric data ( values).
* `DiscretizationPolicy`: Stateless operator (e.g., `MarchingCubesPolicy`, `CutCellPolicy`).


* **Call**: `weights = policy.compute_weights(cell_id, grid, implicit_domain)`

### 8.2 Protocol: Parallelism (`CommunicationPlan`)

Resolves the Halo Exchange problem in distributed computing[^1].

* **Principle**: Pre-computed communication graph, zero runtime search.
* **Components**:
* `CommunicationPlan`: Contains `send_map` (Rank -> Indices) and `recv_map`.


* **Call**: `domain.update_halos(field_data)`. Under the hood, this automatically calls MPI non-blocking communication.

---

## 9. Conclusion

This architecture specification (v1.0), via **8 Core Atomic Traits**, **1 Auxiliary Extension**, and **Recursive Composition Mechanisms**, constructs a Geometry & Topology system that is both rigorous and flexible. It successfully addresses the engineering challenges of Performance (Zero-Cost Abstraction), Generality (Full Spectrum Coverage), and Completeness (System-Level Composition).

---

## References


[^1]: Gropp, W., Lusk, E., & Skjellum, A. (1999). *Using MPI: Portable Parallel Programming with the Message-passing Interface*. MIT press.