# Network Backend System Guide

The MFG_PDE package includes a unified network backend system that automatically selects the optimal network library based on problem size, operation type, and available libraries. This system provides seamless integration with multiple high-performance network libraries while maintaining a consistent API.

## Supported Backends

### 🎯 **Primary: igraph** (Recommended)
- **Best for**: Most network operations (1K-100K nodes)
- **Performance**: 10-100x faster than NetworkX
- **Memory**: Efficient C-based implementation  
- **Visualization**: Excellent built-in layouts
- **Installation**: `pip install igraph`

### 🚀 **Scale: networkit** (Large Networks)
- **Best for**: Large-scale networks (>100K nodes)
- **Performance**: Parallel algorithms, billions of nodes
- **Memory**: Optimized for massive networks
- **Algorithms**: Specialized for scale
- **Installation**: `pip install networkit`

### 🔧 **Complete: networkx** (Fallback)
- **Best for**: Algorithm completeness, small networks (<10K)
- **Performance**: Slower but comprehensive
- **Algorithms**: Most complete library
- **Compatibility**: Standard in Python ecosystem
- **Installation**: `pip install networkx`

## Automatic Backend Selection

The system automatically chooses the optimal backend based on:

```python
from mfg_pde import NetworkBackendType, get_backend_manager

manager = get_backend_manager()

# Automatic selection based on size
small_backend = manager.choose_backend(100)     # → igraph or networkx
medium_backend = manager.choose_backend(10000)  # → igraph
large_backend = manager.choose_backend(100000)  # → networkit or igraph
```

### Selection Criteria

| Network Size | Operation Type | Preferred Backend | Fallback |
|-------------|----------------|------------------|----------|
| < 1K nodes | General | igraph | networkx |
| 1K-100K | General | igraph | networkx |
| > 100K | General | networkit | igraph |
| Any | Visualization | igraph | networkx |
| Any | Algorithms | networkx | igraph |
| Any | Large-scale | networkit | igraph |

## Usage Examples

### Basic Usage with Automatic Selection

```python
from mfg_pde import create_network, NetworkType

# Automatic backend selection
grid = create_network(NetworkType.GRID, 2500, width=50, height=50)
network_data = grid.create_network()

print(f"Backend used: {network_data.backend_type.value}")
print(f"Performance: {network_data.num_nodes} nodes, {network_data.num_edges} edges")
```

### Explicit Backend Selection

```python
from mfg_pde import GridNetwork, NetworkBackendType

# Force specific backend
grid = GridNetwork(100, 100, backend_preference=NetworkBackendType.IGRAPH)
network_data = grid.create_network()

# Or force at creation time
network_data = grid.create_network(force_backend=NetworkBackendType.NETWORKIT)
```

### Global Backend Preference

```python
from mfg_pde import set_preferred_backend, NetworkBackendType

# Set global preference
set_preferred_backend(NetworkBackendType.IGRAPH)

# All subsequent network creations will prefer igraph
grid = create_network('grid', 10000)
```

## Performance Comparison

Based on typical use cases:

### Small Networks (< 1K nodes)
- **igraph**: ~1.5x faster than NetworkX
- **networkx**: Good performance, best algorithms
- **Choice**: Either igraph or networkx

### Medium Networks (1K-100K nodes)  
- **igraph**: 10-50x faster than NetworkX
- **networkit**: Similar to igraph, better for very large
- **Choice**: igraph for best balance

### Large Networks (> 100K nodes)
- **networkit**: 10-100x faster, parallel algorithms
- **igraph**: Still good, may hit memory limits
- **Choice**: networkit for optimal performance

## Network Types and Backends

### Grid Networks
```python
from mfg_pde import GridNetwork, NetworkBackendType

# Optimal for medium-sized grids
grid = GridNetwork(100, 100, backend_preference=NetworkBackendType.IGRAPH)
data = grid.create_network()
```

### Random Networks (Erdős–Rényi)
```python
from mfg_pde import RandomNetwork

# Automatic selection based on size
random_net = RandomNetwork(50000, connection_prob=0.01)
data = random_net.create_network(seed=42)
```

### Scale-Free Networks (Barabási-Albert)
```python
from mfg_pde import ScaleFreeNetwork

# Custom preferential attachment implementation
sf_net = ScaleFreeNetwork(10000, num_edges_per_node=3)
data = sf_net.create_network(seed=42)
```

## Backend Information and Diagnostics

### Check Available Backends
```python
from mfg_pde import get_backend_manager

manager = get_backend_manager()
info = manager.get_backend_info()

print("Available backends:", info['available_backends'])
for backend, caps in info['backend_capabilities'].items():
    print(f"{backend}: max {caps['max_recommended_nodes']:,} nodes")
```

### Backend Capabilities
```python
from mfg_pde import NetworkBackendType

manager = get_backend_manager()
caps = manager.get_capabilities(NetworkBackendType.IGRAPH)

print(f"Max nodes: {caps.max_recommended_nodes:,}")
print(f"Speed rating: {caps.speed_rating}/5")
print(f"Parallel support: {caps.parallel_support}")
```

## Installation Recommendations

### Minimal Installation
```bash
pip install mfg_pde
# Includes igraph by default
```

### Performance Installation
```bash
pip install "mfg_pde[networks]"
# Includes igraph, networkit, networkx
```

### Complete Installation
```bash  
pip install "mfg_pde[networks,performance]"
# All network backends + performance tools
```

## Migration from NetworkX-only Code

The backend system is fully backward compatible:

```python
# Old NetworkX-only code still works
from mfg_pde import create_network
grid = create_network('grid', 100)

# New code automatically gets performance benefits
# No changes needed - backend selection is automatic
```

## Advanced Backend Configuration

### Operation-Specific Selection
```python
from mfg_pde import OperationType

# Force specific operation type
backend = manager.choose_backend(
    num_nodes=50000,
    operation_type=OperationType.VISUALIZATION,
    force_backend=NetworkBackendType.IGRAPH
)
```

### Custom Backend Logic
```python
def choose_custom_backend(num_nodes: int) -> NetworkBackendType:
    """Custom backend selection logic."""
    if num_nodes < 5000:
        return NetworkBackendType.NETWORKX  # Algorithm-rich
    elif num_nodes < 50000:
        return NetworkBackendType.IGRAPH    # Balanced
    else:
        return NetworkBackendType.NETWORKIT # Scale
```

## Best Practices

1. **Let the system choose**: The automatic selection is optimized for most use cases
2. **Install igraph**: Essential for good performance on medium networks
3. **Add networkit**: Critical for networks > 100K nodes
4. **Keep networkx**: Provides algorithm completeness and fallback
5. **Profile your code**: Use the demo script to benchmark your specific use case

## Troubleshooting

### Backend Not Available
```python
from mfg_pde import BackendNotAvailableError

try:
    network.create_network(force_backend=NetworkBackendType.IGRAPH)
except BackendNotAvailableError:
    print("igraph not installed: pip install igraph")
```

### Performance Issues
- Small networks: Check if NetworkX is being used (expected)
- Medium networks: Ensure igraph is installed and available
- Large networks: Install networkit for optimal performance

### Import Warnings
The system will warn about missing backends:
```
ImportWarning: igraph not available. Install with: pip install igraph
igraph provides excellent performance for most network operations.
```

## Summary

The unified backend system provides:

- ✅ **Automatic optimization**: Best performance without code changes
- ✅ **Graceful fallbacks**: Works even with minimal installations  
- ✅ **Consistent API**: Same interface regardless of backend
- ✅ **Easy scaling**: Handles small to billion-node networks
- ✅ **Future-proof**: Easy to add new backends

This system ensures your MFG network computations always use the optimal backend for your specific problem size and requirements.