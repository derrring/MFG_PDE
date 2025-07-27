"""
JAX utility functions for MFG_PDE accelerated solvers.

This module provides common utilities for JAX-based implementations,
including numerical operations, device management, and optimization helpers.
"""

import warnings
from typing import Tuple, Optional, Any, Callable, Union
import numpy as np

from . import HAS_JAX, HAS_GPU, DEFAULT_DEVICE

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad, jacfwd, jacrev, device_put
    from jax.lax import scan, cond
    import optax
else:
    # Dummy implementations for graceful fallback
    jax = None
    jnp = np
    device_put = lambda x, device=None: x


def ensure_jax_available():
    """Ensure JAX is available, raise error if not."""
    if not HAS_JAX:
        raise ImportError(
            "JAX is required for accelerated solvers. "
            "Install with: pip install jax jaxlib"
        )


def to_device(array: Union[np.ndarray, Any], device: Optional[Any] = None) -> Any:
    """Move array to specified device."""
    ensure_jax_available()
    
    if device is None:
        device = DEFAULT_DEVICE
    
    return device_put(array, device)


def from_device(array: Any) -> np.ndarray:
    """Move array from device to numpy."""
    ensure_jax_available()
    
    if hasattr(array, '__array__'):
        return np.asarray(array)
    return array


@jit
def finite_difference_1d(u: jnp.ndarray, dx: float, order: int = 2) -> jnp.ndarray:
    """
    Compute first derivative using finite differences.
    
    Args:
        u: 1D array to differentiate
        dx: Grid spacing
        order: Order of accuracy (2, 4, or 6)
    
    Returns:
        First derivative array
    """
    if order == 2:
        # Second-order central difference
        u_padded = jnp.pad(u, 1, mode='edge')
        return (u_padded[2:] - u_padded[:-2]) / (2 * dx)
    
    elif order == 4:
        # Fourth-order central difference
        u_padded = jnp.pad(u, 2, mode='edge')
        return (-u_padded[4:] + 8*u_padded[3:-1] - 8*u_padded[1:-3] + u_padded[:-4]) / (12 * dx)
    
    elif order == 6:
        # Sixth-order central difference
        u_padded = jnp.pad(u, 3, mode='edge')
        return (u_padded[6:] - 9*u_padded[5:-1] + 45*u_padded[4:-2] - 
                45*u_padded[2:-4] + 9*u_padded[1:-5] - u_padded[:-6]) / (60 * dx)
    
    else:
        raise ValueError(f"Unsupported order: {order}")


@jit
def finite_difference_2d(u: jnp.ndarray, dx: float, order: int = 2) -> jnp.ndarray:
    """
    Compute second derivative using finite differences.
    
    Args:
        u: 1D array to differentiate
        dx: Grid spacing
        order: Order of accuracy (2, 4, or 6)
    
    Returns:
        Second derivative array
    """
    if order == 2:
        # Second-order central difference for second derivative
        u_padded = jnp.pad(u, 1, mode='edge')
        return (u_padded[2:] - 2*u_padded[1:-1] + u_padded[:-2]) / (dx**2)
    
    elif order == 4:
        # Fourth-order central difference for second derivative
        u_padded = jnp.pad(u, 2, mode='edge')
        return (-u_padded[4:] + 16*u_padded[3:-1] - 30*u_padded[2:-2] + 
                16*u_padded[1:-3] - u_padded[:-4]) / (12 * dx**2)
    
    elif order == 6:
        # Sixth-order central difference for second derivative
        u_padded = jnp.pad(u, 3, mode='edge')
        return (2*u_padded[6:] - 27*u_padded[5:-1] + 270*u_padded[4:-2] - 
                490*u_padded[3:-3] + 270*u_padded[2:-4] - 27*u_padded[1:-5] + 
                2*u_padded[:-6]) / (180 * dx**2)
    
    else:
        raise ValueError(f"Unsupported order: {order}")


@jit
def apply_boundary_conditions(u: jnp.ndarray, bc_type: str = 'neumann', 
                            bc_value: float = 0.0) -> jnp.ndarray:
    """
    Apply boundary conditions to array.
    
    Args:
        u: Array to apply boundary conditions to
        bc_type: Type of boundary condition ('neumann', 'dirichlet', 'periodic')
        bc_value: Boundary condition value
    
    Returns:
        Array with boundary conditions applied
    """
    if bc_type == 'dirichlet':
        # Set boundary values
        u = u.at[0].set(bc_value)
        u = u.at[-1].set(bc_value)
    
    elif bc_type == 'neumann':
        # Zero derivative at boundaries (copy interior values)
        u = u.at[0].set(u[1])
        u = u.at[-1].set(u[-2])
    
    elif bc_type == 'periodic':
        # Periodic boundary conditions
        u = u.at[0].set(u[-2])
        u = u.at[-1].set(u[1])
    
    return u


@jit
def tridiagonal_solve(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, 
                     d: jnp.ndarray) -> jnp.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm.
    
    Args:
        a: Lower diagonal (size n-1)
        b: Main diagonal (size n)
        c: Upper diagonal (size n-1)
        d: Right-hand side (size n)
    
    Returns:
        Solution vector
    """
    n = len(b)
    
    # Pad arrays to same size for vectorization
    a_pad = jnp.concatenate([jnp.array([0.0]), a])
    c_pad = jnp.concatenate([c, jnp.array([0.0])])
    
    # Forward elimination
    def forward_step(carry, i):
        c_mod, d_mod = carry
        w = a_pad[i] / b[i-1] if i > 0 else 0.0
        c_new = c_mod.at[i].set(c_pad[i] / (b[i] - w * c_mod[i-1]) if i > 0 else c_pad[i] / b[i])
        d_new = d_mod.at[i].set((d[i] - w * d_mod[i-1]) / (b[i] - w * c_mod[i-1]) if i > 0 else d[i] / b[i])
        return (c_new, d_new), None
    
    c_mod = jnp.zeros_like(c_pad)
    d_mod = jnp.zeros_like(d)
    (c_mod, d_mod), _ = scan(forward_step, (c_mod, d_mod), jnp.arange(n))
    
    # Back substitution
    def backward_step(carry, i):
        x = carry
        i_actual = n - 1 - i
        x_new = d_mod[i_actual] - c_mod[i_actual] * x[i_actual + 1] if i_actual < n - 1 else d_mod[i_actual]
        x = x.at[i_actual].set(x_new)
        return x, None
    
    x = jnp.zeros(n)
    x, _ = scan(backward_step, x, jnp.arange(n))
    
    return x


@jit
def compute_hamiltonian(u_x: jnp.ndarray, m: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Compute Hamiltonian for HJB equation.
    
    Args:
        u_x: Spatial derivative of value function
        m: Density function
        sigma: Diffusion coefficient
    
    Returns:
        Hamiltonian value
    """
    # H(x, p, m) = (1/2) * p^2 + interaction_term(x, m)
    kinetic_term = 0.5 * u_x**2
    
    # Simple interaction term (can be customized)
    interaction_term = jnp.log(m + 1e-8)  # Logarithmic interaction
    
    return kinetic_term + interaction_term


@jit
def compute_optimal_control(u_x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute optimal control from value function gradient.
    
    Args:
        u_x: Spatial derivative of value function
    
    Returns:
        Optimal control
    """
    return -u_x  # For quadratic cost


@jit
def compute_drift(u_x: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Compute drift term for Fokker-Planck equation.
    
    Args:
        u_x: Spatial derivative of value function
        sigma: Diffusion coefficient
    
    Returns:
        Drift term
    """
    return compute_optimal_control(u_x)


@jit
def mass_conservation_constraint(m: jnp.ndarray, dx: float) -> float:
    """
    Compute mass conservation constraint.
    
    Args:
        m: Density function
        dx: Grid spacing
    
    Returns:
        Mass conservation error
    """
    total_mass = jnp.sum(m) * dx
    return jnp.abs(total_mass - 1.0)


def create_optimization_schedule(learning_rate: float = 1e-3, 
                               decay_steps: int = 1000,
                               decay_rate: float = 0.9) -> Any:
    """
    Create learning rate schedule for optimization.
    
    Args:
        learning_rate: Initial learning rate
        decay_steps: Steps between decay
        decay_rate: Decay rate
    
    Returns:
        Optax scheduler
    """
    ensure_jax_available()
    
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    
    return optax.adam(schedule)


@jit
def compute_convergence_error(u_new: jnp.ndarray, u_old: jnp.ndarray,
                            m_new: jnp.ndarray, m_old: jnp.ndarray) -> float:
    """
    Compute convergence error for MFG system.
    
    Args:
        u_new: New value function
        u_old: Old value function  
        m_new: New density function
        m_old: Old density function
    
    Returns:
        Convergence error
    """
    u_error = jnp.linalg.norm(u_new - u_old) / (jnp.linalg.norm(u_old) + 1e-8)
    m_error = jnp.linalg.norm(m_new - m_old) / (jnp.linalg.norm(m_old) + 1e-8)
    
    return u_error + m_error


@jit
def adaptive_time_step(error: float, dt: float, tolerance: float = 1e-6,
                      safety_factor: float = 0.9, 
                      max_factor: float = 2.0,
                      min_factor: float = 0.1) -> float:
    """
    Compute adaptive time step based on error estimate.
    
    Args:
        error: Current error estimate
        dt: Current time step
        tolerance: Target tolerance
        safety_factor: Safety factor for step size adjustment
        max_factor: Maximum increase factor
        min_factor: Minimum decrease factor
    
    Returns:
        New time step
    """
    if error == 0:
        factor = max_factor
    else:
        factor = safety_factor * jnp.power(tolerance / error, 0.2)
    
    factor = jnp.clip(factor, min_factor, max_factor)
    return dt * factor


def vectorized_solve(solve_func: Callable, problems: list, 
                    batch_size: Optional[int] = None) -> list:
    """
    Solve multiple problems in vectorized batches.
    
    Args:
        solve_func: JAX-compiled solving function
        problems: List of problem configurations
        batch_size: Batch size for processing (None for auto)
    
    Returns:
        List of solutions
    """
    ensure_jax_available()
    
    if batch_size is None:
        # Auto-determine batch size based on available memory
        batch_size = min(len(problems), 32)  # Conservative default
    
    results = []
    
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i + batch_size]
        
        # Convert to JAX arrays if needed
        batch_arrays = []
        for problem in batch:
            if hasattr(problem, 'to_jax_arrays'):
                batch_arrays.append(problem.to_jax_arrays())
            else:
                batch_arrays.append(problem)
        
        # Vectorized solve
        batch_results = vmap(solve_func)(batch_arrays)
        
        # Convert back to numpy if needed
        batch_results_np = [from_device(result) for result in batch_results]
        results.extend(batch_results_np)
    
    return results


def profile_jax_function(func: Callable, *args, num_runs: int = 10) -> dict:
    """
    Profile JAX function performance.
    
    Args:
        func: JAX function to profile
        *args: Arguments to function
        num_runs: Number of runs for timing
    
    Returns:
        Performance metrics
    """
    ensure_jax_available()
    
    import time
    
    # Compile function first
    compiled_func = jit(func)
    
    # Warm-up run
    _ = compiled_func(*args)
    
    # Time compilation
    compile_start = time.time()
    compiled_func = jit(func)
    _ = compiled_func(*args)
    compile_time = time.time() - compile_start
    
    # Time execution
    exec_times = []
    for _ in range(num_runs):
        start = time.time()
        result = compiled_func(*args)
        
        if HAS_GPU:
            # Ensure GPU computation is complete
            result.block_until_ready()
        
        exec_times.append(time.time() - start)
    
    return {
        'compile_time': compile_time,
        'mean_exec_time': np.mean(exec_times),
        'std_exec_time': np.std(exec_times),
        'min_exec_time': np.min(exec_times),
        'max_exec_time': np.max(exec_times),
        'speedup_vs_numpy': None  # Could be computed with numpy baseline
    }


def memory_usage_tracker():
    """Create memory usage tracker for JAX computations."""
    ensure_jax_available()
    
    class MemoryTracker:
        def __init__(self):
            self.peak_memory = 0
            self.current_memory = 0
        
        def update(self):
            if HAS_GPU:
                try:
                    # Get GPU memory usage
                    for device in jax.devices('gpu'):
                        if hasattr(device, 'memory_stats'):
                            stats = device.memory_stats()
                            current = stats.get('bytes_in_use', 0) / 1024**3  # GB
                            self.current_memory = max(self.current_memory, current)
                            self.peak_memory = max(self.peak_memory, current)
                except:
                    pass
        
        def reset(self):
            self.peak_memory = 0
            self.current_memory = 0
    
    return MemoryTracker()


# Export utility functions
__all__ = [
    'ensure_jax_available',
    'to_device',
    'from_device',
    'finite_difference_1d',
    'finite_difference_2d',
    'apply_boundary_conditions',
    'tridiagonal_solve',
    'compute_hamiltonian',
    'compute_optimal_control',
    'compute_drift',
    'mass_conservation_constraint',
    'create_optimization_schedule',
    'compute_convergence_error',
    'adaptive_time_step',
    'vectorized_solve',
    'profile_jax_function',
    'memory_usage_tracker'
]