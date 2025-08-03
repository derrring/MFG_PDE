#!/bin/bash
# MFG_PDE Environment Variables Setup
# Source this file after activating your conda environment

echo "🔧 Setting up MFG_PDE environment variables..."

# NumPy/SciPy Performance Optimizations
export OPENBLAS_NUM_THREADS=4          # Adjust based on your CPU cores
export MKL_NUM_THREADS=4               # If using Intel MKL
export NUMBA_NUM_THREADS=4             # Numba parallel threads
export OMP_NUM_THREADS=4               # OpenMP threads

# JAX Configuration (if using JAX acceleration)
export JAX_PLATFORM_NAME=cpu          # Use 'gpu' if you have CUDA
export JAX_ENABLE_X64=true             # Enable 64-bit precision

# Python Optimizations
export PYTHONOPTIMIZE=1               # Enable Python optimizations
export PYTHONHASHSEED=random          # Randomize hash seed for security

# Memory Management
export MALLOC_TRIM_THRESHOLD_=100000  # Trim memory more aggressively

# Matplotlib Backend (for headless servers)
export MPLBACKEND=Agg                 # Use for non-interactive plotting

# Development Settings
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Add current directory to Python path

# MFG_PDE Specific Settings
export MFG_PDE_CACHE_DIR="$HOME/.cache/mfg_pde"     # Cache directory
export MFG_PDE_LOG_LEVEL=INFO                        # Logging level
export MFG_PDE_USE_PARALLEL=true                     # Enable parallelization

# Create cache directory if it doesn't exist
mkdir -p "$MFG_PDE_CACHE_DIR"

echo "✅ Environment variables configured for optimal MFG_PDE performance"
echo "📊 Using $OPENBLAS_NUM_THREADS threads for linear algebra operations"

# Verify NumPy configuration
python -c "
import numpy as np
print('NumPy configuration:')
np.show_config()
print(f'NumPy using {np.__config__.get_info(\"openblas_info\", {})} for BLAS')
"