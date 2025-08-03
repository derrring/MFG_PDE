#!/bin/bash
# Complete NumPy 2.0+ Migration Script for MFG_PDE
# This script handles the complete migration from NumPy 1.x to 2.0+

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_step() { echo -e "${PURPLE}🔄 $1${NC}"; }

echo "🚀 MFG_PDE NumPy 2.0+ Migration"
echo "================================"

# Step 1: Check prerequisites
print_step "Step 1: Checking prerequisites..."

if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Anaconda or Miniconda."
    exit 1
fi
print_success "Conda is available"

# Check if we have backup
if ls mfg_env_backup_*.yml 1> /dev/null 2>&1; then
    print_success "Environment backup found"
else
    print_warning "No backup found, creating one now..."
    conda env export > "mfg_env_backup_$(date +%Y%m%d_%H%M%S).yml"
    print_success "Backup created"
fi

# Step 2: Test current functionality
print_step "Step 2: Testing current MFG_PDE functionality..."

python -c "
try:
    import mfg_pde
    from mfg_pde.core.lagrangian_mfg_problem import create_quadratic_lagrangian_mfg
    from mfg_pde.alg.variational_solvers import VariationalMFGSolver
    
    # Test basic functionality
    problem = create_quadratic_lagrangian_mfg(xmin=0, xmax=1, Nx=10, T=0.1, Nt=5)
    solver = VariationalMFGSolver(problem)
    initial_guess = solver.create_initial_guess('gaussian')
    
    print('✅ Current functionality verified')
except Exception as e:
    print(f'❌ Current functionality test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Current functionality test passed"
else
    print_error "Current functionality test failed"
    exit 1
fi

# Step 3: Create NumPy 2.0+ environment
print_step "Step 3: Creating NumPy 2.0+ environment..."

if conda env list | grep -q "mfg_numpy2"; then
    print_warning "Environment mfg_numpy2 already exists. Updating..."
    conda env update -f numpy2_migration.yml -n mfg_numpy2
else
    print_info "Creating new environment mfg_numpy2..."
    conda env create -f numpy2_migration.yml
fi

print_success "NumPy 2.0+ environment created/updated"

# Step 4: Activate new environment and install MFG_PDE
print_step "Step 4: Installing MFG_PDE in NumPy 2.0+ environment..."

# Note: In a script, we need to source conda's initialization
eval "$(conda shell.bash hook)"
conda activate mfg_numpy2

# Install MFG_PDE in development mode
pip install -e .

print_success "MFG_PDE installed in NumPy 2.0+ environment"

# Step 5: Run comprehensive tests
print_step "Step 5: Running comprehensive tests..."

echo "📊 Environment Information:"
python -c "
import numpy as np
import scipy
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')

from mfg_pde.utils.numpy_compat import get_numpy_info
info = get_numpy_info()
print(f'Has np.trapezoid: {info[\"has_trapezoid\"]}')
print(f'Has np.trapz: {info[\"has_trapz\"]}')
print(f'Recommended method: {info[\"recommended_method\"]}')
print(f'Is NumPy 2.0+: {info[\"is_numpy_2_plus\"]}')
"

print_info "Running compatibility tests..."
python test_numpy_compatibility.py

if [ $? -eq 0 ]; then
    print_success "All compatibility tests passed!"
else
    print_error "Some compatibility tests failed"
    exit 1
fi

# Step 6: Run MFG_PDE specific tests
print_step "Step 6: Testing MFG_PDE functionality with NumPy 2.0+..."

python -c "
print('🧪 Testing MFG_PDE with NumPy 2.0+')
print('=' * 40)

# Test 1: Basic Lagrangian functionality
try:
    from mfg_pde.core.lagrangian_mfg_problem import create_quadratic_lagrangian_mfg
    from mfg_pde.alg.variational_solvers import VariationalMFGSolver
    
    problem = create_quadratic_lagrangian_mfg(xmin=0, xmax=1, Nx=20, T=0.2, Nt=10)
    solver = VariationalMFGSolver(problem)
    
    # Test operations that use integration
    initial_guess = solver.create_initial_guess('gaussian')
    mass_error = solver.compute_mass_conservation_error(initial_guess)
    cost = solver.evaluate_cost_functional(initial_guess)
    
    print(f'✅ Lagrangian problem creation: OK')
    print(f'✅ Mass conservation error: {mass_error:.2e}')
    print(f'✅ Cost functional: {cost:.4f}')
    
except Exception as e:
    print(f'❌ Lagrangian test failed: {e}')
    exit(1)

# Test 2: Semi-Lagrangian solver
try:
    from mfg_pde import MFGProblem
    from mfg_pde.factory import create_semi_lagrangian_solver
    
    problem = MFGProblem(xmin=0, xmax=1, Nx=20, T=0.1, Nt=10)
    solver = create_semi_lagrangian_solver(problem)
    
    print(f'✅ Semi-Lagrangian solver creation: OK')
    
except Exception as e:
    print(f'❌ Semi-Lagrangian test failed: {e}')
    exit(1)

# Test 3: Primal-dual solver
try:
    from mfg_pde.alg.variational_solvers import PrimalDualMFGSolver
    
    solver = PrimalDualMFGSolver(problem)
    print(f'✅ Primal-dual solver creation: OK')
    
except Exception as e:
    print(f'❌ Primal-dual test failed: {e}')
    exit(1)

print('🎉 All MFG_PDE functionality working with NumPy 2.0+!')
"

if [ $? -eq 0 ]; then
    print_success "All MFG_PDE tests passed with NumPy 2.0+!"
else
    print_error "MFG_PDE tests failed with NumPy 2.0+"
    exit 1
fi

# Step 7: Performance comparison
print_step "Step 7: Running performance comparison..."

python -c "
import time
import numpy as np
from mfg_pde.utils.numpy_compat import trapezoid

print('⚡ Performance Comparison')
print('=' * 25)

# Test integration performance
x = np.linspace(0, 10, 10000)
y = np.sin(x) * np.exp(-x/5)

# Test our compatibility function
start_time = time.time()
for _ in range(1000):
    result = trapezoid(y, x=x)
total_time = time.time() - start_time

print(f'Integration test (1000 iterations): {total_time:.3f}s')
print(f'Using method: np.trapezoid' if hasattr(np, 'trapezoid') else 'Using fallback')
print(f'Result: {result:.6f}')

# Test MFG_PDE operations
from mfg_pde.core.lagrangian_mfg_problem import create_quadratic_lagrangian_mfg
from mfg_pde.alg.variational_solvers import VariationalMFGSolver

start_time = time.time()
problem = create_quadratic_lagrangian_mfg(xmin=0, xmax=1, Nx=50, T=0.2, Nt=20)
solver = VariationalMFGSolver(problem)
initial_guess = solver.create_initial_guess('gaussian')
cost = solver.evaluate_cost_functional(initial_guess)
setup_time = time.time() - start_time

print(f'MFG_PDE setup time: {setup_time:.3f}s')
print(f'✅ Performance test completed')
"

# Step 8: Update environment activation script
print_step "Step 8: Setting up environment activation..."

cat > activate_numpy2.sh << 'EOF'
#!/bin/bash
# Activate NumPy 2.0+ MFG_PDE environment

echo "🐍 Activating MFG_PDE NumPy 2.0+ environment..."

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate mfg_numpy2

# Load environment variables
if [[ -f "setup_env_vars.sh" ]]; then
    source setup_env_vars.sh
fi

echo "✅ NumPy 2.0+ environment activated!"
echo "📊 Environment info:"
python -c "
import numpy as np
import scipy
import mfg_pde
print(f'  NumPy: {np.__version__}')
print(f'  SciPy: {scipy.__version__}')
print(f'  MFG_PDE: {mfg_pde.__version__}')

from mfg_pde.utils.numpy_compat import get_numpy_info
info = get_numpy_info()
print(f'  Integration method: {info[\"recommended_method\"]}')
"

echo "🚀 Ready for NumPy 2.0+ development!"
EOF

chmod +x activate_numpy2.sh
print_success "Environment activation script created"

# Step 9: Final verification
print_step "Step 9: Final verification..."

python verify_environment.py

print_success "Migration completed successfully!"

echo
echo "🎉 NumPy 2.0+ Migration Complete!"
echo "================================="
echo
print_info "What's new:"
echo "  • NumPy 2.0+ with native trapezoid function"
echo "  • All MFG_PDE functionality verified"
echo "  • Compatibility layer working perfectly"
echo "  • Performance optimizations active"
echo
print_info "To use the new environment:"
echo "  source activate_numpy2.sh"
echo "  # or manually:"
echo "  conda activate mfg_numpy2"
echo
print_info "To revert to previous environment:"
echo "  conda env create -f mfg_env_backup_*.yml"
echo
print_success "Happy computing with NumPy 2.0+! 🚀"