from dataclasses import dataclass
from typing import Optional, Callable, Tuple


''' Struct like class for boundary conditions
For most applications, FEniCS and FiPy are particularly user-friendly while still being powerful enough for research-level problems. 
'''

@dataclass
class BoundaryConditions:
    """Class for boundary condition configuration
    Note the dimension of the matrix operator for the boundary conditions
    periodic: M * M
    dirichlet: M-1 * M-1  
    neumann: M+1 * M+1
    no_flux: M * M (special case of Neumann for FP equations)
    robin: M+1 * M+1
    """

    type: str  # 'periodic', 'dirichlet', 'neumann', 'no_flux', or 'robin'
    # For Dirichlet: value of u
    # For Neumann: value of du/dn 
    # For no_flux: F(boundary) = 0 where F = v*m - D*dm/dx
    left_value: Optional[float] = None
    right_value: Optional[float] = None
    # For Robin: value of \gamma in \alpha* u + \beta* du/dn= g

    # Additional parameters for Robin boundary conditions
    left_alpha: Optional[float] = None   # coefficient of u
    left_beta: Optional[float] = None    # coefficient of du/dn
    right_alpha: Optional[float] = None  # coefficient of u
    right_beta: Optional[float] = None   # coefficient of du/dn
    
    def __post_init__(self):
        """Validate boundary condition parameters"""
        if self.type == 'robin':
            if any(v is None for v in [self.left_alpha, self.left_beta, self.right_alpha, self.right_beta]):
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")

