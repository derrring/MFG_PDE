from abc import ABC, abstractmethod

class MFGSolver(ABC):
    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def solve(self, Niter, **kwargs):
        """Solves the MFG system and returns U, M, and convergence info."""
        pass

    @abstractmethod
    def get_results(self):
        """Returns the computed U and M."""
        pass