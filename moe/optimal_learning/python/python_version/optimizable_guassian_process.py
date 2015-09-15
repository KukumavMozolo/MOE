import numpy
import scipy
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface

__author__ = 'mw'
class OptimizableGaussianProcess( OptimizableInterface):

    r"""Implementation of Expected Improvement computation in Python: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    When available, fast, analytic formulas replace monte-carlo loops.

    .. Note:: Equivalent methods of ExpectedImprovementInterface and OptimizableInterface are aliased below (e.g.,
      compute_expected_improvement and compute_objective_function, etc).

    See :class:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface` for further details.

    """

    def __init__(
            self,
            gaussian_process
    ):
        self._gaussian_process = gaussian_process


        self._current_point = numpy.zeros((1, gaussian_process.dim))


    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.dim

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return self._current_point

    def set_current_point(self, current_point):
        self._current_point = current_point.T

    current_point = property(get_current_point, set_current_point)

    def compute_objective_function(self):
        return -1*self._gaussian_process.compute_mean_of_points(self._current_point.reshape(1,2))

    def compute_grad_objective_function(self):

        return -1*self._gaussian_process.compute_grad_mean_of_points(self._current_point.reshape(1,2))

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')

