import numpy
from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS
from moe.optimal_learning.python.python_version.optimization import multistart_optimize
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface

__author__ = 'mw'

def multistart_gp_optimization(
        gp_optimizer,
        num_multistarts,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Optimizable to find the best point according to gp mean

    :param gp_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type gp_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``ei_optimizer``
    :type num_multistarts: int > 0
    :param randomness: random source(s) used to generate multistart points and perform monte-carlo integration (when applicable) (UNUSED)
    :type randomness: (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the expected improvement (solving the q,p-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_evaluator.dim)

    """
    random_starts = gp_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    best_point, _ = multistart_optimize(gp_optimizer, starting_points=random_starts)

    # TODO(GH-59): Have GD actually indicate whether updates were found.
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return [best_point]


class OptimizableGaussianProcess( OptimizableInterface):

    r"""Optimizable wrapper for GP, allows to perform gradient descent on gaussian process mean
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
        return -1.0*self._gaussian_process.compute_mean_of_points(self._current_point.reshape(1,self.dim()))

    def compute_grad_objective_function(self):
        return -1.0*self._gaussian_process.compute_grad_mean_of_points(self._current_point.reshape(1,self.dim()))

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')

