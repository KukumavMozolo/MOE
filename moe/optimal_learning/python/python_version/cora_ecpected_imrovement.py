__author__ = 'mx'

import numpy as np
from expected_improvement import ExpectedImprovement
from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS



class CoraExpectedImprovement(ExpectedImprovement):


    def __init__(
            self,
            gaussian_process,
            c_idx,
            c_0,
            c_max,
            c_delta,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            randomness=None,
            mvndst_parameters=None
    ):
        super(CoraExpectedImprovement,self).__init__(
            gaussian_process,
            points_to_sample,
            points_being_sampled,
            num_mc_iterations,
            randomness,
            mvndst_parameters)
        self.c_idx = c_idx
        self.c_0 = c_0
        self.c_max = c_max
        self.c_delta = c_delta


    def compute_expected_improvement(self,  force_monte_carlo=False, force_1d_ei=False):
        r""" computes the average expected improvement of point to sample over all possible discrete values coregressor c can take.
        :param c_idx: position of coregressor in point
        :type c_idx: int64
        :param c_0: minimum value coregressor can take
        :type c_0: float64
        :param c_max: maximum value coregressor can take
        :type c_max: float64
        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: bool
        :param force_1d_ei: whether to force using the 1EI method. Used for testing purposes only. Takes precedence when force_monte_carlo is also True
        :type force_1d_ei: bool
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64
        """
        c_current = self.c_0
        ei = 0
        count = 0

        while c_current <= self.c_max:
            self._points_to_sample[:, self.c_idx] = c_current
            ei +=  super(CoraExpectedImprovement, self).compute_expected_improvement(force_monte_carlo, force_1d_ei)
            c_current += self.c_delta
            count += 1.0

        ei /= count
        return ei

    def compute_grad_expected_improvement(self, force_monte_carlo=False):
        r""" computes the gradient of cora expected improvement which is the average gradient over the gradients of the expected improvement
        for all discrete values c can take
        :param c_idx: position of coregressor in point
        :type c_idx: int64
        :param c_0: minimum value coregressor can take
        :type c_0: float64
        :param c_max: maximum value coregressor can take
        :type c_max: float64
        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        """
        #print("GradCoraEI")
        c_current = self.c_0
        n = self._points_to_sample.shape[0]
        grad_ei = np.zeros((n, self._gaussian_process.dim))
        count = 0.0

        while c_current <= self.c_max:
            #print(self.points_to_sample)
            self.points_to_sample[:,self.c_idx] = c_current
            grad_ei +=  super(CoraExpectedImprovement, self).compute_grad_expected_improvement(force_monte_carlo)
            c_current += self.c_delta
            count += 1.0

        grad_ei /= count
        #print("Gradient = " + str(grad_ei))
        return grad_ei

    compute_grad_objective_function = compute_grad_expected_improvement
    compute_objective_function = compute_expected_improvement