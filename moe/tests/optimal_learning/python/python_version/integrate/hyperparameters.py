# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy

import pytest
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool

from moe.optimal_learning.python.python_version.integrated_gaussian_process import IntegratedGaussianProcess
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.expected_improvement import  ExpectedImprovement
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.optimizable_guassian_process import OptimizableGaussianProcess
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer, \
    multistart_optimize, LBFGSBOptimizer, LBFGSBParameters
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood, multistart_hyperparameter_optimization


class TestExpectedImprovement(GaussianProcessTestCase):

    """Verify that the "naive" and "vectorized" EI implementations in Python return the same result.

    The code for the naive implementation of EI is straightforward to read whereas the vectorized version is a lot more
    opaque. So we verify one against the other.

    Fully verifying the monte carlo implemetation (e.g., conducting convergence tests, comparing against analytic results)
    is expensive and already a part of the C++ unit test suite.

    """


    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def base_setup(cls):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        super(TestExpectedImprovement, cls).base_setup()

    @classmethod
    def _check_ei_symmetry(cls, ei_eval, point_to_sample, shifts):
        """Compute ei at each ``[point_to_sample +/- shift for shift in shifts]`` and check for equality.

        :param ei_eval: properly configured ExpectedImprovementEvaluator object
        :type ei_eval: ExpectedImprovementInterface subclass
        :param point_to_sample: point at which to center the shifts
        :type point_to_sample: array of float64 with shape (1, )
        :param shifts: shifts to use in the symmetry check
        :type shifts: tuple of float64
        :return: None; assertions fail if test conditions are not met

        """
        __tracebackhide__ = True
        for shift in shifts:
            ei_eval.current_point = point_to_sample - shift
            left_ei = ei_eval.compute_expected_improvement()
            left_grad_ei = ei_eval.compute_grad_expected_improvement()

            ei_eval.current_point = point_to_sample + shift
            right_ei = ei_eval.compute_expected_improvement()
            right_grad_ei = ei_eval.compute_grad_expected_improvement()

            cls.assert_scalar_within_relative(left_ei, right_ei, 0.0)
            cls.assert_vector_within_relative(left_grad_ei, -right_grad_ei, 0.0)

    def test_time_stationary_ego_plot(self):
        high = 0.6
        low = -0.5
        self.noiselvl = 0.3
        theta = self.get_fixed_hyperparams(low, high)
        print(theta)
        location = '/home/kaw/Dokumente/Thesis/results/hyperparameters_noise_' +str(self.noiselvl)
        numpy.save(location, theta)
        assert(True)

    def get_fixed_hyperparams(self, low, high):
        points_for_fitting = self.get_starting_points(500, low, high)
        data = HistoricalData(2, points_for_fitting)
        theta = self.fit_hyperparameters(data)
        return theta

    def fit_hyperparameters(self, data):

        lml = GaussianProcessLogMarginalLikelihood(SquareExponential(numpy.array([0.1, 0.1, 0.1])), data)

        max_num_steps = 400  # this is generally *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 10
        num_steps_averaged = 0
        gamma = 0.2
        pre_mult = 1.0
        max_relative_change = 0.3
        tolerance = 1.0e-11
        gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )
        domain = TensorProductDomain([ClosedInterval(0.1, 6.0), ClosedInterval(0.1, 2.2), ClosedInterval(0.1, 2.2)])
        hyperOptimizer = GradientDescentOptimizer(domain, lml, gd_parameters)
        best_hyperparameters = multistart_hyperparameter_optimization(hyperOptimizer, 1)
        return best_hyperparameters

    def function_to_minimize(self, point):
        return -1*self.get_ctrsinshift(point[0], point[1])

    def get_ctrsinshift(self, alpha, time):
        '''
        ctr model depending on hyperparameter and sinus of time
        :param alpha:
        :param time:
        :return:
        '''
        #return numpy.exp(-(alpha) ** 2) + 1.0 +  numpy.sin(time) + 0.3*numpy.random.normal(0,1)
        #return numpy.sin(alpha*5) * numpy.exp(-(time)**2) + 1.0 + numpy.random.normal(0,1)
        #return numpy.exp(-(alpha - numpy.sin(time/2)) ** 2) + numpy.sin(time)
        return self.func2(alpha, time, 1)

    def func2(self, a, b, n):
        def x(a,b):
            return  numpy.sin(a*5)

        def y(a,b):
            tmp = numpy.copy(a)
            if(a >0):
                return 0.5*numpy.exp(-(b)**2)
            else:
                return numpy.sin(b*5)

        e = numpy.random.normal(0, 1, (n,n))
        #return np.sin(a*5) * np.exp(-(b)**2) + 1.0 #+ e
        return x(a,b)*y(a,b)  + numpy.random.normal(0, self.noiselvl)



    def get_starting_points(self, n, low, high):
        points = list()
        for i in range(n):
            x = numpy.random.uniform(-0.5, 0.7)
            y = numpy.random.uniform(low, high)
            points.append(SamplePoint(numpy.array([x, y]), self.function_to_minimize([x, y]), self.noiselvl))
        return points

    def multistart_expected_improvement_optimization(self,
        ei_optimizer, num_multistarts):
        x_tmp = numpy.zeros((num_multistarts,2))
        #x = numpy.linspace(-1.95, 1.95, num_multistarts)
        x = numpy.random.uniform(-0.5, 0.6, size=(1,num_multistarts))
        x_tmp[:,0] = x
        best_point, random_starts_values, function_argument_list = multistart_optimize(ei_optimizer, starting_points=x_tmp)
        return best_point, function_argument_list, x_tmp[:,0]







import copy_reg
import types



def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)