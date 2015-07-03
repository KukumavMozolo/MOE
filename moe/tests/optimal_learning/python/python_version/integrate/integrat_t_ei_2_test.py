# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy

import pytest
import matplotlib.pyplot as plt

from moe.optimal_learning.python.python_version.integrated_gaussian_process import GaussianProcess
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.expected_improvement import  ExpectedImprovement
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
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
        numpy.random.seed(7859)
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

    # def test_multistart_monte_carlo_expected_improvement_optimization(self):
    #     """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
    #     numpy.random.seed(7858)  # TODO(271): Monte Carlo only works for this seed
    #     index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
    #     domain, gaussian_process = self.gp_test_environments[index]
    #
    #     max_num_steps = 75  # this is *too few* steps; we configure it this way so the test will run quickly
    #     max_num_restarts = 5
    #     num_steps_averaged = 50
    #     gamma = 0.2
    #     pre_mult = 1.5
    #     max_relative_change = 1.0
    #     tolerance = 3.0e-2  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
    #     gd_parameters = GradientDescentParameters(
    #         max_num_steps,
    #         max_num_restarts,
    #         num_steps_averaged,
    #         gamma,
    #         pre_mult,
    #         max_relative_change,
    #         tolerance,
    #     )
    #     num_multistarts = 2
    #
    #     # Expand the domain so that we are definitely not doing constrained optimization
    #     expanded_domain = TensorProductDomain([ClosedInterval(-2.0, 2.0),ClosedInterval(0.0, 10.0)])
    #     num_to_sample = 1
    #     repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)
    #
    #     num_mc_iterations = 10000
    #     # Just any random point that won't be optimal
    #     points_to_sample = repeated_domain.generate_random_point_in_domain()
    #     cora_ei_eval = ExpectedImprovement(gaussian_process, points_to_sample, num_mc_iterations=num_mc_iterations)
    #     # Compute EI and its gradient for the sake of comparison
    #     cora_param = list([1,0,10,0.1])
    #     ei_initial = cora_ei_eval.compute_cora_expected_improvement(*cora_param,force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
    #     grad_ei_initial = cora_ei_eval.compute_cora_grad_expected_improvement(*cora_param)
    #
    #     ei_optimizer = GradientDescentOptimizer(repeated_domain, cora_ei_eval, gd_parameters)
    #     best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)
    #
    #     # Check that gradients are "small"
    #     cora_ei_eval.current_point = best_point
    #     ei_final = cora_ei_eval.compute_cora_expected_improvement(*cora_param,force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
    #     grad_ei_final = cora_ei_eval.compute_cora_grad_expected_improvement(*cora_param)
    #     self.assert_vector_within_relative(grad_ei_final, numpy.zeros(grad_ei_final.shape), tolerance)
    #
    #     # Check that output is in the domain
    #     assert repeated_domain.check_point_inside(best_point) is True
    #
    #     # Since we didn't really converge to the optimal EI (too costly), do some other sanity checks
    #     # EI should have improved
    #     assert ei_final >= ei_initial


    def test_multistart_monte_carlo_expected_improvement_varying(self):
        """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
        numpy.random.seed(7858)  # TODO(271): Monte Carlo only works for this seed

        max_num_steps = 2000  # this is *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 10
        num_steps_averaged = 0
        gamma = 0.5
        pre_mult = 1.5
        max_relative_change = 1.0
        tolerance = 3.0e-5  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
        gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )
        approx_grad = False
        max_func_evals = 15000
        max_metric_correc = 10
        pgtol = 1.0e-5
        epsilon = 1.0e-8

        lbfgs_parameters = LBFGSBParameters(
            approx_grad,
            max_func_evals,
            max_metric_correc,
            tolerance,
            pgtol,
            epsilon
        )
        num_multistarts = 6

        high = 6
        low = 0

        # Expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(-2.0, 2.0), ClosedInterval(low, high)])
        num_to_sample = 1
        repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)

        num_mc_iterations = 1000
        # Just any random point that won't be optimal
        points_to_sample = numpy.array([0.5, 0.5])
        point = SamplePoint(numpy.array([1, 1]), self.function_to_minimize([1, 1]), 0.05)
        point2 = SamplePoint(numpy.array([0.8, 1]), self.function_to_minimize([0.8, 1]), 0.05)
        point3 = SamplePoint(numpy.array([1.8, 5]), self.function_to_minimize([1.8, 5]), 0.05)
        point4 = SamplePoint(numpy.array([-1.1, 3]), self.function_to_minimize([-1.1, 3]), 0.05)

        data = HistoricalData(2, [point,point2,point3,point4])
        params = [1, low,  high]
        iters = 200
        # points_sampled = list()
        # function_vals = list()
        # points_sampled.append(1)
        # function_vals.append(self.function_to_minimize(numpy.array([1,0])))
        #theta = [3.56,0.8,1.35]
        theta = self.fit_hyperparameters(data)
        cov = SquareExponential(theta)
        gaussian_process = GaussianProcess(cov, data, *params)
        for i in range(iters):
            #find new point to sample
            cora_ei_eval = ExpectedImprovement(gaussian_process,  num_mc_iterations=num_mc_iterations)
            ei_optimizer = LBFGSBOptimizer(repeated_domain, cora_ei_eval, lbfgs_parameters)
            best_point, function_argument_list, starts = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, i)
            best_point[:,1] = numpy.random.uniform(low,high,1)#random time corresponds to rl testcase
            #evaluate point
            data = self.append_evaluation(data, best_point)
            #fit new gaussian process to data
            theta = self.fit_hyperparameters(data)
            cov = SquareExponential(theta)
            gaussian_process = GaussianProcess(cov, data, *params)
            #plot
            points_sampled = data.points_sampled
            function_vals = data.points_sampled_value
            plot_estimate(i, -2, 2, gaussian_process, cora_ei_eval, points_sampled, function_vals, theta, function_argument_list, starts)
        assert(False)




    def fit_hyperparameters(self, data):

        lml = GaussianProcessLogMarginalLikelihood(SquareExponential(numpy.array([0.1, 0.1])), data)

        max_num_steps = 400  # this is generally *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 5
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
        domain = TensorProductDomain([ClosedInterval(0.2, 5.), ClosedInterval(0.2, 5.), ClosedInterval(0.2, 5.)])
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
        return numpy.sin(alpha*5) + numpy.exp(-(time)**2) + 1.0 + numpy.random.normal(0,1)
        #return numpy.exp(-(alpha - numpy.sin(time/2)) ** 2) + numpy.sin(time)



    def append_evaluation(self, points, new_point):
        r"""
        :param points: HistoricalData
        :type points: HistoricalData
        """
        points.append_historical_data(new_point, self.function_to_minimize(new_point[0]), 0.01)
        return points

def multistart_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        iter
):
    random_starts = ei_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    x_tmp = numpy.zeros((num_multistarts,2))
    x = numpy.linspace(-1.95, 1.95, num_multistarts)
    x_tmp[:,0] = x
    best_point, random_starts_values, function_argument_list = multistart_optimize(ei_optimizer, starting_points=x_tmp)
    print(function_argument_list)
    # plot_multistart_starts_values(best_point,random_starts_values, iter)
    return best_point, function_argument_list, x_tmp[:,0]

def plot_estimate(iter,low, high, gaussian_process, expected_improvement, points_sampled, function_vals, hyperparams, function_argument_list, starts):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    ax1 = fig.add_subplot(3,1,1)
    n = 60
    x = numpy.linspace(low, high, n)
    x_tmp = numpy.zeros((n,2))
    x_tmp[:,0] = x
    y = gaussian_process.compute_mean_of_points(x_tmp)
    var = gaussian_process.compute_variance_of_points(x_tmp)
    var = numpy.sqrt(var[numpy.diag_indices(n)])
    low = y - var
    high = y + var
    ax1.plot(x,y)
    ax1.fill_between(x,low,high, color='gray')
    ax1.scatter(points_sampled[:-1,0], function_vals[:-1])
    ax1.scatter(points_sampled[-1,0], function_vals[-1], color = 'red')
    ax1.plot(x, var, color='red')
    ep = expected_improvement.evaluate_at_point_list(x_tmp)
    plt.ylim((-5,5))
    plt.xlim((-2,2))
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(x, ep, color = 'black')
    function_x = function_argument_list[:,0]
    function_y = numpy.zeros_like(function_argument_list)[:,0]
    colors = numpy.random.rand(function_x.shape[0])
    ax2.scatter(function_x,function_y, c = colors )
    ax2.scatter(starts, function_y, c = colors, alpha = 0.3)
    plt.xlim((-2,2))
    ax3 = fig.add_subplot(3,1,3)
    plt.xlim((-2,2))
    ax3.plot(x, -1*(numpy.sin(x*5) + 1. +numpy.exp(-(0.6)**2)))
    plt.title("$Hyperparams:$" +" "+"$\sigma_f= $" + "$"+'%.2f' % hyperparams[0] +"$"+ " $,l_1=$"+ "$" +'%.2f' % hyperparams[1] + "$"+ " $,l_2=$" + "$"+'%.2f' % hyperparams[2]+ "$")
    plt.savefig('/home/maxweule/Documents/Thesis/plots/' + str(iter) + '.png')
    plt.close()
