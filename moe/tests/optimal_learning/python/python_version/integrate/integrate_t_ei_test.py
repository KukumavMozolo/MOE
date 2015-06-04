# # -*- coding: utf-8 -*-
# """Test the Python implementation of Expected Improvement and its gradient."""
# import numpy
#
# import pytest
#
# import moe
# from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
# from moe.optimal_learning.python.geometry_utils import ClosedInterval
# from moe.optimal_learning.python.python_version.cora_ecpected_imrovement import CoraExpectedImprovement
# from moe.optimal_learning.python.python_version.covariance import SquareExponential
# from moe.optimal_learning.python.python_version.domain import TensorProductDomain
# from moe.optimal_learning.python.python_version.expected_improvement import multistart_expected_improvement_optimization
# from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
# from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer, LBFGSBParameters, LBFGSBOptimizer
# from moe.optimal_learning.python.repeated_domain import RepeatedDomain
# from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput
# from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood, multistart_hyperparameter_optimization
#
#
# class TestExpectedImprovement(GaussianProcessTestCase):
#
#     """Verify that the "naive" and "vectorized" EI implementations in Python return the same result.
#
#     The code for the naive implementation of EI is straightforward to read whereas the vectorized version is a lot more
#     opaque. So we verify one against the other.
#
#     Fully verifying the monte carlo implemetation (e.g., conducting convergence tests, comparing against analytic results)
#     is expensive and already a part of the C++ unit test suite.
#
#     """
#
#     precompute_gaussian_process_data = True
#
#     noise_variance_base = 0.002
#     dim = 2
#     num_hyperparameters = dim + 1
#
#     gp_test_environment_input = GaussianProcessTestEnvironmentInput(
#         dim,
#         num_hyperparameters,
#         0,
#         noise_variance_base=noise_variance_base,
#         hyperparameter_interval=ClosedInterval(3.0, 5.0),
#         lower_bound_interval=ClosedInterval(-2.0, 0.5),
#         upper_bound_interval=ClosedInterval(2.0, 3.5),
#         covariance_class=SquareExponential,
#         spatial_domain_class=TensorProductDomain,
#         hyperparameter_domain_class=TensorProductDomain,
#         gaussian_process_class=GaussianProcess,
#     )
#
#     num_sampled_list = (1, 2, 5, 10, 16, 20, 50)
#
#     num_mc_iterations = 747
#     rng_seed = 314
#
#     approx_grad = True
#     max_func_evals = 150000
#     max_metric_correc = 10
#     factr = 10.0
#     pgtol = 1e-10
#     epsilon = 1e-8
#     BFGS_parameters = LBFGSBParameters(
#         approx_grad,
#         max_func_evals,
#         max_metric_correc,
#         factr,
#         pgtol,
#         epsilon,
#     )
#
#     @classmethod
#     @pytest.fixture(autouse=True, scope='class')
#     def base_setup(cls):
#         """Run the standard setup but seed the RNG first (for repeatability).
#
#         It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
#         which makes it difficult to set meaningful tolerances for the checks.
#
#         """
#         numpy.random.seed(7859)
#         super(TestExpectedImprovement, cls).base_setup()
#
#
#     @classmethod
#     def _check_ei_symmetry(cls, ei_eval, point_to_sample, shifts):
#         """Compute ei at each ``[point_to_sample +/- shift for shift in shifts]`` and check for equality.
#
#         :param ei_eval: properly configured ExpectedImprovementEvaluator object
#         :type ei_eval: ExpectedImprovementInterface subclass
#         :param point_to_sample: point at which to center the shifts
#         :type point_to_sample: array of float64 with shape (1, )
#         :param shifts: shifts to use in the symmetry check
#         :type shifts: tuple of float64
#         :return: None; assertions fail if test conditions are not met
#
#         """
#         __tracebackhide__ = True
#         for shift in shifts:
#             ei_eval.current_point = point_to_sample - shift
#             left_ei = ei_eval.compute_expected_improvement()
#             left_grad_ei = ei_eval.compute_grad_expected_improvement()
#
#             ei_eval.current_point = point_to_sample + shift
#             right_ei = ei_eval.compute_expected_improvement()
#             right_grad_ei = ei_eval.compute_grad_expected_improvement()
#
#             cls.assert_scalar_within_relative(left_ei, right_ei, 0.0)
#             cls.assert_vector_within_relative(left_grad_ei, -right_grad_ei, 0.0)
#
#     # def test_multistart_monte_carlo_expected_improvement_optimization(self):
#     #     """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
#     #     numpy.random.seed(7858)  # TODO(271): Monte Carlo only works for this seed
#     #     index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
#     #     domain, gaussian_process = self.gp_test_environments[index]
#     #
#     #     max_num_steps = 75  # this is *too few* steps; we configure it this way so the test will run quickly
#     #     max_num_restarts = 5
#     #     num_steps_averaged = 50
#     #     gamma = 0.2
#     #     pre_mult = 1.5
#     #     max_relative_change = 1.0
#     #     tolerance = 3.0e-2  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
#     #     gd_parameters = GradientDescentParameters(
#     #         max_num_steps,
#     #         max_num_restarts,
#     #         num_steps_averaged,
#     #         gamma,
#     #         pre_mult,
#     #         max_relative_change,
#     #         tolerance,
#     #     )
#     #     num_multistarts = 2
#     #
#     #     # Expand the domain so that we are definitely not doing constrained optimization
#     #     expanded_domain = TensorProductDomain([ClosedInterval(-2.0, 2.0),ClosedInterval(0.0, 10.0)])
#     #     num_to_sample = 1
#     #     repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)
#     #
#     #     num_mc_iterations = 10000
#     #     # Just any random point that won't be optimal
#     #     points_to_sample = repeated_domain.generate_random_point_in_domain()
#     #     cora_ei_eval = ExpectedImprovement(gaussian_process, points_to_sample, num_mc_iterations=num_mc_iterations)
#     #     # Compute EI and its gradient for the sake of comparison
#     #     cora_param = list([1,0,10,0.1])
#     #     ei_initial = cora_ei_eval.compute_cora_expected_improvement(*cora_param,force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
#     #     grad_ei_initial = cora_ei_eval.compute_cora_grad_expected_improvement(*cora_param)
#     #
#     #     ei_optimizer = GradientDescentOptimizer(repeated_domain, cora_ei_eval, gd_parameters)
#     #     best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)
#     #
#     #     # Check that gradients are "small"
#     #     cora_ei_eval.current_point = best_point
#     #     ei_final = cora_ei_eval.compute_cora_expected_improvement(*cora_param,force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
#     #     grad_ei_final = cora_ei_eval.compute_cora_grad_expected_improvement(*cora_param)
#     #     self.assert_vector_within_relative(grad_ei_final, numpy.zeros(grad_ei_final.shape), tolerance)
#     #
#     #     # Check that output is in the domain
#     #     assert repeated_domain.check_point_inside(best_point) is True
#     #
#     #     # Since we didn't really converge to the optimal EI (too costly), do some other sanity checks
#     #     # EI should have improved
#     #     assert ei_final >= ei_initial
#
#
#     def test_multistart_monte_carlo_expected_improvement_varying(self):
#         """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
#         numpy.random.seed(7858)  # TODO(271): Monte Carlo only works for this seed
#
#         max_num_steps = 1000  # this is *too few* steps; we configure it this way so the test will run quickly
#         max_num_restarts = 1
#         num_steps_averaged = 0
#         gamma = 0.01
#         pre_mult = 1.5
#         max_relative_change = 1.0
#         tolerance = 3.0e-2  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
#         gd_parameters = GradientDescentParameters(
#             max_num_steps,
#             max_num_restarts,
#             num_steps_averaged,
#             gamma,
#             pre_mult,
#             max_relative_change,
#             tolerance,
#         )
#         num_multistarts = 1
#
#         # Expand the domain so that we are definitely not doing constrained optimization
#         expanded_domain = TensorProductDomain([ClosedInterval(-2.0, 2.0), ClosedInterval(0.0, 6.0)])
#         num_to_sample = 1
#         repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)
#
#         num_mc_iterations = 10000
#         # Just any random point that won't be optimal
#         points_to_sample = numpy.array([0.5, 0.5])
#         point = SamplePoint(numpy.array([1, 1]), self.function_to_minimize([1, 1]), 0.05)
#         point2 = SamplePoint(numpy.array([1, 0]), self.function_to_minimize([1, 0]), 0.05)
#         data = HistoricalData(2, [point, point2])
#
#         for i in range(1):
#             theta = self.fit_hyperparameters(data)
#             cov = SquareExponential(theta)
#             gaussian_process = GaussianProcess(cov, data)
#             cora_ei_eval = CoraExpectedImprovement(gaussian_process, 1, 0.0, 6.0, 1,  num_mc_iterations=num_mc_iterations)
#             ei_optimizer = GradientDescentOptimizer(repeated_domain, cora_ei_eval, gd_parameters)
#             best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)
#             # print("before" +str(best_point))
#             best_point[:,1] = numpy.random.uniform(0.0,6.0,1)
#             # print("after" + str(best_point))
#             self.append_evaluation(data, best_point)
#
#         for point in data.points_sampled:
#             print(str(point) + "has value" + str(self.function_to_minimize(point)))
#         assert(False)
#
#
#
#
#     def fit_hyperparameters(self, data):
#
#         lml = GaussianProcessLogMarginalLikelihood(SquareExponential(numpy.array([0.1, 0.1])), data)
#
#         max_num_steps = 400  # this is generally *too few* steps; we configure it this way so the test will run quickly
#         max_num_restarts = 5
#         num_steps_averaged = 0
#         gamma = 0.2
#         pre_mult = 1.0
#         max_relative_change = 0.3
#         tolerance = 1.0e-11
#         gd_parameters = GradientDescentParameters(
#             max_num_steps,
#             max_num_restarts,
#             num_steps_averaged,
#             gamma,
#             pre_mult,
#             max_relative_change,
#             tolerance,
#         )
#         domain = TensorProductDomain([ClosedInterval(1.0, 4.0), ClosedInterval(1.0, 4.0), ClosedInterval(1.0, 4.0)])
#         hyperOptimizer = GradientDescentOptimizer(domain, lml, gd_parameters)
#         best_hyperparameters = multistart_hyperparameter_optimization(hyperOptimizer, 1)
#         return best_hyperparameters
#
#     def function_to_minimize(self, point):
#         return -1*self.get_ctrsinshift(point[0], point[1])
#
#     def get_ctrsinshift(self, alpha, time):
#         '''
#         ctr model depending on hyperparameter and sinus of time
#         :param alpha:
#         :param time:
#         :return:
#         '''
#         return numpy.exp(-(alpha) ** 2) + 1.0 + numpy.sin(time)
#
#     def append_evaluation(self, points, new_point):
#         r"""
#         :param points: HistoricalData
#         :type points: HistoricalData
#         """
#         points.append_historical_data(new_point, self.function_to_minimize(new_point[0]), 0.05)
#
