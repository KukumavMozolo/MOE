# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy

import pytest
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool

from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogMarginalLikelihood
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
from moe.optimal_learning.python.python_version.log_likelihood import multistart_hyperparameter_optimization
from os.path import expanduser


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
        high = -3.0
        low = 3.0
        self.noiselvl = 0.3
        print('Multiple Diemsion Integration Test')
        theta_0 = [ 0.2, 1.54859412,  1.54859412,  1.54859412]#self.get_fixed_hyperparams(low, high, low, high)
        print(theta_0)
        for dsigma in [0.0, 0.5, 1.0, 1.5, 2.0]:
            numpy.random.seed(numpy.random.randint(1,9999))
            print("Here")
            #number of ego iterations
            iterations = 150
            nr_threads = 8
            runs = 500
            pre_samples = 1
            theta = numpy.copy(theta_0)
            plot = False
            pool = Pool(nr_threads)
            self.results = list()

            [pool.apply_async(self.time_stationary_ego,args=self.get_args(x, iterations, theta, dsigma, pre_samples, plot), callback=self.collect_results) for x in range(runs)]
            pool.close()
            pool.join()
            # args = self.get_args(1, iterations, theta, dsimgma, pre_samples, plot)
            # self.results = self.time_stationary_ego(*args)
            res = numpy.asarray(self.results)
            print(res)
            home = expanduser("~")
            location = home + '/Documents/Thesis/results/results_multi_dsigma_' +str(dsigma) + '_runs_'+str(runs)+ '_pre_'+str(pre_samples) + '_iters_'+str(iterations)
            numpy.save(location, res)
            print('Results where saved to: ' + location)
            #print(res)
            #mean = res.mean(axis=0)
            #asd = numpy.load('/home/max/Documents/Thesis/results/results' + str(nr_threads*runs) + '.npy')

        assert(True)
    def collect_results(self, res):
        self.results.append(res)

    def get_args(self, i, iterations, theta, sigma_2 = 0, pre_samples = 10, plot = False):
        num_multistarts = 4
        #define integral bounds ove time
        high1 = 3.0
        low1 = -3.0
        high2= 3.0
        low2 = -3.0
        # Expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(low1, high1), ClosedInterval(low2, high2), ClosedInterval(low1, high1)])
        num_to_sample = 1
        repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)
        #variable that holds all parameters to create integrated gaussian process
        params = [[1,2], [low1,low2],  [high1,high2]]
        #number of ego iterations
        # get gradient descent and lbfgs parameters
        _, lbfgs_parameters = self.get_params()

        points = self.get_starting_points(pre_samples, low1, high1, low2,high2)
        data = HistoricalData(3, points)
        return theta, repeated_domain,iterations,data, params, lbfgs_parameters, num_multistarts, i, sigma_2, plot

    def get_fixed_hyperparams(self, low1, high1, low2,high2):
        points_for_fitting = self.get_starting_points(1000, low1, high1, low2,high2)
        data = HistoricalData(3, points_for_fitting)
        theta = self.fit_hyperparameters(data)
        return theta
    def get_params(self):
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
        pgtol = 1.0e-8
        epsilon = 1.0e-8

        lbfgs_parameters = LBFGSBParameters(
            approx_grad,
            max_func_evals,
            max_metric_correc,
            tolerance,
            pgtol,
            epsilon
        )
        return gd_parameters, lbfgs_parameters


    def time_stationary_ego(self, theta, repeated_domain, iterations, data, params, lbfgs_parameters, num_multistarts,threadid, sigma_2=0, plot = True):
        res = numpy.zeros((iterations))
        idx = params[0]
        low = params[1]
        high = params[2]
        # theta = self.fit_hyperparameters(data)
        theta[idx] += sigma_2
        print(theta)
        cov = SquareExponential(theta)
        print(data)
        gaussian_process = IntegratedGaussianProcess(cov, data, *params)
        ts1 = numpy.random.uniform(low[0],high[0],10)
        ts2 = numpy.random.uniform(low[0],high[0],10)
        ts = [ts1,ts2]
        pers = self.cartesian([ts1,ts2])
        print(ts)
        for i in range(iterations):
            print('Thread: '+ str( threadid) + ' at : ' + str(100*i/iterations) + '%')
            #find new point to sample
            cora_ei_eval = ExpectedImprovement(gaussian_process, T=ts)
            ei_optimizer = LBFGSBOptimizer(repeated_domain, cora_ei_eval, lbfgs_parameters)
            best_point, function_argument_list, starts = self.multistart_expected_improvement_optimization(ei_optimizer, num_multistarts)
            best_point[0,1:] = pers[numpy.random.randint(0,100)]#random time corresponds to rl testcase
            #evaluate point
            data = self.append_evaluation(data, best_point, self.noiselvl)
            #fit new gaussian process to data
            #theta =self.fit_hyperparameters(data)
            cov = SquareExponential(theta)
            gaussian_process = IntegratedGaussianProcess(cov, data, *params)
            best_gp_mean = self.get_optimum(gaussian_process)
            points_sampled = data.points_sampled
            function_vals = data.points_sampled_value
            if(plot == True):
                print("plotting")
                self.plot_estimate(i, low, high, gaussian_process, cora_ei_eval, points_sampled, function_vals, theta, function_argument_list, starts, best_gp_mean ,threadid)
            res[i]= best_gp_mean[0]
        return res

    def cartesian(self,arrays, out=None):
            """
            Generate a cartesian product of input arrays.

            Parameters
            ----------
            arrays : list of array-like
                1-D arrays to form the cartesian product of.
            out : ndarray
                Array to place the cartesian product in.

            Returns
            -------
            out : ndarray
                2-D array of shape (M, len(arrays)) containing cartesian products
                formed of input arrays.

            Examples
            --------
            >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
            array([[1, 4, 6],
                   [1, 4, 7],
                   [1, 5, 6],
                   [1, 5, 7],
                   [2, 4, 6],
                   [2, 4, 7],
                   [2, 5, 6],
                   [2, 5, 7],
                   [3, 4, 6],
                   [3, 4, 7],
                   [3, 5, 6],
                   [3, 5, 7]])

            """

            arrays = [numpy.asarray(x) for x in arrays]
            dtype = arrays[0].dtype

            n = numpy.prod([x.size for x in arrays])
            if out is None:
                out = numpy.zeros([n, len(arrays)], dtype=dtype)

            m = n / arrays[0].size
            out[:,0] = numpy.repeat(arrays[0], m)
            if arrays[1:]:
                self.cartesian(arrays[1:], out=out[0:m,1:])
                for j in xrange(1, arrays[0].size):
                    out[j*m:(j+1)*m,1:] = out[0:m,1:]
            return out


    def fit_hyperparameters(self, data):

        lml = GaussianProcessLogMarginalLikelihood(SquareExponential(numpy.array([0.1, 0.1, 0.1, 0.1])), data)

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
        domain = TensorProductDomain([ClosedInterval(0.2, 10.0), ClosedInterval(0.2, 10.2), ClosedInterval(0.2, 10.2), ClosedInterval(0.2, 10.2)])
        hyperOptimizer = GradientDescentOptimizer(domain, lml, gd_parameters)
        best_hyperparameters = multistart_hyperparameter_optimization(hyperOptimizer, 1)
        return best_hyperparameters

    def function_to_minimize(self, point):
        return -1*self.get_4dfunction(point)

    def get_4dfunction(self, point):
        if isinstance(point, list):
            point = numpy.asarray(point)
        res = 2*numpy.exp(-0.5*numpy.dot(point,numpy.transpose(point))) + numpy.random.normal(0, self.noiselvl)  -0.30226193759
        return res
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

    def append_evaluation(self, points, new_point, var):
        r"""
        :param points: HistoricalData
        :type points: HistoricalData
        """
        points.append_historical_data(new_point, self.function_to_minimize(new_point[0]), var)
        return points

    def get_starting_points(self, n, low1, high1, low2, high2):
        points = list()
        for i in range(n):
            x = numpy.random.uniform(low1, high1)
            y = numpy.random.uniform(low2, high2)
            z = numpy.random.uniform(low2, high2)
            points.append(SamplePoint(numpy.array([x, y, z]), self.function_to_minimize([x, y, z]), self.noiselvl))
        return points

    def multistart_expected_improvement_optimization(self,
        ei_optimizer, num_multistarts):
        x_tmp = numpy.zeros((num_multistarts,3))
        #x = numpy.linspace(-1.95, 1.95, num_multistarts)
        x = numpy.random.uniform(-3.0, 3.0, size=(1,num_multistarts))
        x_tmp[:,0] = x
        best_point, random_starts_values, function_argument_list = multistart_optimize(ei_optimizer, starting_points=x_tmp)
        return best_point, function_argument_list, x_tmp[:,0]

    def plot_estimate(self, iter,lowx, highx, gaussian_process,
                      expected_improvement, points_sampled, function_vals, hyperparams,
                      function_argument_list, starts, best_gp_mean, threadid):
        lowx = lowx[0]
        highx = highx[0]
        fig = plt.figure(figsize=(6,10))
        fig.subplots_adjust(hspace=.5)
        ax1 = fig.add_subplot(3,1,1)
        n = 60
        x = numpy.linspace(lowx, highx, n)
        lowx = lowx-0.1
        highx = highx+0.1
        x_tmp = numpy.zeros((n,3))
        x_tmp[:,0] = x
        y = gaussian_process.compute_mean_of_points(x_tmp)
        var = gaussian_process.compute_variance_of_points(x_tmp)
        var = numpy.sqrt(var[numpy.diag_indices(n)])
        low = y - 2*var
        high = y + 2*var
        ax1.plot(x,y)
        ax1.fill_between(x,low,high, color='gray')
        ax1.scatter(points_sampled[:-1,0], function_vals[:-1])
        ax1.scatter(points_sampled[-1,0], function_vals[-1], color = 'red')
        ax1.scatter(points_sampled[-1,0], function_vals[-1], color = 'red')
        ax1.plot(x, var, color='red')
        ax1.scatter(best_gp_mean[0], best_gp_mean[1], color = 'black')
        ep = expected_improvement.evaluate_at_point_list(x_tmp)
        plt.ylim((-5,5))
        plt.xlim((lowx,highx))
        ax2 = fig.add_subplot(3,1,2)
        ax2.plot(x, ep, color = 'black')
        function_x = function_argument_list[:,0]
        function_y = numpy.zeros_like(function_argument_list)[:,0]
        colors = numpy.random.rand(function_x.shape[0])
        ax2.scatter(function_x,function_y, c = colors )
        ax2.scatter(starts, function_y, c = colors, alpha = 0.3)
        plt.xlim((lowx,highx))
        ax3 = fig.add_subplot(3,1,3)
        plt.xlim((lowx,highx))
        ax3.plot(x, -1*(numpy.sin(x*5) + 1.))
        plt.title("$Hyperparams:$" +" "+"$\sigma_f= $" + "$"+'%.2f' % hyperparams[0] +"$"+ " $,l_1=$"+ "$" +'%.2f' % hyperparams[1] + "$"+ " $,l_2=$" + "$"+'%.2f' % hyperparams[2]+ "$")
        home = expanduser("~")
        plt.savefig(home +'/Documents/Thesis/plots/iters/' + str(iter) + '.png')
        plt.close()

    def get_optimum(self,gp, n_starts = 4):
        approx_grad = False
        max_func_evals = 15000
        max_metric_correc = 10
        pgtol = 1.0e-8
        epsilon = 1.0e-8
        tolerance = 3.0e-5
        lbfgs_parameters = LBFGSBParameters(
            approx_grad,
            max_func_evals,
            max_metric_correc,
            tolerance,
            pgtol,
            epsilon
        )
        optimizable_gp = OptimizableGaussianProcess(gp)
        expanded_domain = TensorProductDomain([ClosedInterval(-3.0, 3.0), ClosedInterval(-3.0, 3.0), ClosedInterval(-3.0, 3.0)])
        gp_optimizer = LBFGSBOptimizer(expanded_domain, optimizable_gp, lbfgs_parameters)
        #use allways same starting position here, assuming optimum is known
        x = numpy.linspace(-3.0, 3.0, n_starts)
        x_tmp = numpy.zeros((n_starts,3))
        x_tmp[:,0] = x
        best_point, random_starts_values, function_argument_list = multistart_optimize(gp_optimizer, starting_points=x_tmp, num_multistarts = n_starts)
        best_point[1:] = 0
        return best_point





import copy_reg
import types



def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)