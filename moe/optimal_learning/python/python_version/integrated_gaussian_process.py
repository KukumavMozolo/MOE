# -*- coding: utf-8 -*-
"""Implementation (Python) of GaussianProcessInterface.

This file contains a class to manipulate a Gaussian Process through numpy/scipy.

See :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface` for more details.

"""
import copy

import numpy
from scipy.special import erf

import scipy.linalg
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from moe.optimal_learning.python.interfaces.gaussian_process_interface import GaussianProcessInterface
from moe.optimal_learning.python.python_version import python_utils


#: Minimum allowed standard deviation value in ``ComputeGradCholeskyVarianceOfPointsPerPoint`` (= machine precision).
#: Values that are too small result in problems b/c we may compute ``std_dev/var`` (which is enormous
#: if ``std_dev = 1.0e-150`` and ``var = 1.0e-300``) since this only arises when we fail to compute ``std_dev = var = 0.0``.
#: Note: this is only relevant if noise = 0.0; this minimum will not affect GPs with noise since this value
#: is below the smallest amount of noise users can meaningfully add.
#: This value was chosen to be consistent with the singularity condition in scipy.linalg.cho_factor
#: and tested for robustness with the setup in test_1d_analytic_ei_edge_cases().
MINIMUM_STD_DEV_GRAD_CHOLESKY = numpy.finfo(numpy.float64).eps



class GaussianProcess(GaussianProcessInterface):

    r"""Implementation of a GaussianProcess strictly in Python.

    .. Note:: Comments in this class are copied from this object's superclass in :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface`.

    Object that encapsulates Gaussian Process Priors (GPPs).  A GPP is defined by a set of
    (sample point, function value, noise variance) triples along with a covariance function that relates the points.
    Each point has dimension dim.  These are the training data; for example, each sample point might specify an experimental
    cohort and the corresponding function value is the objective measured for that experiment.  There is one noise variance
    value per function value; this is the measurement error and is treated as N(0, noise_variance) Gaussian noise.

    GPPs estimate a real process \ms f(x) = GP(m(x), k(x,x'))\me (see file docs).  This class deals with building an estimator
    to the actual process using measurements taken from the actual process--the (sample point, function val, noise) triple.
    Then predictions about unknown points can be made by sampling from the GPP--in particular, finding the (predicted)
    mean and variance.  These functions (and their gradients) are provided in ComputeMeanOfPoints, ComputeVarianceOfPoints,
    etc.

    Further mathematical details are given in the implementation comments, but we are essentially computing:

    | ComputeMeanOfPoints    : ``K(Xs, X) * [K(X,X) + \sigma_n^2 I]^{-1} * y = Ks^T * K^{-1} * y``
    | ComputeVarianceOfPoints: ``K(Xs, Xs) - K(Xs,X) * [K(X,X) + \sigma_n^2 I]^{-1} * K(X,Xs) = Kss - Ks^T * K^{-1} * Ks``

    This (estimated) mean and variance characterize the predicted distributions of the actual \ms m(x), k(x,x')\me
    functions that underly our GP.

    The "independent variables" for this object are ``points_to_sample``. These points are both the "p" and the "q" in q,p-EI;
    i.e., they are the parameters of both ongoing experiments and new predictions. Recall that in q,p-EI, the q points are
    called ``points_to_sample`` and the p points are called ``points_being_sampled.`` Here, we need to make predictions about
    both point sets with the GP, so we simply call the union of point sets ``points_to_sample.``

    In GP computations, there is really no distinction between the "q" and "p" points from EI, ``points_to_sample`` and
    ``points_being_sampled``, respectively. However, in EI optimization, we only need gradients of GP quantities wrt
    ``points_to_sample``, so users should call members functions with ``num_derivatives = num_to_sample`` in that context.

    """

    def __init__(self, covariance_function, historical_data, idx,low, high):
        """Construct a GaussianProcess object that knows how to call C++ for evaluation of member functions.

        TODO(GH-56): Have a way to set private RNG state for self.sample_point_from_gp()

        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object

        """
        self._covariance = copy.deepcopy(covariance_function)
        self._historical_data = copy.deepcopy(historical_data)
        self.idx = idx
        self.low = low
        self.high = high
        self._build_precomputed_data()


    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._historical_data.dim

    @property
    def num_sampled(self):
        """Return the number of sampled points."""
        return self._historical_data.num_sampled

    @property
    def _points_sampled(self):
        """Return the coordinates of the already-sampled points; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled

    @property
    def _points_sampled_value(self):
        """Return the function values measured at each of points_sampled; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_value

    @property
    def _points_sampled_noise_variance(self):
        """Return the noise variance associated with points_sampled_value; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_noise_variance

    def get_covariance_copy(self):
        """Return a copy of the covariance object specifying the Gaussian Process.

        :return: covariance object encoding assumptions about the GP's behavior on our data
        :rtype: interfaces.covariance_interface.CovarianceInterface subclass

        """
        return copy.deepcopy(self._covariance)

    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.

        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: data_containers.HistoricalData

        """
        return copy.deepcopy(self._historical_data)

    def _build_precomputed_data(self):
        """Set up precomputed data (cholesky factorization of K and K^-1 * y)."""
        if self.num_sampled == 0:
            self._K_chol = numpy.array([])
            self._K_inv_y = numpy.array([])
        else:
            covariance_matrix = python_utils.build_covariance_matrix(
                self._covariance,
                self._points_sampled,
                noise_variance=self._points_sampled_noise_variance,
            )

            C = self._build_integrated_term_maxtrix(self._covariance, self._points_sampled)
            self._K_Inv = numpy.linalg.inv(covariance_matrix)
            self._K_C = numpy.empty((covariance_matrix.shape[0],covariance_matrix.shape[0]))
            for i in range(covariance_matrix.shape[0]):
                for j in range(covariance_matrix.shape[0]):
                     self._K_C[i,j] = C[i,j] * self._K_Inv[i,j]

            self._K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
            self._K_inv_y = scipy.linalg.cho_solve(self._K_chol, self._points_sampled_value)

    def _compute_mean_of_points(self, points_to_sample):
        r"""Compute the mean of this GP at each of point of ``Xs`` (``points_to_sample``).

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_mean_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: mean: where mean[i] is the mean at points_to_sample[i]
        :rtype: array of float64 with shape (num_to_sample)

        """
        if self.num_sampled == 0:
            return numpy.zeros(points_to_sample.shape[0])

        K_star = self._build_integrated_covariance_maxtrix(
            self._covariance,
            self._points_sampled,
            points_to_sample,
        )
        mu_star = numpy.dot(K_star.T, self._K_inv_y)
        return mu_star * 1.0/(self.high - self.low)


    def _build_integrated_covariance_maxtrix(self, covariance, points_sampled, points_to_sample):
        _hyperparameters = covariance.get_hyperparameters()
        covariance = SquareExponential(_hyperparameters[:-1])
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        cov_mat = numpy.empty((points_sampled.shape[0], points_to_sample.shape[0]), order='F')
        normalization1 = numpy.sqrt(numpy.pi/2.0)*l
        normalization2 = (numpy.sqrt(2.0) * l)
        for j, point_two in enumerate(points_to_sample):
            for i, point_one in enumerate(points_sampled):
                tmp_point_two = point_two[:-1]
                tmp_point_one = point_one[:-1]
                cov_mat[i, j] = covariance.covariance(tmp_point_one, tmp_point_two)
                #integrate only over the last index corresponding to the coregressor that should be marginalized
                #sqrt(pi/2)*L * (erf(\frac{b-x}{\sqrt(2)L} - erf(\frac{a-x}{\sqrt(2)L}
                high_bound = (self.high - point_one[-1])/ normalization2
                high_bound = erf(high_bound)
                low_bound =  (self.low - point_one[-1])/ normalization2
                low_bound = erf(low_bound)
                diff = high_bound - low_bound
                cov_mat[i, j] = cov_mat[i, j] * normalization1 * diff
        return cov_mat



    def _compute_grad_mean_of_points(self, points_to_sample, num_derivatives=-1):
        r"""Compute the gradient of the mean of this GP at each of point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        Observe that ``grad_mu`` is nominally sized: ``grad_mu[num_to_sample][num_to_sample][dim]``. This is
        the the d-th component of the derivative evaluated at the i-th input wrt the j-th input.
        However, for ``0 <= i,j < num_to_sample``, ``i != j``, ``grad_mu[j][i][d] = 0``.
        (See references or implementation for further details.)
        Thus, ``grad_mu`` is stored in a reduced form which only tracks the nonzero entries.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_mean_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_mu: gradient of the mean of the GP. ``grad_mu[i][d]`` is actually the gradient
          of ``\mu_i`` wrt ``x_{i,d}``, the d-th dim of the i-th entry of ``points_to_sample``.
        :rtype: array of float64 with shape (num_derivatives, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)
        grad_K_star = numpy.empty((num_derivatives, self._points_sampled.shape[0], self.dim))
        for i, point_one in enumerate(points_to_sample[:num_derivatives, ...]):
            for j, point_two in enumerate(self._points_sampled):
                grad_K_star[i, j, ...] = self._compute_integrated_grad_covariance(point_one, point_two)

        # y_{k,i} = A_{k,j,i} * x_j
        grad_mu_star = numpy.einsum('ijk, j', grad_K_star, self._K_inv_y)

        return grad_mu_star

    def _compute_integrated_grad_covariance(self, point_one, point_two):
        r"""gradient = cov(x_{*i}^',x_j^')*\frac{1}{L} * ( x_j - x_*) * \sqrt(pi/2) * (erf(\frac{b-x_{jn}}{\sqrt(2)L}) - erf(\frac{a-x_{jn}}{\sqrt(2)L}))
        for i < n and for i = t: gradient = 0
        """
        _hyperparameters = self._covariance.get_hyperparameters()
        covariance = SquareExponential(_hyperparameters[:-1])
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        grad_cov = point_two - point_one
        tmp_point_two = point_two[:-1]
        tmp_point_one = point_one[:-1]
        grad_cov *= covariance.covariance(tmp_point_one, tmp_point_two) #\sigma_f^2 exp(-\frac{1}{2l^2} |X_{x,i}' - X_{xj}'|^2)
        normalization = numpy.sqrt(2.0) * l
        high_bound = (self.high - point_two[-1]) / normalization
        high_bound = erf(high_bound)
        low_bound = (self.low - point_two[-1]) / normalization
        low_bound = erf(low_bound)
        diff = high_bound - low_bound
        grad_cov = grad_cov * diff *numpy.sqrt(numpy.pi/2.)/l
        grad_cov[-1] = 0.0 #gradient for marginal is zero
        return grad_cov

    def _compute_integrated_grad_covariance2(self, point_one, point_two):
        _hyperparameters = self._covariance.get_hyperparameters()
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        diff = point_one - point_two
        normalisation = numpy.sqrt(2.0) * l
        higher_bound = (self.high - point_one[-1])/normalisation
        higher_bound = erf(higher_bound)
        lower_bound = (self.low - point_one[-1])/normalisation
        lower_bound = erf(lower_bound)
        diff *= (higher_bound - lower_bound)
        return diff

    def _compute_integrated_grad_covariance3(self, point_one, point_two):
        _hyperparameters = self._covariance.get_hyperparameters()
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        covariance = SquareExponential(_hyperparameters[:-1])
        diff = point_one - point_two
        diff[-1]=0
        tmp_point_one = point_one[:-1]
        tmp_point_two = point_two[:-1]
        diff*= covariance.covariance(tmp_point_one,tmp_point_two)
        return diff

    def _build_integrated_covariance_maxtrix2(self, covariance, points_sampled, points_to_sample):
        _hyperparameters = covariance.get_hyperparameters()
        covariance = SquareExponential(_hyperparameters[:-1])
        cov_mat = numpy.empty((points_sampled.shape[0], points_to_sample.shape[0]), order='F')
        for j, point_two in enumerate(points_to_sample):
            for i, point_one in enumerate(points_sampled):
                tmp_point_two = point_two[:-1]
                tmp_point_one = point_one[:-1]
                cov_mat[i, j] = covariance.covariance(tmp_point_one, tmp_point_two)
        return cov_mat


    def _compute_variance_of_points(self, points_to_sample):
        r"""Compute the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        The variance matrix is symmetric although we currently return the full representation.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: var_star: variance matrix of this GP
        :rtype: array of float64 with shape (num_to_sample, num_to_sample)

        """
        _hyperparameters = self._covariance.get_hyperparameters()
        sigma_f = _hyperparameters[0]
        sigma_f_sq =sigma_f * sigma_f
        l = numpy.copy(_hyperparameters[-1]) # is it the last??

        covariance = SquareExponential(_hyperparameters[:-1])
        var_star = numpy.empty((points_to_sample.shape[0],points_to_sample.shape[0]))
        norm1 = 2*l*(numpy.sqrt(numpy.pi/2.0))
        norm2 = numpy.sqrt(2.0) * l
        norm3 = numpy.sqrt(2.0/numpy.pi) *l
        diff = (self.high-self.low)
        const = norm1* (diff * erf(diff/norm2) + norm3 * (numpy.exp(-diff**2/(2.0*l**2))))
        for i, point_one in enumerate(points_to_sample):
            for j, point_two in enumerate(points_to_sample):
                tmp_point_one = point_one[:-1]
                tmp_point_two = point_two[:-1]
                var_star[i,j] = covariance.covariance(tmp_point_one, tmp_point_two) * const

        K_star = self._build_integrated_covariance_maxtrix2(
            self._covariance,
            self._points_sampled,
            points_to_sample,
        )
        K_star_K_C_Inv_K_star = numpy.dot(numpy.dot(K_star.T, self._K_C), K_star)


        # V = scipy.linalg.solve_triangular(
        #     self._K_C_chol[0],
        #     K_star,
        #     lower=self._K_C_chol[1],
        #     overwrite_b=True,
        # )

        normalization = l**2 * numpy.pi/2.0

        # cheaper to go through scipy.linalg.get_blas_funcs() which can compute A = alpha*B*C + beta*A in one pass
        # vtv_norm = numpy.dot(V.T, V) * normalization
        # tmp = var_star - vtv_norm
        tmp = var_star - K_star_K_C_Inv_K_star * normalization
        return tmp * 1.0/(self.high - self.low)

    def _build_integrated_term_maxtrix(self, covariance, points_sampled):
        _hyperparameters = covariance.get_hyperparameters()
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        diffs = numpy.empty((1, points_sampled.shape[0]), order='F')
        norm = numpy.sqrt(2.0) * l
        for i, point_one in enumerate(points_sampled):
            high_bound = erf((self.high - point_one[-1])/(norm))
            low_bound = erf((self.low - point_one[-1])/(norm))
            diffs[0,i] = high_bound - low_bound
        return numpy.dot(diffs.T,diffs)

    def compute_variance_of_points(self, points_to_sample):
        return self._compute_variance_of_points(points_to_sample)

    def compute_mean_of_points(self, points_to_sample):
        return self._compute_mean_of_points(points_to_sample)

    def compute_grad_mean_of_points(self, points_to_sample, num_derivatives=-1):
        return self._compute_grad_mean_of_points(points_to_sample, num_derivatives)

    def _compute_grad_variance_of_points_per_point(self, points_to_sample, var_of_grad):
        return self.__compute_grad_variance_of_points_per_point(points_to_sample, var_of_grad)

    def compute_cholesky_variance_of_points(self, points_to_sample):
        r"""Compute the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: cholesky factorization of the variance matrix of this GP, lower triangular
        :rtype: array of float64 with shape (num_to_sample, num_to_sample), lower triangle filled in

        """
        return scipy.linalg.cholesky(
            self.compute_variance_of_points(points_to_sample),
            lower=True,
            overwrite_a=True,
        )

    def __compute_grad_variance_of_points_per_point234(self, points_to_sample, var_of_grad):
        r"""Compute the gradient of the variance (matrix) of this GP at a single point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        See :meth:`~moe.optimal_learning.python.python_version.gaussian_process.GaussianProcess.compute_grad_variance_of_points` for more details.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param var_of_grad: index of ``points_to_sample`` to be differentiated against
        :type var_of_grad: int in {0, .. ``num_to_sample``-1}
        :return: grad_var: gradient of the variance matrix of this GP
        :rtype: array of float64 with shape (num_to_sample, num_to_sample, dim)

        """
        # TODO(GH-62): This can be improved/optimized. see: gpp_math.cpp, GaussianProcess::ComputeGradVarianceOfPoints
        num_to_sample = points_to_sample.shape[0]

        # Compute grad variance
        grad_var = numpy.zeros((num_to_sample, num_to_sample, self.dim))

        K_star = self._build_integrated_covariance_maxtrix2(
            self._covariance,
            self._points_sampled,
            points_to_sample,
        )

        _hyperparameters = self._covariance.get_hyperparameters()
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        covariance = SquareExponential(_hyperparameters[:-1])
        norm = numpy.pi /2.0
        K_inv_times_K_star = numpy.dot(self._K_C, K_star)
        for i, point_one in enumerate(points_to_sample):
            for j, point_two in enumerate(points_to_sample):
                if var_of_grad == i and var_of_grad == j:
                    tmp_point_one = point_one[:-1]
                    tmp_point_two = point_two[:-1]
                    grad_var[i, j, ...] = covariance.covariance(tmp_point_one, tmp_point_two) * self.grad_var(point_one, point_two, l, -1)  #the K_star_start part
                    for q, sampled_two in enumerate(self._points_sampled):
                        for p, sampled_one in enumerate(self._points_sampled):
                            tmp_sampled_two = sampled_two[:-1]
                            tmp_sampled_one = sampled_one[:-1]
                            grad_var[i, j, ...] -=  K_inv_times_K_star[q, i] * (sampled_two - point_one) * covariance.covariance(tmp_point_one, tmp_sampled_one) * norm
                            grad_var[i, j, ...] -=  K_inv_times_K_star[p, i] * (sampled_one - point_one) * covariance.covariance(tmp_point_one, tmp_sampled_two) * norm
                            grad_var[i,j,-1] = 0
                elif var_of_grad == i:
                    grad_var[i, j, ...] = covariance.covariance(tmp_point_one, tmp_point_two) * self.grad_var(point_one, point_two, l, -1)
                    for idx_two, sampled_two in enumerate(self._points_sampled):
                        grad_var[i, j, ...] -= K_inv_times_K_star[idx_two, j] * self._compute_integrated_grad_covariance2(sampled_two, point_one) * norm
                elif var_of_grad == j:
                    grad_var[i, j, ...] = covariance.covariance(tmp_point_two, tmp_point_one) * self.grad_var(point_two, point_one,l, -1)
                    for idx_one, sampled_one in enumerate(self._points_sampled):
                        grad_var[i, j, ...] -= K_inv_times_K_star[idx_one, i] * self._compute_integrated_grad_covariance2(sampled_two, point_two) * norm
        return grad_var

    def __compute_grad_variance_of_points_per_point(self, points_to_sample, var_of_grad):
        num_to_sample = points_to_sample.shape[0]

        # Compute grad variance
        grad_var = numpy.zeros((num_to_sample, num_to_sample, self.dim))
        _hyperparameters = self._covariance.get_hyperparameters()
        l = numpy.copy(_hyperparameters[-1]) # is it the last??
        covariance = SquareExponential(_hyperparameters[:-1])
        norm =numpy.pi /2.0

        for i, point_one in enumerate(points_to_sample):
            for j, point_two in enumerate(points_to_sample):
                if var_of_grad == i and var_of_grad == j:
                    tmp_point_one = point_one[:-1]
                    tmp_point_two = point_two[:-1]
                    grad_var[i, j, ...] = covariance.covariance(tmp_point_one, tmp_point_two) * self.grad_var(point_one, point_two, l, -1)  #the K_star_start part
                    for q, x_q in enumerate(self._points_sampled):
                        for p, x_p in enumerate(self._points_sampled):
                            tmp_x_q = x_q[:-1]
                            tmp_x_p = x_p[:-1]
                            grad_var[i, j, ...] -= norm * self._K_C[q, p] * covariance.covariance(tmp_point_one, tmp_x_p) * covariance.covariance(tmp_point_one, tmp_x_q) * (x_p + x_q - 2*point_one)
                            grad_var[i,j,-1] = 0
        return grad_var


    def grad_var(self, point_one, point_two, l, idx):
        diff_points = point_two - point_one
        diff_points[idx] = 0
        norm = 2.0 * numpy.sqrt(numpy.pi/2.0) /l
        norm2 = numpy.sqrt(2.0) * l
        norm3 = numpy.sqrt(2.0/numpy.pi) *l
        norm4 = 2.0 * l**2
        diff_bounds = self.high - self.low
        return diff_points * norm  * (diff_bounds * erf(diff_bounds/norm2) + norm3 * (numpy.exp(-(diff_bounds**2)/norm4) - 1))

    def compute_grad_variance_of_points(self, points_to_sample, num_derivatives=-1):
        r"""Compute the gradient of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        This function is similar to compute_grad_cholesky_variance_of_points() (below), except this does not include
        gradient terms from the cholesky factorization. Description will not be duplicated here.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_var: gradient of the variance matrix of this GP
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)
        grad_var = numpy.empty((num_derivatives, points_to_sample.shape[0], points_to_sample.shape[0], self.dim))
        for i in xrange(num_derivatives):
            grad_var[i, ...] = self._compute_grad_variance_of_points_per_point(points_to_sample, i)
        return grad_var

    def _compute_grad_cholesky_variance_of_points_per_point(self, points_to_sample, chol_var, var_of_grad):
        r"""Compute the gradient of the cholesky factorization of the variance (matrix) of this GP a single point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        See :meth:`~moe.optimal_learning.python.python_version.gaussian_process.GaussianProcess.compute_grad_cholesky_variance_of_points` for more details.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param chol_var: the cholesky factorization (L) of the variance matrix; only the lower triangle is accessed
        :type chol_var: array of float64 with shape (num_to_sample, num_to_sample)
        :param var_of_grad: index of ``points_to_sample`` to be differentiated against
        :type var_of_grad: int in {0, .. ``num_to_sample``-1}
        :return: grad_chol: gradient of the cholesky factorization of the variance matrix of this GP.
          ``grad_chol[j][i][d]`` is actually the gradients of ``var_{j,i}`` with
          respect to ``x_{k,d}``, the d-th dimension of the k-th entry of ``points_to_sample``, where
          k = ``var_of_grad``
        :rtype: array of float64 with shape (num_to_sample, num_to_sample, dim)

        """
        # TODO(GH-62): This can be improved/optimized. see: gpp_math.cpp, GaussianProcess::ComputeGradCholeskyVarianceOfPoints
        num_to_sample = points_to_sample.shape[0]

        # Compute grad cholesky
        # Zero out the upper half of the matrix
        grad_chol = self._compute_grad_variance_of_points_per_point(points_to_sample, var_of_grad)
        for i in xrange(num_to_sample):
            for j in xrange(num_to_sample):
                if i < j:
                    grad_chol[i, j, ...] = numpy.zeros(self.dim)

        # Step 2 of Appendix 2
        for k in xrange(num_to_sample):
            L_kk = chol_var[k, k]
            if L_kk > MINIMUM_STD_DEV_GRAD_CHOLESKY:
                grad_chol[k, k, ...] *= 0.5 / L_kk
                for j in xrange(k + 1, num_to_sample):
                    grad_chol[j, k, ...] = (grad_chol[j, k, ...] - chol_var[j, k] * grad_chol[k, k, ...]) / L_kk
                for j in xrange(k + 1, num_to_sample):
                    for i in xrange(j, num_to_sample):
                        grad_chol[i, j, ...] += -grad_chol[i, k, ...] * chol_var[j, k] - chol_var[i, k] * grad_chol[j, k, ...]

        return grad_chol

    def compute_grad_cholesky_variance_of_points(self, points_to_sample, chol_var=None, num_derivatives=-1):
        r"""Compute the gradient of the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        .. Warning:: ``points_to_sample`` should not contain duplicate points.

        This function accounts for the effect on the gradient resulting from
        cholesky-factoring the variance matrix.  See Smith 1995 for algorithm details.

        Observe that ``grad_chol`` is nominally sized:
        ``grad_chol[num_to_sample][num_to_sample][num_to_sample][dim]``.
        Let this be indexed ``grad_chol[k][j][i][d]``, which is read the derivative of ``var[j][i]``
        with respect to ``x_{k,d}`` (x = ``points_to_sample``)


        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_cholesky_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param chol_var: the cholesky factorization (L) of the variance matrix; only the lower triangle is accessed
        :type chol_var: array of float64 with shape (num_to_sample, num_to_sample)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_chol: gradient of the cholesky factorization of the variance matrix of this GP.
          ``grad_chol[k][j][i][d]`` is actually the gradients of ``var_{j,i}`` with
          respect to ``x_{k,d}``, the d-th dimension of the k-th entry of ``points_to_sample``, where
          k = ``var_of_grad``
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)

        # Users can pass this in directly b/c it has often been computed already.
        if chol_var is None:
            var_star = self.compute_variance_of_points(points_to_sample)
            # Note: only access the lower triangle of chol_var; upper triangle is garbage
            # cho_factor returns a tuple, (factorized_matrix, lower_tri_flag); grab the matrix
            chol_var = scipy.linalg.cho_factor(var_star, lower=True, overwrite_a=True)[0]

        grad_chol_decomp = numpy.empty((num_derivatives, points_to_sample.shape[0], points_to_sample.shape[0], self.dim))
        for i in xrange(num_derivatives):
            grad_chol_decomp[i, ...] = self._compute_grad_cholesky_variance_of_points_per_point(points_to_sample, chol_var, i)
        return grad_chol_decomp

    def add_sampled_points(self, sampled_points, validate=False):
        r"""Add sampled point(s) (point, value, noise) to the GP's prior data.

        Also forces recomputation of all derived quantities for GP to remain consistent.

        :param sampled_points: SamplePoint objects to load into the GP (containing point, function value, and noise variance)
        :type sampled_points: list of :class:`moe.optimal_learning.python.SamplePoint` objects
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        # TODO(GH-192): Insert the new covariance (and cholesky covariance) rows into the current matrix  (O(N^2))
        # instead of recomputing everything (O(N^3))
        self._historical_data.append_sample_points(sampled_points, validate=validate)
        self._build_precomputed_data()

    def sample_point_from_gp(self, point_to_sample, noise_variance=0.0):
        r"""Sample a function value from a Gaussian Process prior, provided a point at which to sample.

        Uses the formula ``function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2``, where ``w1, w2``
        are draws from N(0,1).

        .. NOTE::
             Set noise_variance to 0 if you want "accurate" draws from the GP.
             BUT if the drawn (point, value) pair is meant to be added back into the GP (e.g., for testing), then this point
             MUST be drawn with noise_variance equal to the noise associated with "point" as a member of "points_sampled"

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.sample_point_from_gp`

        :param point_to_sample: point (in dim dimensions) at which to sample from this GP
        :type points_to_sample: array of float64 with shape (dim)
        :param noise_variance: amount of noise to associate with the sample
        :type noise_variance: float64 >= 0.0
        :return: sample_value: function value drawn from this GP
        :rtype: float64

        """
        point = numpy.array(point_to_sample, copy=False, ndmin=2)
        mean = self.compute_mean_of_points(point)[0]
        variance = self.compute_variance_of_points(point)[0, 0]

        return mean + numpy.sqrt(variance) * numpy.random.normal() + numpy.sqrt(noise_variance) * numpy.random.normal()
