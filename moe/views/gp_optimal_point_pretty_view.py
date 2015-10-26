# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views for ``gp_next_points_*`` endpoints; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

Include:

    1. Class that extends :class:`moe.views.optimizable_gp_pretty_view.GpPrettyView` for next_points optimizers

"""
import numpy

import moe.optimal_learning.python.cpp_wrappers.expected_improvement
from moe.optimal_learning.python.python_version.optimizable_gaussian_process import OptimizableGaussianProcess
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
import moe.optimal_learning.python.python_version.optimization as python_optimization
from moe.optimal_learning.python.timing import timing_context
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.optimizable_gp_pretty_view import OptimizableGpPrettyView
from moe.views.schemas.gp_optimal_point_pretty_view import GpOptimalPointRequest, GpOptimalPointResponse
from moe.views.utils import _make_gp_from_params, _make_domain_from_params, _make_optimizer_parameters_from_params


EPI_OPTIMIZATION_TIMING_LABEL = 'EPI optimization time'


class GpOptimalPointPrettyView(OptimizableGpPrettyView):

    """A class to encapsulate 'pretty' ``gp_next_points_*`` views; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    Extends :class:`moe.views.optimizable_gp_pretty_view.GpPrettyView` with:

        1. gaussian_process generation from params
        2. Converting params into a C++ consumable set of optimizer parameters
        3. A method (compute_next_points_to_sample_response) for computing the next best points to sample from a gaussian_process

    """

    request_schema = GpOptimalPointRequest()
    response_schema = GpOptimalPointResponse()

    _pretty_default_request = {
            "gp_historical_info": GpPrettyView._pretty_default_gp_historical_info,
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {
                        "min": 0.0,
                        "max": 1.0,
                    },
                    ],
                },
            }

    def compute_optimal_point_response(self, params, optimizer_method_name, route_name, *args, **kwargs):
        """Compute optimal point by performing gradient descent on the gp mean
        """
        num_to_sample = 1
        max_num_threads = params.get('max_num_threads')

        gaussian_process = _make_gp_from_params(params)

        ei_opt_status = {}

        # Calculate the next best points to sample given the historical data

        optimizer_class, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

        domain = RepeatedDomain(num_to_sample, _make_domain_from_params(params, python_version=True))
        optimizable_gp_evaluator = OptimizableGaussianProcess(gaussian_process)
        opt_method = getattr(moe.optimal_learning.python.python_version.optimizable_gaussian_process, optimizer_method_name)

        optimizable_gp_optimizer = optimizer_class(
                domain,
                optimizable_gp_evaluator,
                optimizer_parameters,
                num_random_samples=num_random_samples,
                )

        with timing_context(EPI_OPTIMIZATION_TIMING_LABEL):
            optimal_point = opt_method(
                optimizable_gp_optimizer,
                params.get('optimizer_info')['num_multistarts'],  # optimizer_parameters.num_multistarts,
                *args,
                **kwargs
            )

        mean_value = optimizable_gp_evaluator.compute_objective_function()
        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': optimal_point,
                'status': {
                    'function_value': mean_value[0],
                    'optimizer_success': ei_opt_status,
                    },
                })
