# -*- coding: utf-8 -*-
"""Base request/response schemas for ``gp_next_points_*`` endpoints; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS, MAX_ALLOWED_NUM_THREADS, DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS
from moe.views.schemas import base_schemas


class GpOptimalPointRequest(base_schemas.StrictMappingSchema):

    """A request colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    """



    gp_historical_info = base_schemas.GpHistoricalInfo()
    domain_info = base_schemas.BoundedDomainInfo()
    covariance_info = base_schemas.CovarianceInfo(
            missing=base_schemas.CovarianceInfo().deserialize({}),
            )
    optimizer_info = base_schemas.OptimizerInfo(
            missing=base_schemas.OptimizerInfo().deserialize({}),
            )
    gp_integral_info = base_schemas.GpIntegralInfo()


class GpOptimalPointStatus(base_schemas.StrictMappingSchema):

    """A status schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar expected_improvement: (*float64 >= 0.0*) EI evaluated at ``points_to_sample`` (:class:`moe.views.schemas.base_schemas.ListOfExpectedImprovements`)
    :ivar optimizer_success: (*dict*) Whether or not the optimizer converged to an optimal set of ``points_to_sample``

    """

    function_value = colander.SchemaNode(
            colander.Float(),
            )
    optimizer_success = colander.SchemaNode(
        colander.Mapping(unknown='preserve'),
        default={'found_update': False},
    )


class GpOptimalPointResponse(base_schemas.StrictMappingSchema):

    """A response colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar points_to_sample: (*list of list of float64*) points in the domain to sample next (:class:`moe.views.schemas.base_schemas.ListOfPointsInDomain`)
    :ivar status: (:class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsStatus`) dict indicating final EI value and
      optimization status messages (e.g., success)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint": "gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "status": {
                "expected_improvement": "0.443478498868",
                "optimizer_success": {
                    'gradient_descent_tensor_product_domain_found_update': True,
                    },
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = base_schemas.ListOfPointsInDomain()
    status = GpOptimalPointStatus()
