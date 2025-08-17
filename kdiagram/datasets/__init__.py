#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""
Datasets submodule for k-diagram, including data generation tools
and loading APIs.
"""
from .load import (
    load_uncertainty_data,
    load_zhongshan_subsidence,
)
from .make import (
    make_cyclical_data,
    make_fingerprint_data,
    make_multi_model_quantile_data,
    make_taylor_data,
    make_uncertainty_data,
)

__all__ = [
    "make_uncertainty_data",
    "load_uncertainty_data",
    "make_taylor_data",
    "make_multi_model_quantile_data",
    "make_fingerprint_data",
    "make_cyclical_data",
    "load_zhongshan_subsidence",
]
