# -*- coding: utf-8 -*-
#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""
Datasets submodule for k-diagram, including data generation tools
and loading APIs.
"""
from .make import ( 
    make_uncertainty_data,
    make_taylor_data,
    make_multi_model_quantile_data,
    make_fingerprint_data, 
    make_cyclical_data 
    )

from .load import (
    # load_synthetic_uncertainty_data, #  renamed to load_uncertainty_data
    load_uncertainty_data, 
    load_zhongshan_subsidence
    )
__all__ = [
    'make_uncertainty_data',
    'load_uncertainty_data',
    'make_taylor_data',
    'make_multi_model_quantile_data',
    'make_fingerprint_data', 
    'make_cyclical_data', 
    'load_zhongshan_subsidence'
    ]
