# -*- coding: utf-8 -*-

"""
Datasets submodule for k-diagram, including data generation tools
and loading APIs.
"""
from .make import ( 
    make_uncertainty_data,
    make_taylor_data,
    make_multi_model_quantile_data,
    make_fingerprint_data 
    )
from .load import load_synthetic_uncertainty_data

__all__ = [
    'make_uncertainty_data',
    'load_synthetic_uncertainty_data',
    'make_taylor_data',
    'make_multi_model_quantile_data',
    'make_fingerprint_data' 
    ]