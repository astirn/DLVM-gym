#!/usr/bin/env bash

PRIOR="crypt/vampprior,crypt/vampprior_unleashed,crypt/dp_vampprior,crypt/dp_vampprior_unleashed"
PYTHONPATH=. uv run python src/experiments/toy.py -cn experiments/toy --multirun \
    paths.log_dir=logs/Toy \
    priors@model.prior_params=$PRIOR
