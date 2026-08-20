#!/usr/bin/env bash

PRIOR="crypt/vampprior,crypt/vampprior_unchained,crypt/dp_vampprior,crypt/dp_vampprior_unchained"
PYTHONPATH=. uv run python src/experiments/toy.py -cn experiments/toy --multirun \
    paths.log_dir='logs/'"$(basename "$0" .sh)" \
    priors@model.prior_params=$PRIOR
