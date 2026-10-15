#!/usr/bin/env bash
PRIORS="standard_normal,crypt/dp_vampprior_unleashed"
SEED="1"

PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/differential_expression --multirun\
    hydra.launcher.n_jobs=1 \
    priors@model.prior_params=$PRIORS \
    loggers.wandb.mode=disabled \
    seed=$SEED
