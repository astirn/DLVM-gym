#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "Differential Expression" "$@"

ENCODE_COVARIATES="true"
SEED="1"

# Standard normal prior
PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=$ENCODE_COVARIATES \
    model.kl_warmup_epochs='${trainer.max_epochs}' \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params="standard_normal" \
    loggers.wandb.mode=disabled \
    seed=$SEED

# DP VampPrior Unleashed
PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=$ENCODE_COVARIATES \
    model.kl_warmup_epochs=0 \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params="crypt/dp_vampprior_unleashed" \
    loggers.wandb.mode=disabled \
    seed=$SEED
