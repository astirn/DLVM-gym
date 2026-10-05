#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../arg_parser.sh
source "${SCRIPT_DIR}/../arg_parser.sh"

parse_args "OpenProblems baseline experiments." "$@"

SEED="1,2,3"

# Standard normal prior
PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates="false,true" \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params="standard_normal" \
    loggers.wandb.name='${model.prior_params.name}-${model.encode_covariates}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v2"] \
    seed=$SEED

# VMM (Stirn & Knowles, 2025)
PYTHONPATH=. uv run python src/experiments/scRNA.py -cn experiments/scRNA/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    model.encode_covariates=true \
    paths.log_dir='logs/${experiment_name}/'"$SCRIPT_NAME" \
    priors@model.prior_params=crypt/vmm \
    model.prior_params.num_data='${data.batch_size.train}' \
    model.kl_warmup_epochs=0 \
    model.u_initial_strategy=data \
    loggers.wandb.name='${model.prior_params.name}-${model.prior_params.num_data}' \
    loggers.wandb.group='${experiment_name}/'"$SCRIPT_NAME" \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=baseline \
    loggers.wandb.tags=["v2"] \
    seed=$SEED

