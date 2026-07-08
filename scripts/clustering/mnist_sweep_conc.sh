PRIOR="crypt/dp_vampprior,crypt/dp_vampprior_unchained"
CONCENTRATION="0.1,0.5,1.0,2.0,5.0"
KL_WARMUP_STEPS="0,1000,2500,5000,10000"
U_INITIAL_STRATEGY="data,stratified_data,random"
SEED="1,2,3,4,5"

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/mnist --multirun \
    hydra.launcher.n_jobs=5 \
    priors@model.prior_params=$PRIOR \
    model.prior_params.concentration=$CONCENTRATION \
    model.u_initial_strategy=$U_INITIAL_STRATEGY \
    model.kl_warmup_steps=$KL_WARMUP_STEPS \
    loggers.wandb.name='${model.prior_params.name}-${model.prior_params.concentration}-${model.u_initial_strategy}-${model.kl_warmup_steps}' \
    loggers.wandb.mode=online \
    loggers.wandb.job_type=sweep \
    seed=$SEED
