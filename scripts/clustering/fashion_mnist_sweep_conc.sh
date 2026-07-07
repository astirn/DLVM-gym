CONCENTRATION="0.01,0.1,1.0,2.0,5.0"
KL_WARMUP_STEPS="0,1000,2500,5000,10000"
U_INITIAL_STRATEGY="data,stratified_data,random"
SEED="1,2,3,4,5"

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/fashion_mnist --multirun \
    model.prior_params.concentration=$CONCENTRATION \
    model.u_initial_strategy=$U_INITIAL_STRATEGY \
    model.kl_warmup_steps=$KL_WARMUP_STEPS \
    loggers.wandb.mode=online \
    loggers.wandb.job_type=sweep \
    seed=$SEED
