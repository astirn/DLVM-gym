CONCENTRATION=0.5,0.75,1.0,1.25,1.5
STRATIFIED_U_INITIAL=true,false
SEED=1,2,3,4,5

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/mnist --multirun \
    model.prior_params.concentration=$CONCENTRATION \
    model.stratified_u_initial=$STRATIFIED_U_INITIAL \
    loggers.wandb.mode=enabled \
    loggers.wandb.job_type=sweep \
    loggers.wandb.tags=["v1"] \
    seed=$SEED
