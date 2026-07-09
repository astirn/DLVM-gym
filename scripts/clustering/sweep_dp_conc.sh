help() {
    cat <<EOF
Usage: $0 <dataset> [options]

Sweep DP concentration hyperparameters for a clustering dataset.

Arguments:
  dataset   Dataset config name (e.g. mnist, fashion_mnist)

Options:
  -n, --num_jobs N   Number of parallel Hydra jobs (default: 1)
  --debug            Disable wandb logging (default: wandb online)

Examples:
  $0 mnist
  $0 fashion_mnist -n 10
  $0 mnist --debug
EOF
}

DEBUG=false
DATASET=""
N_JOBS="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            help
            exit 0
            ;;
        -n|--num_jobs)
            if [[ -z "${2:-}" ]]; then
                echo "Error: $1 requires a value" >&2
                help >&2
                exit 1
            fi
            N_JOBS="$2"
            shift 2
            ;;
        --num_jobs=*)
            N_JOBS="${1#*=}"
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            help >&2
            exit 1
            ;;
        *)
            if [[ -z "$DATASET" ]]; then
                DATASET="$1"
            else
                echo "Unknown argument: $1" >&2
                help >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    help >&2
    exit 1
fi

if [[ "$DEBUG" == true ]]; then
    WANDB_MODE=disabled
else
    WANDB_MODE=online
fi

PRIOR="crypt/dp_vampprior,crypt/dp_vampprior_unchained"
CONCENTRATION="0.1,0.5,1.0,2.0,5.0"
KL_WARMUP_STEPS="0,1000,2500,5000,10000"
U_INITIAL_STRATEGY="data,stratified_data,random"
SEED="1,2,3,4,5"

PYTHONPATH=. uv run python src/experiments/clustering.py -cn experiments/clustering/${DATASET} --multirun \
    hydra.launcher.n_jobs=$N_JOBS \
    priors@model.prior_params=$PRIOR \
    model.prior_params.concentration=$CONCENTRATION \
    model.u_initial_strategy=$U_INITIAL_STRATEGY \
    model.kl_warmup_steps=$KL_WARMUP_STEPS \
    loggers.wandb.name='${model.prior_params.name}-${model.prior_params.concentration}-${model.u_initial_strategy}-${model.kl_warmup_steps}' \
    loggers.wandb.mode=$WANDB_MODE \
    loggers.wandb.job_type=sweep \
    seed=$SEED
