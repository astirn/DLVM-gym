DEFAULT_N_JOBS=1
SCRIPT_NAME="$(basename "$0" .sh)"

help() {
    local description="$1"
    cat <<EOF
Usage: $0 <dataset> [options]

${description}

Arguments:
  dataset   Dataset config name (e.g. mnist, fashion_mnist)

Options:
  -n, --num_jobs N   Number of parallel Hydra jobs (default: ${DEFAULT_N_JOBS})
  --debug            Disable wandb logging (default: wandb online)

Examples:
  $0 mnist
  $0 fashion_mnist -n 10
  $0 mnist --debug
EOF
}

parse_args() {
    local description="$1"
    shift

    DEBUG=false
    DATASET=""
    N_JOBS="${DEFAULT_N_JOBS}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                help "$description"
                exit 0
                ;;
            -n|--num_jobs)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: $1 requires a value" >&2
                    help "$description" >&2
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
                help "$description" >&2
                exit 1
                ;;
            *)
                if [[ -z "$DATASET" ]]; then
                    DATASET="$1"
                else
                    echo "Unknown argument: $1" >&2
                    help "$description" >&2
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$DATASET" ]]; then
        help "$description" >&2
        exit 1
    fi

    if [[ "$DEBUG" == true ]]; then
        WANDB_MODE=disabled
    else
        WANDB_MODE=online
    fi
}
