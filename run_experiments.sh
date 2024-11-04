SEED=42823  # a very special day <3

export CUDA_LAUNCH_BLOCKING=1

# toy experiments
python experiments_toy.py --seed $SEED --dataset "isotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "anisotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "moons"

# clustering experiments
python experiments_clustering.py --seed $SEED --dataset "MNIST" --num_trials 5

# scRNA-seq experiments
python experiments_scRNA.py --seed $SEED --dataset "cortex"
python experiments_scRNA.py --seed $SEED --dataset "lung atlas"
python experiments_scRNA.py --seed $SEED --dataset "pbmc"
python experiments_scRNA.py --seed $SEED --dataset "retina"
python experiments_scRNA.py --seed $SEED --dataset "split-seq"

# merge scRNA-seq tables
python merge_tables.py --output "results/scRNA/${SEED}/integration_results.tex" --tables \
    "results/scRNA/${SEED}/cortex/integration_results.pkl" \
    "results/scRNA/${SEED}/lung atlas/integration_results.pkl" \
    "results/scRNA/${SEED}/pbmc/integration_results.pkl" \
    "results/scRNA/${SEED}/retina/integration_results.pkl" \
    "results/scRNA/${SEED}/split-seq/integration_results.pkl"