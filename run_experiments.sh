SEED=42823  # a very special day <3

export CUDA_LAUNCH_BLOCKING=1

# toy experiments
python experiments_toy.py --seed $SEED --dataset "isotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "anisotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "moons"

# clustering experiments
python experiments_clustering.py --num_trials 5 --seed $SEED --dataset "MNIST"

# scRNA-seq experiments
python experiments_scRNA.py --num_trials 5 --seed $SEED --dataset "cortex"
python experiments_scRNA.py --num_trials 5 --seed $SEED --dataset "lung atlas"
python experiments_scRNA.py --num_trials 5 --seed $SEED --dataset "pbmc"
python experiments_scRNA.py --num_trials 5 --seed $SEED --dataset "retina"
python experiments_scRNA.py --num_trials 5 --seed $SEED --dataset "split-seq"

# merge scRNA-seq tables
for table in "integration_results"; do
    python merge_tables.py --output "results/scRNA/${SEED}/${table}.tex" --tables \
        "results/scRNA/${SEED}/cortex/${table}.pkl" \
        "results/scRNA/${SEED}/lung atlas/${table}.pkl" \
        "results/scRNA/${SEED}/pbmc/${table}.pkl" \
        "results/scRNA/${SEED}/retina/${table}.pkl" \
        "results/scRNA/${SEED}/split-seq/${table}.pkl"
done
