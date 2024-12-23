SEED=42823  # a very special day <3

export CUDA_LAUNCH_BLOCKING=1

# toy experiments
python experiments_toy.py --seed $SEED --dataset "isotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "anisotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "moons"

# clustering experiments
python experiments_clustering.py --num_trials $NUM_TRIALS --seed $SEED --dataset "MNIST"
