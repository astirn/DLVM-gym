SEED=4282023

# toy experiments
python experiments_toy.py --seed $SEED --dataset "isotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "anisotropic_gmm"
python experiments_toy.py --seed $SEED --dataset "moons"

# clustering experiments
python experiments_clustering.py --seed $SEED --dataset "MNIST"
python experiments_clustering.py --seed $SEED --dataset "FashionMNIST"

# scRNA-seq experiments
python experiments_scRNA.py --seed $SEED --dataset "cortex"
python experiments_scRNA.py --seed $SEED --dataset "lung atlas"
python experiments_scRNA.py --seed $SEED --dataset "pbmc"
python experiments_scRNA.py --seed $SEED --dataset "retina"
python experiments_scRNA.py --seed $SEED --dataset "split-seq"
