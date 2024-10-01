# toy experiments
python experiments_toy.py --dataset "isotropic_gmm"
python experiments_toy.py --dataset "anisotropic_gmm"
python experiments_toy.py --dataset "moons"

# clustering experiments
python experiments_clustering.py --dataset "MNIST"
python experiments_clustering.py --dataset "FashionMNIST"

# scRNA-seq experiments
python experiments_scRNA.py --dataset "cortex"
python experiments_scRNA.py --dataset "pbmc"
python experiments_scRNA.py --dataset "lung atlas"
python experiments_scRNA.py --dataset "retina"
python experiments_scRNA.py --dataset "split-seq"
