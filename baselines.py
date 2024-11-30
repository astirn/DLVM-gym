import numpy as np

from anndata import AnnData
from experiments_scRNA import load_dataset
from scanpy.pp import combat, pca
from scanpy.external.pp import harmony_integrate, mnn_correct
from scib_metrics.benchmark import Benchmarker


def additional_baselines(data: AnnData, num_pcs: int) -> AnnData:
    data.X = np.asarray(np.log1p(data.X.todense()))
    data.obs['batch'] = data.obs['batch'].astype(str)
    pca(data, n_comps=num_pcs)

    harmony_integrate(data, adjusted_basis='Harmony', key='batch')

    return data


if __name__ == '__main__':
    dataset = 'lung atlas'
    num_pcs = 30

    anndata = load_dataset(dataset)
    anndata = additional_baselines(anndata, num_pcs)
    bm = Benchmarker(
        anndata,
        batch_key='batch',
        label_key='cell_type',
        embedding_obsm_keys=['Harmony'],
        n_jobs=-1,
    )
    bm.benchmark()
    df = bm.get_results(min_max_scale=False).copy()

    print(1)