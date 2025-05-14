import seaborn as sns

from cytoolz import interleave, pipe
from cytoolz.curried import partition

COLOR_PALETTE = pipe(
    sns.color_palette('tab20c', n_colors=20) + sns.color_palette('tab20b', n_colors=20),
    partition(4),
    interleave,
    list
)

TABLE_COLUMNS = {
    'decoupled_variance': 'Decoupled',
    'wishart_prior': 'Wishart',
    'dirichlet_prior': 'Dirichlet',
    'dirichlet_process': 'DP',
    'dirichlet_process_variational_family': 'DP VF',

    'log_marginal': 'log p(x)',
    'posterior_predictive': 'log p(x|x)',

    'acc': 'Accuracy',
    'nmi': 'NMI',
    'clusters': 'Clusters',
}

TEST_CASES = [
    dict(),

    dict(decoupled_variance=True),
    dict(wishart_prior=True),

    dict(dirichlet_prior=True),
    dict(dirichlet_prior=True, decoupled_variance=True),
    dict(dirichlet_prior=True, wishart_prior=True),

    dict(dirichlet_process=True, dirichlet_process_variational_family=False),
    dict(dirichlet_process=True, dirichlet_process_variational_family=False, decoupled_variance=True),
    dict(dirichlet_process=True, dirichlet_process_variational_family=False, wishart_prior=True),

    dict(dirichlet_process=True, dirichlet_process_variational_family=True),
    dict(dirichlet_process=True, dirichlet_process_variational_family=True, decoupled_variance=True),
    dict(dirichlet_process=True, dirichlet_process_variational_family=True, wishart_prior=True),
]
