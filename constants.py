import seaborn as sns


COLOR_PALETTE = (
        sns.color_palette('tab20', n_colors=20)[0::2] +
        sns.color_palette('tab20', n_colors=20)[1::2]
)

TABLE_COLUMNS = {
    'decoupled_variance': 'Decoupled',
    'wishart_prior': 'Wishart',
    'categorical_prior': 'Categorical',
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

    dict(dirichlet_process=True, dirichlet_process_variational_family=True),
    dict(dirichlet_process=True, dirichlet_process_variational_family=True, decoupled_variance=True),
    dict(dirichlet_process=True, dirichlet_process_variational_family=True, wishart_prior=True),
]
