import seaborn as sns

from cytoolz import interleave, pipe
from cytoolz.curried import partition

COLOR_PALETTE = pipe(
    sns.color_palette('tab20c', n_colors=20) + sns.color_palette('tab20b', n_colors=20),
    partition(4),
    interleave,
    list
)

TEST_CASES = [
    dict(decoupled_variance=False),
    dict(decoupled_variance=True),
    dict(dirichlet_prior=True, decoupled_variance=False),
    dict(dirichlet_prior=True, decoupled_variance=True),
    dict(dirichlet_process=True, decoupled_variance=False),
    dict(dirichlet_process=True, decoupled_variance=True),
]
