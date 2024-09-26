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
    dict(decoupled_variance=False, dirichlet_prior=True),
    dict(decoupled_variance=True, dirichlet_prior=True),
    dict(decoupled_variance=False, dirichlet_process=True),
    dict(decoupled_variance=True, dirichlet_process=True),
]
