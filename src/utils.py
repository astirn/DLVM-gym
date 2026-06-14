import torch
import torch.distributions as dist


def concat_distributions(distributions: list[dist.Distribution]) -> dist.Distribution:
    if not distributions:
        raise ValueError("Cannot concatenate empty list of distributions")

    first = distributions[0]

    if isinstance(first, dist.Categorical):
        categoricals: list[dist.Categorical] = [
            d for d in distributions if isinstance(d, dist.Categorical)
        ]
        if len(categoricals) != len(distributions):
            raise ValueError("All distributions must be Categorical")
        return dist.Categorical(logits=torch.concat([d.logits for d in categoricals]))

    if isinstance(first, dist.Independent):
        independents: list[dist.Independent] = [
            d for d in distributions if isinstance(d, dist.Independent)
        ]
        if len(independents) != len(distributions):
            raise ValueError("All distributions must be Independent")
        reinterpreted_batch_ndims = independents[0].reinterpreted_batch_ndims
        return dist.Independent(
            concat_distributions([d.base_dist for d in independents]),
            reinterpreted_batch_ndims,
        )

    if isinstance(first, dist.Normal):
        normals: list[dist.Normal] = [
            d for d in distributions if isinstance(d, dist.Normal)
        ]
        if len(normals) != len(distributions):
            raise ValueError("All distributions must be Normal")
        return dist.Normal(
            loc=torch.concat([d.loc for d in normals]),
            scale=torch.concat([d.scale for d in normals]),
        )

    if isinstance(first, dist.MultivariateNormal):
        multivariate_normals: list[dist.MultivariateNormal] = [
            d for d in distributions if isinstance(d, dist.MultivariateNormal)
        ]
        if len(multivariate_normals) != len(distributions):
            raise ValueError("All distributions must be MultivariateNormal")
        return dist.MultivariateNormal(
            loc=torch.concat([d.loc for d in multivariate_normals]),
            covariance_matrix=torch.concat(
                [d.covariance_matrix for d in multivariate_normals]
            ),
        )

    raise ValueError(f"Unsupported distribution type: {type(first)}")
