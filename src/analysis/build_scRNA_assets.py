import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from toolz import keyfilter

from .constants import ASSETS, PRIOR_NAME_MAP
from .utils import get_relevant_runs

DATASETS = {
    "op_dkd": "DKD",
    "op_immune_cell_atlas": "Immune Cell Atlas",
    "op_gtex_v9": "GTEx",
    "op_mouse_pancreas_atlas": "Mouse Pancreas Atlas",
    "op_hypomap": "HypoMap",
    "op_tabula_sapiens": "Tabula Sapiens",
}
PRIOR_ORDER = [
    "VampPrior",
    "DPVampPriorUnchained",
    "VMM",
]

MODEL_METRICS = {
    "Log Marginal": "greater",
    "NMI": "greater",
}

BATCH_CORRECTION_METRICS = {
    "BRAS": "greater",
    "Graph connectivity": "greater",
    "KBET": "greater",
    # "PCR comparison": "greater",
    "iLISI": "greater",
}

BIO_CONSERVATION_METRICS = {
    "Isolated labels": "greater",
    "KMeans ARI": "greater",
    "KMeans NMI": "greater",
    "Silhouette label": "greater",
    "cLISI": "greater",
}


@dataclass
class Parameter:
    display_name: str
    retrieval_fn: Callable[[dict[str, Any]], Any]
    filter_key: str | None = None
    filter_values: list[Any] | None = None


def _get_scRNA_results(
    experiment: str,
    tag: str,
    parameters: Parameter | list[Parameter],
) -> None:
    filter_list: list[dict[str, Any]] = [
        {"group": {"$eq": experiment}},
        {"tags": {"$eq": tag}},
    ]
    if isinstance(parameters, Parameter):
        parameters = [parameters]
    for parameter in parameters:
        if parameter.filter_key is not None and parameter.filter_values is not None:
            filter_list.append(
                {f"config.{parameter.filter_key}": {"$in": parameter.filter_values}}
            )
    filters = {
        "$and": filter_list,
    }
    df = pd.DataFrame()
    for config, summary in get_relevant_runs(filters):
        if "test/log_marginal" not in summary:
            continue
        index = {
            "Prior": config["model"]["prior_params"]["name"],
            "Init.": config["model"]["u_initial_strategy"],
            "seed": config["seed"],
        }
        for parameter in parameters:
            index[parameter.display_name] = parameter.retrieval_fn(config)
        metrics = {
            "Log Marginal": summary["test/log_marginal"],
            "NMI": summary["last/nmi"],
        }
        for ckpt in ["best_val_loss", "best_val_nmi", "last"]:
            metrics.update(
                **{
                    "Checkpoint": ckpt,
                },
                **keyfilter(lambda k: f"{ckpt}/Batch correction" in k, summary),
                **keyfilter(lambda k: f"{ckpt}/Bio conservation" in k, summary),
            )
            new_row = pd.DataFrame({**index, **metrics}, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)

    save_path = ASSETS / f"{experiment}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_path)


def cache_results():
    for dataset in DATASETS.keys():
        _get_scRNA_results(
            f"scRNA/{dataset}/op_baselines",
            tag="v2",
            parameters=[
                Parameter(
                    display_name="KL Warmup",
                    retrieval_fn=lambda c: c["model"]["kl_warmup_epochs"],
                    filter_key="model.kl_warmup_epochs",
                ),
                Parameter(
                    display_name="Encode covariates",
                    retrieval_fn=lambda c: c["model"]["encode_covariates"],
                    filter_key="model.encode_covariates",
                ),
            ],
        )
        _get_scRNA_results(
            f"scRNA/{dataset}/sweep_dp_conc",
            tag="v1",
            parameters=[
                Parameter(
                    display_name="$\\alpha$",
                    retrieval_fn=lambda c: c["model"]["prior_params"]["concentration"],
                    filter_key="model.prior_params.concentration",
                ),
                Parameter(
                    display_name="KL Warmup",
                    retrieval_fn=lambda c: c["model"]["kl_warmup_epochs"],
                    filter_key="model.kl_warmup_epochs",
                ),
                Parameter(
                    display_name="Encode covariates",
                    retrieval_fn=lambda c: c["model"]["encode_covariates"],
                    filter_key="model.encode_covariates",
                ),
            ],
        )


def _summary_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["Aggregate/Batch correction"] = df[
        [f"Batch correction/{m}" for m in BATCH_CORRECTION_METRICS.keys()]
    ].mean(axis=1)
    df["Aggregate/Bio conservation"] = df[
        [f"Bio conservation/{m}" for m in BIO_CONSERVATION_METRICS.keys()]
    ].mean(axis=1)
    df["Aggregate/Total"] = (
        0.4 * df["Aggregate/Batch correction"] + 0.6 * df["Aggregate/Bio conservation"]
    )
    return df


PRECISION = 3


def _fmt_score(mean: float, std: float) -> str:
    return f"{mean:.{PRECISION}f} $\\pm$ {std:.{PRECISION-1}f}"


GROUP_ORDER = ["Our metrics", "Bio conservation", "Batch correction", "Aggregate"]


def _insert_dataset_midrules(latex: str, dataset_names: list[str]) -> str:
    # drop pandas' auto \cline rules from sparsifying the multirow index
    lines = [
        line
        for line in latex.splitlines()
        if not re.fullmatch(r"(\\cline\{\d+-\d+\}\s*)+", line.strip())
    ]
    latex = "\n".join(lines)

    # extend the column_format's vertical rules into the group-header row, and center the labels
    latex = re.sub(r"\\multicolumn\{(\d+)\}\{r\}", r"\\multicolumn{\1}{|c}", latex)

    # only the first sub-row of each dataset block renders the \multirow{...}{Dataset}
    for name in dataset_names[1:]:
        pattern = re.compile(
            rf"\n(\\multirow(?:\[[a-z]\])?\{{\d+\}}\{{\*\}}\{{{re.escape(name)}\}})"
        )
        latex = pattern.sub(r"\n\\midrule\n\1", latex, count=1)
    return latex


def build_table():

    common_index = [
        "Dataset",
        "Prior",
        "Encode covariates",
        "KL Warmup",
    ]
    metric_cols = (
        list(MODEL_METRICS.keys())
        + [f"last/Bio conservation/{m}" for m in BIO_CONSERVATION_METRICS.keys()]
        + [f"last/Batch correction/{m}" for m in BATCH_CORRECTION_METRICS.keys()]
    )
    rename_cols = {
        **{m: f"Our metrics/{m}" for m in MODEL_METRICS.keys()},
        **{
            f"last/Bio conservation/{m}": f"Bio conservation/{m}"
            for m in BIO_CONSERVATION_METRICS.keys()
        },
        **{
            f"last/Batch correction/{m}": f"Batch correction/{m}"
            for m in BATCH_CORRECTION_METRICS.keys()
        },
    }

    df_table = pd.DataFrame()
    for i, dataset in enumerate(DATASETS):

        # Standard Normal and VMM
        df_baselines = pd.read_pickle(ASSETS / f"scRNA/{dataset}/op_baselines.pkl")
        df_baselines["Dataset"] = DATASETS[dataset]
        df_baselines.set_index(common_index, inplace=True)
        df_baselines = df_baselines[metric_cols].rename(columns=rename_cols)
        df_baselines = _summary_scores(df_baselines)  # per-seed aggregates
        mean_baselines = df_baselines.groupby(common_index).mean()
        std_baselines = df_baselines.groupby(common_index).std()

        # DPVampPriorUnchained
        df_alpha = pd.read_pickle(ASSETS / f"scRNA/{dataset}/sweep_dp_conc.pkl")
        df_alpha["Dataset"] = DATASETS[dataset]
        df_alpha.set_index(common_index + ["$\\alpha$"], inplace=True)
        df_alpha = df_alpha[metric_cols].rename(columns=rename_cols)
        df_alpha = _summary_scores(df_alpha)  # per-seed aggregates
        mean_alpha = df_alpha.groupby(common_index + ["$\\alpha$"]).mean()
        std_alpha = df_alpha.groupby(common_index + ["$\\alpha$"]).std()

        # select the best-alpha row by mean Aggregate/Total, keep matching SE
        best = mean_alpha["Aggregate/Total"].idxmax()
        mean_alpha = mean_alpha.loc[[best], :]
        std_alpha = std_alpha.loc[[best], :]
        mean_alpha.index = mean_alpha.index.droplevel("$\\alpha$")
        std_alpha.index = std_alpha.index.droplevel("$\\alpha$")

        # combine priors and drop KL Warmup from the index
        mean_dataset = pd.concat([mean_baselines, mean_alpha])
        std_dataset = pd.concat([std_baselines, std_alpha])
        mean_dataset.index = mean_dataset.index.droplevel("KL Warmup")
        std_dataset.index = std_dataset.index.droplevel("KL Warmup")

        # format mean +/- SE and bold every prior tied with the best at display precision
        df_dataset = mean_dataset.copy().astype(object)
        for col in mean_dataset.columns:
            rounded = mean_dataset[col].round(PRECISION)
            best_val = rounded.max()  # all metrics are "greater is better"
            for idx in mean_dataset.index:
                text = _fmt_score(mean_dataset.loc[idx, col], std_dataset.loc[idx, col])
                if rounded.loc[idx] == best_val:
                    text = f"\\textbf{{{text}}}"
                df_dataset.loc[idx, col] = text

        df_table = pd.concat([df_table, df_dataset])

    df_table.columns = pd.MultiIndex.from_tuples(
        tuple(c.split("/", 1)) for c in df_table.columns
    )
    group_order = [g for g in GROUP_ORDER if g in df_table.columns.get_level_values(0)]
    df_table = df_table.reindex(columns=group_order, level=0)

    df_table = df_table.rename(index=PRIOR_NAME_MAP, level="Prior")

    # vertical rule before the index and before each metric group
    group_sizes = [
        int((df_table.columns.get_level_values(0) == g).sum()) for g in group_order
    ]
    column_format = "l" * df_table.index.nlevels + "".join(
        "|" + "l" * size for size in group_sizes
    )

    latex_table = df_table.to_latex(
        escape=False,
        multirow=True,
        na_rep="--",
        column_format=column_format,
    )
    latex_table = _insert_dataset_midrules(latex_table, list(DATASETS.values()))
    (ASSETS / "scRNA" / "scRNA_table.tex").write_text(latex_table)


if __name__ == "__main__":
    cache_results()
    build_table()
