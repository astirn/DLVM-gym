import itertools
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from toolz import keyfilter

from .constants import ASSETS, PRIOR_NAME_MAP
from .utils import get_relevant_runs

DATASETS = ["MNIST", "FashionMNIST"]
PRIOR_ORDER = [
    "VampPrior",
    "VampPriorUnchained",
    "DPVampPrior",
    "DPVampPriorUnchained",
    "VMM",
]
PRIOR_COLORS = dict(
    zip(PRIOR_ORDER, sns.color_palette("colorblind", n_colors=len(PRIOR_ORDER)))
)
DOMINATED_STYLE = {"marker": ".", "s": 20, "alpha": 0.6, "edgecolors": "none"}
NON_DOMINATED_STYLE = {"marker": ".", "s": 50, "alpha": 1.0, "edgecolors": "none"}
FRONTIER_STYLE = {"linestyle": "-", "linewidth": 1.0}
STAR_STYLE = {"marker": "*", "s": 100, "edgecolor": "k"}

PRIOR_HYPER_PARAMS = {
    "StandardNormal": None,
    "VampPrior": ["Init.", "K"],
    "VampPriorUnchained": ["Init.", "K"],
    "DPVampPrior": ["Init.", "$\\alpha$"],
    "DPVampPriorUnchained": ["Init.", "$\\alpha$"],
    "VMM": ["Init.", "num_data"],
}


@dataclass
class Metric:
    wandb_name: str
    alternative: str | None = None


METRICS = {
    "Log marginal": Metric("test/log_marginal", "greater"),
    "Distortion": Metric("test/distortion", "less"),
    "Rate": Metric("test/rate", "less"),
    "Marginal KL": Metric("test/marginal_kl", "less"),
    "Clusters": Metric("test/clusters", None),
    "NMI": Metric("test/nmi", "greater"),
}

HV = "Hypervolume"
P_STAR = "$P^{\\ast}$"
P_STAR_DIST = "Min $||P^{\\ast} - (1,1)||_2$"
P_STAR_LM = f"{P_STAR} Log marginal"
P_STAR_NMI = f"{P_STAR} NMI"
MAX_LM = "Max Log marginal"
MAX_NMI = "Max NMI"
TABLE_METRICS = {
    HV: "greater",
    P_STAR_DIST: "less",
    MAX_LM: "greater",
    MAX_NMI: "greater",
}


def _get_clustering_results(
    experiment: str,
    tag: str,
    param_name: str | None = None,
    plot_name: str | None = None,
    value_filter: list[Any] | None = None,
) -> None:
    filter_list: list[dict[str, Any]] = [
        {"group": {"$eq": experiment}},
        {"tags": {"$eq": tag}},
    ]
    if value_filter is not None and param_name is not None:
        filter_list.append(
            {f"config.model.prior_params.{param_name}": {"$in": value_filter}}
        )
    filters = {
        "$and": filter_list,
    }
    df = pd.DataFrame()
    for config, summary in get_relevant_runs(filters):
        index = {
            "Prior": config["model"]["prior_params"]["name"],
            "Init.": config["model"]["u_initial_strategy"],
            "seed": config["seed"],
        }
        if param_name is not None and plot_name is not None:
            index[plot_name] = config["model"]["prior_params"][param_name]
        try:
            metrics = {
                key: summary[metric.wandb_name] for key, metric in METRICS.items()
            }
            new_row = pd.DataFrame({**index, **metrics}, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        except KeyError:
            continue

    save_path = ASSETS / f"{experiment}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_path)


def cache_results():
    for dataset in DATASETS:
        _get_clustering_results(
            f"Clustering/{dataset}/baselines",
            tag="v3",
            param_name="num_data",
            plot_name="num_data",
        )
        _get_clustering_results(
            f"Clustering/{dataset}/sweep_k",
            tag="v3",
            param_name="k",
            plot_name="K",
        )
        _get_clustering_results(
            f"Clustering/{dataset}/sweep_dp_conc",
            tag="v3",
            param_name="concentration",
            plot_name="$\\alpha$",
        )


def _add_statistical_tests(
    df: pd.DataFrame, plot_name: str, null_prior: str, alt_prior: str
) -> pd.DataFrame:
    df = df.set_index(["Init.", plot_name]).sort_index()
    for index in df.index.unique():
        df_config = df.loc[index].set_index("seed", append=True).sort_index()
        null = df_config[df_config["Prior"] == null_prior]
        alt = df_config[df_config["Prior"] == alt_prior]
        for name, metric in METRICS.items():
            if metric.alternative is None:
                continue
            delta = alt[name] - null[name]
            test = pg.wilcoxon(delta, alternative=metric.alternative)
            df.loc[index, name + " p-val"] = test["p_val"].item()
    df.reset_index(inplace=True)
    return df


def plot_sweep(
    experiment: str, param: str, filter: set | None = None, log_x: bool = False
):
    metrics = keyfilter(lambda k: k != "Clusters", METRICS)

    for dataset in DATASETS:
        df = pd.read_pickle(ASSETS / f"Clustering/{dataset}/{experiment}.pkl")
        if filter is not None:
            df = df[df[param].isin(filter)]

        priors = list(sorted(df["Prior"].dropna().unique()))
        if len(priors) != 2:
            raise ValueError(
                "Expected exactly two unique 'Prior' values for comparison."
            )
        df = _add_statistical_tests(
            df, param, null_prior=priors[0], alt_prior=priors[1]
        )

        for init in sorted(df["Init."].dropna().unique()):
            df_init = df[df["Init."] == init]

            fig, ax = plt.subplots(
                nrows=1,
                ncols=len(metrics),
                figsize=(4 * len(metrics), 4),
                constrained_layout=False,
            )

            for i, metric in enumerate(metrics.keys()):
                sns.lineplot(
                    data=df_init,
                    x=param,
                    y=metric,
                    hue="Prior",
                    hue_order=PRIOR_COLORS,
                    errorbar=("se", 1),
                    legend=("full" if i == 0 else False),
                    ax=ax[i],
                )
                if log_x:
                    ax[i].set_xscale("log")

                for val in df_init[param].unique():
                    df_k = df_init[df_init[param] == val]
                    p_val = df_k[metric + " p-val"].unique().item()
                    if p_val < 0.05:
                        ymin = df_k[df_k["Prior"] == priors[0]][metric].mean()
                        ymax = df_k[df_k["Prior"] == priors[1]][metric].mean()
                        ax[i].vlines(
                            x=val, ymin=ymin, ymax=ymax, color="k", linestyle="--"
                        )
                        if METRICS[metric].alternative == "greater":
                            ax[i].scatter([val], [ymax], marker="^", color="k", s=25)
                        else:
                            ax[i].scatter([val], [ymax], marker="v", color="k", s=25)

                if metric == "NMI":
                    ax[i].set_ylim(0.5, None)

            handles, labels = ax[0].get_legend_handles_labels()
            keep = [
                i
                for i, label in enumerate(labels)
                if label in df_init["Prior"].unique()
            ]
            if ax[0].legend_ is not None:
                ax[0].legend_.remove()
            fig.legend(
                [handles[i] for i in keep],
                [PRIOR_NAME_MAP.get(str(labels[i]), str(labels[i])) for i in keep],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=len(keep),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.savefig(ASSETS / "Clustering" / dataset / f"{experiment}_{init}.pdf")


def _characterize_pareto_frontier(
    df: pd.DataFrame, bootstrap: bool = False, random_state: int | None = None
) -> tuple[
    pd.DataFrame, dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    # order runs by seed so that, under a shared random_state, bootstrap draws line up
    # across priors by seed rather than by arbitrary row order
    df = df.sort_values("seed").set_index("Prior").sort_index(kind="stable")
    metrics = list(METRICS.keys())

    # average prior model performance per configuration
    df_avg = pd.DataFrame()
    for prior in df.index.get_level_values("Prior").unique():
        df_prior = df.loc[prior]
        hyper_params = PRIOR_HYPER_PARAMS[prior]
        if hyper_params is None:
            if bootstrap:
                df_prior = df_prior.sample(
                    frac=1, replace=True, random_state=random_state
                )
            df_prior = df_prior[metrics].mean().to_frame().T

        else:
            if bootstrap:
                df_prior = df_prior.groupby(hyper_params, group_keys=False).sample(
                    frac=1, replace=True, random_state=random_state
                )
            df_prior = (
                df_prior.groupby(hyper_params)[metrics].mean().reset_index(drop=True)
            )

        df_prior["Prior"] = prior
        df_avg = pd.concat([df_avg, df_prior], ignore_index=True)
    df_avg = df_avg.set_index("Prior").sort_index()

    # set Pareto bounds (origin is Standard normal performance)
    x_min = df_avg.loc["StandardNormal"]["Log marginal"].min()
    x_max = df_avg["Log marginal"].max()
    y_min = df_avg.loc["StandardNormal"]["NMI"].min()
    y_max = df_avg["NMI"].max()
    pareto_bounds = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    performance = pd.DataFrame()
    performance.index.name = "Prior"
    xy = {}
    xy_pareto = {}
    for prior in df_avg.index.get_level_values("Prior").unique():
        if prior == "StandardNormal":
            continue

        # get X, Y values for the current prior and initialization strategy
        x = df_avg.loc[prior]["Log marginal"].to_numpy()
        y = df_avg.loc[prior]["NMI"].to_numpy()

        # determine Pareto-optimal points for the current prior and initialization strategy
        outside = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
        x = x[~outside]
        y = y[~outside]

        # no in-bounds points -> empty frontier: HV = 0, distance undefined
        if x.size == 0:
            performance.loc[prior, HV] = 0.0
            performance.loc[prior, P_STAR_DIST] = np.nan
            performance.loc[prior, MAX_LM] = np.nan
            performance.loc[prior, MAX_NMI] = np.nan
            continue

        # find Pareto-optimal points
        is_pareto = np.ones(x.shape[0], dtype=bool)
        for i, (xi, yi) in enumerate(zip(x, y)):
            is_pareto[i] = not np.any((x >= xi) & (y >= yi) & ((x > xi) | (y > yi)))
        x_pareto, y_pareto = x[is_pareto], y[is_pareto]

        # normalize Pareto-optimal points to [0, 1] range
        x_pareto_normalized = (x_pareto - x_min) / (x_max - x_min)
        y_pareto_normalized = (y_pareto - y_min) / (y_max - y_min)

        # compute hypervolume
        order = np.lexsort((-y_pareto_normalized, -x_pareto_normalized))
        x_sorted, y_sorted = x_pareto_normalized[order], y_pareto_normalized[order]
        hv, y_best = 0.0, 0.0
        for x_i, y_i in zip(x_sorted, y_sorted):
            if y_i > y_best:
                hv += x_i * (y_i - y_best)
                y_best = y_i

        # L2 distance to the optimal point (1, 1)
        l2_to_optimal = (
            (1 - x_pareto_normalized) ** 2 + (1 - y_pareto_normalized) ** 2
        ) ** 0.5
        i_min = np.argmin(l2_to_optimal)

        # save tabular performance metrics
        performance.loc[prior, HV] = hv
        performance.loc[prior, P_STAR_DIST] = l2_to_optimal[i_min]
        performance.loc[prior, P_STAR_LM] = x_pareto[i_min]
        performance.loc[prior, P_STAR_NMI] = y_pareto[i_min]
        performance.loc[prior, MAX_LM] = x_pareto.max()
        performance.loc[prior, MAX_NMI] = y_pareto.max()

        # save points for plotting Pareto frontiers
        xy[prior] = np.vstack([x, y])
        xy_pareto[prior] = np.vstack([x_pareto, y_pareto])

    return performance, pareto_bounds, xy, xy_pareto


def _fmt_point_ci(point_val: float, s: pd.Series, p: int = 3) -> str:
    # point estimate as the value, bootstrap percentile interval as the bracket
    if pd.isna(point_val):
        return "--"
    s = s.dropna()
    if s.empty:
        return f"{point_val:.{p}f}"
    q_lo, q_hi = s.quantile(0.025), s.quantile(0.975)
    text = f"{point_val:.{p}f} [{q_lo:.{p}f}, {q_hi:.{p}f}]"
    if point_val < q_lo or point_val > q_hi:
        text = f"\\textcolor{{red}}{{{text}}}"
    return text


def _prob_delta_gt_zero(metric: str) -> str:
    return f"$P(\\Delta$ {metric} $> 0)$"


def _build_tables(
    mean_performance: pd.DataFrame,
    bootstrap_performance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    priors_present = [p for p in PRIOR_ORDER if p in mean_performance.index.unique()]

    mean_df = (
        mean_performance.reset_index().set_index(["Dataset", "Prior"]).sort_index()
    )
    bs_df = (
        bootstrap_performance.reset_index().set_index(["Dataset", "Prior"]).sort_index()
    )

    # --- Build paired difference table ---
    rows = []
    for dataset in DATASETS:
        if dataset not in bs_df.index.get_level_values("Dataset"):
            continue
        wide = {
            m: bs_df.loc[dataset]
            .reset_index()
            .pivot(index="iteration", columns="Prior", values=m)
            .reindex(columns=priors_present)
            for m in mean_df.columns
        }
        for a, b in itertools.combinations(priors_present, 2):
            row = {"Dataset": dataset, "Comparison": f"{a} $-$ {b}"}
            for m in mean_df.columns:
                d = (wide[m][a] - wide[m][b]).dropna()
                row[_prob_delta_gt_zero(m)] = f"{(d > 0).mean():.3f}"
            rows.append(row)

    paired_df = pd.DataFrame(rows).set_index(["Dataset", "Comparison"])

    # --- Build per-prior mean table ---
    rows = []
    for dataset in DATASETS:
        if dataset not in mean_df.index.get_level_values("Dataset"):
            continue

        prior_rows = {
            prior: {"Dataset": dataset, "Prior": prior} for prior in priors_present
        }
        for metric, direction in TABLE_METRICS.items():

            if direction == "greater":
                best_prior = mean_df.loc[dataset][metric].idxmax()
            elif direction == "less":
                best_prior = mean_df.loc[dataset][metric].idxmin()
            else:
                best_prior = None

            for prior in priors_present:
                center = float(mean_df.loc[(dataset, prior), metric].item())
                ci = bs_df.loc[(dataset, prior), metric]
                text = _fmt_point_ci(center, ci)

                if prior == best_prior:
                    text = f"\\textbf{{{text}}}"

                elif isinstance(best_prior, str):
                    separator = re.escape(" $-$ ")
                    pos_delta = re.compile(
                        rf"^{re.escape(best_prior)}{separator}{re.escape(prior)}$"
                    )
                    neg_delta = re.compile(
                        rf"^{re.escape(prior)}{separator}{re.escape(best_prior)}$"
                    )
                    comp_found = False
                    for comp in paired_df.loc[dataset].index:
                        pos_delta_match = pos_delta.fullmatch(comp)
                        neg_delta_match = neg_delta.fullmatch(comp)
                        if pos_delta_match or neg_delta_match:
                            comp_found = True

                            delta_str = _prob_delta_gt_zero(metric)
                            pgt0 = float(
                                cast(str, paired_df.loc[(dataset, comp), delta_str])
                            )
                            if 0.025 < pgt0 < 0.975:
                                text = f"\\textbf{{{text}}}"
                            break

                    if not comp_found:
                        raise KeyError(
                            f"No comparison found for {prior!r} and {best_prior!r}"
                        )

                prior_rows[prior][metric] = text
        rows.extend(prior_rows.values())

    per_prior_df = pd.DataFrame(rows).set_index(["Dataset", "Prior"])

    def _map_comparison(comparison: str) -> str:
        return " $-$ ".join(PRIOR_NAME_MAP.get(p, p) for p in comparison.split(" $-$ "))

    per_prior_df = per_prior_df.rename(index=PRIOR_NAME_MAP, level="Prior")
    paired_df = paired_df.rename(index=_map_comparison, level="Comparison")

    return per_prior_df, paired_df


def _add_shared_legend(fig, anchor_ax):
    def header(text):
        return Line2D([], [], linestyle="none", marker="none", label=text)

    def points_handle(style, label):
        return Line2D(
            [],
            [],
            color="k",
            linestyle="none",
            marker=style["marker"],
            markersize=np.sqrt(style["s"]),
            alpha=style["alpha"],
            markeredgecolor=style.get("edgecolors", "k"),
            label=label,
        )

    handles = [header("Priors")]
    handles.append(
        Line2D(
            [],
            [],
            color="k",
            marker="x",
            linestyle="none",
            markersize=8,
            label="StandardNormal",
        )
    )
    for prior in PRIOR_ORDER:
        handles.append(
            Line2D(
                [],
                [],
                color=PRIOR_COLORS[prior],
                marker="s",
                linestyle="none",
                markersize=8,
                label=PRIOR_NAME_MAP.get(prior, prior),
            )
        )

    handles.append(header("Style"))
    handles.append(Line2D([], [], color="k", label="Pareto frontier", **FRONTIER_STYLE))
    handles.append(points_handle(DOMINATED_STYLE, "Dominated points"))
    handles.append(points_handle(NON_DOMINATED_STYLE, "Non-dominated points"))
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker=STAR_STYLE["marker"],
            markersize=np.sqrt(STAR_STYLE["s"]),
            markerfacecolor="white",
            markeredgecolor=STAR_STYLE["edgecolor"],
            label="L2 optimal point",
        )
    )

    legend = fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=anchor_ax.transAxes,
        frameon=False,
    )
    for text in legend.get_texts():
        if text.get_text() in ("Priors", "Style"):
            text.set_fontweight("bold")


def pareto_analysis(n_boot: int):

    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(DATASETS),
        figsize=(5 * len(DATASETS), 5),
        constrained_layout=False,
    )

    mean_performance = pd.DataFrame()
    bootstrap_performance = pd.DataFrame()
    for i, dataset in enumerate(DATASETS):

        # Standard Normal and VMM
        df_baselines = pd.read_pickle(ASSETS / f"Clustering/{dataset}/baselines.pkl")
        df_sn = df_baselines[df_baselines["Prior"] == "StandardNormal"]
        df_vmm = df_baselines[df_baselines["Prior"] == "VMM"]

        # VampPrior and VampPriorUnchained
        df_k = pd.read_pickle(ASSETS / f"Clustering/{dataset}/sweep_k.pkl")

        # DPVampPrior and DPVampPriorUnchained
        df_alpha = pd.read_pickle(ASSETS / f"Clustering/{dataset}/sweep_dp_conc.pkl")

        # Combine all priors into a single DataFrame for plotting
        df = pd.concat([df_sn, df_k, df_alpha, df_vmm])
        df = df[
            (df["Prior"] == "StandardNormal")
            | (
                df["Prior"].isin(["VampPrior", "VampPriorUnchained"])
                & df["Init."].isin(["data", "random"])
            )
            | (
                df["Prior"].isin(["DPVampPrior", "DPVampPriorUnchained"])
                & df["Init."].isin(["stratified_data", "random"])
                & df["$\\alpha$"].isin(
                    [0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 10.0, 11.0, 100.0]
                )
            )
            | (
                df["Prior"].isin(["VMM"])
                & df["Init."].isin(["data", "random"])
                & df["num_data"].isin([64, 256, 1024, 4096, 16384, 60000])
            )
        ]

        # average performance and the underlying points
        perf, pareto_bounds, xy, xy_pareto = _characterize_pareto_frontier(df)

        # bootstrap performance
        bs_list = []
        for b in range(n_boot):
            bs_b = _characterize_pareto_frontier(df, bootstrap=True, random_state=b)[0]
            bs_b["iteration"] = b
            bs_list.append(bs_b)
        bs_perf = pd.concat(bs_list)

        # aggregate performance
        perf["Dataset"] = dataset
        bs_perf["Dataset"] = dataset
        mean_performance = pd.concat([mean_performance, perf])
        bootstrap_performance = pd.concat([bootstrap_performance, bs_perf])

        # set title and aspect ratio for the subplot
        ax[i].set_title(dataset)
        ax[i].set_box_aspect(1)

        # plot standard normal
        ax[i].scatter(
            pareto_bounds["x_min"],
            pareto_bounds["y_min"],
            color="k",
            marker="x",
            s=50,
            label="Standard normal",
        )

        for prior in xy.keys():

            # plot all points
            ax[i].scatter(
                xy[prior][0],
                xy[prior][1],
                color=PRIOR_COLORS[prior],
                **DOMINATED_STYLE,
            )

            # plot frontier points
            ax[i].scatter(
                xy_pareto[prior][0],
                xy_pareto[prior][1],
                color=PRIOR_COLORS[prior],
                label=prior,
                **NON_DOMINATED_STYLE,
            )

            # demarcate the Pareto frontier and hyper-volume enclosure
            i_sort = np.argsort(xy_pareto[prior][0])
            x_pareto_plot = xy_pareto[prior][0][i_sort].tolist()
            y_pareto_plot = xy_pareto[prior][1][i_sort].tolist()
            x_pareto_plot = (
                [pareto_bounds["x_min"]] + x_pareto_plot + [x_pareto_plot[-1]]
            )
            y_pareto_plot = (
                [y_pareto_plot[0]] + y_pareto_plot + [pareto_bounds["y_min"]]
            )
            ax[i].plot(
                x_pareto_plot,
                y_pareto_plot,
                color=PRIOR_COLORS[prior],
                **FRONTIER_STYLE,
            )

            # point with largest-area rectangle
            ax[i].scatter(
                mean_performance.loc[prior, P_STAR_LM],
                mean_performance.loc[prior, P_STAR_NMI],
                color=PRIOR_COLORS[prior],
                **STAR_STYLE,
            )

        ax[i].set_xlabel("Per-datum log marginal")
        ax[i].set_ylabel("NMI")
        ax[i].set_xlim(pareto_bounds["x_min"] - 0.5, pareto_bounds["x_max"] + 0.5)
        ax[i].set_ylim(pareto_bounds["y_min"] - 0.05, pareto_bounds["y_max"] + 0.05)
        ax[i].hlines(
            y=(pareto_bounds["y_min"], pareto_bounds["y_max"]),
            xmin=pareto_bounds["x_min"],
            xmax=pareto_bounds["x_max"],
            linestyles="dashed",
            colors="gray",
        )
        ax[i].vlines(
            x=(pareto_bounds["x_min"], pareto_bounds["x_max"]),
            ymin=pareto_bounds["y_min"],
            ymax=pareto_bounds["y_max"],
            linestyles="dashed",
            colors="gray",
        )

    _add_shared_legend(fig, ax[-1])
    fig.savefig(ASSETS / "Clustering" / "pareto_figure.pdf", bbox_inches="tight")

    # build tables
    mean_table, diff_table = _build_tables(mean_performance, bootstrap_performance)
    mean_latex = mean_table.to_latex(
        escape=False,
        multirow=True,
        na_rep="--",
    )
    diff_latex = diff_table.to_latex(
        escape=False,
        multirow=True,
        na_rep="--",
    )
    (ASSETS / "Clustering" / "pareto_table_mean.tex").write_text(mean_latex)
    (ASSETS / "Clustering" / "pareto_table_diff.tex").write_text(diff_latex)


if __name__ == "__main__":
    cache_results()
    plot_sweep("sweep_k", "K")
    plot_sweep(
        "sweep_dp_conc",
        "$\\alpha$",
        filter={0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 10.0},
        log_x=True,
    )
    pareto_analysis(n_boot=10_000)
