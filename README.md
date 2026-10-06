# VampPrior-Unleashed

Code submission for VampPrior-Unleashed.

## 1. Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python >= 3.12).

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

All commands below are run through `uv run` from the repository root.

## 2. Weights & Biases

Experiment results are logged with [Weights & Biases](https://wandb.ai) (W&B). Set the following environment variables:

```bash
export WANDB_API_KEY=<your API key>
export WANDB_PATH=<project>   # where runs are logged and later fetched by the analysis
```

If W&B is not accessible, remove `wandb` from the defaults list in [conf/loggers/default.yaml](conf/loggers/default.yaml):

```yaml
defaults:
  - csv
  - tensorboard
  # - wandb
```

(Note: the analysis in step 4 reads from W&B, so it will not work without it unless rewritten to utilize a different logger)

## 3. Run experiments

Experiment scripts live in [scripts/](scripts). Each takes a dataset config name and optional flags; run a script without arguments or see [scripts/arg_parser.sh](scripts/arg_parser.sh) for usage. For a complete reproduction run all of:

```bash
bash scripts/toy.sh

bash scripts/clustering/baselines.sh mnist -n 5
bash scripts/clustering/baselines.sh fashion_mnist -n 5

bash scripts/clustering/sweep_k.sh mnist -n 5
bash scripts/clustering/sweep_k.sh fashion_mnist -n 5

bash scripts/clustering/sweep_dp_conc.sh mnist -n 5
bash scripts/clustering/sweep_dp_conc.sh fashion_mnist -n 5

bash scripts/scRNA/op_baselines.sh op_dkd -n 1
bash scripts/scRNA/op_baselines.sh op_immune_cell_atlas -n 1
bash scripts/scRNA/op_baselines.sh op_gtex_v9 -n 1
bash scripts/scRNA/op_baselines.sh op_mouse_pancreas_atlas -n 1
bash scripts/scRNA/op_baselines.sh op_hypomap -n 1
bash scripts/scRNA/op_baselines.sh op_tabula_sapiens -n 1

bash scripts/scRNA/sweep_dp_conc.sh op_dkd -n 3
bash scripts/scRNA/sweep_dp_conc.sh op_immune_cell_atlas -n 3
bash scripts/scRNA/sweep_dp_conc.sh op_gtex_v9 -n 3
bash scripts/scRNA/sweep_dp_conc.sh op_mouse_pancreas_atlas -n 3
bash scripts/scRNA/sweep_dp_conc.sh op_hypomap -n 3
bash scripts/scRNA/sweep_dp_conc.sh op_tabula_sapiens -n 3
```

Options: `-n/--num_jobs N` sets the number of parallel Hydra jobs, and `--debug` disables W&B logging.

## 4. Run analysis

The scripts in [src/analysis/](src/analysis) build the tables and figures used in the paper:

```bash
PYTHONPATH=. uv run python src/analysis/build_clustering_assets.py
PYTHONPATH=. uv run python src/analysis/build_scRNA_assets.py
```

This step requires W&B (`WANDB_API_KEY` and `WANDB_PATH`), since results are fetched from the W&B API in [src/analysis/utils.py](src/analysis/utils.py). It could be rewritten to read from the CSV logger output instead.
