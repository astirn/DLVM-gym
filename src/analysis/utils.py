import os
from typing import Any, Generator

import wandb
from tqdm import tqdm


def get_relevant_runs(filters: dict[str, Any]) -> Generator[
    tuple[dict, dict],
    None,
    None,
]:
    api = wandb.Api(timeout=10)
    runs = api.runs(
        path=os.getenv("WANDB_PATH"),
        filters=filters,
    )
    for run in tqdm(runs, desc="Fetching runs"):
        config = run.config
        summary = run.summary._json_dict
        yield config, summary
