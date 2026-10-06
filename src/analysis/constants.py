from pathlib import Path

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

PRIOR_NAME_MAP = {
    "VampPrior": "VampPrior",
    "VampPriorUnchained": "VampPrior-Unleashed",
    "DPVampPrior": "DP-VampPrior",
    "DPVampPriorUnchained": "DP-VampPrior-Unleashed",
    "VMM": "VMM",
}
