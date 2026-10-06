from pathlib import Path

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

PRIOR_NAME_MAP = {
    "VampPrior": "VampPrior",
    "VampPriorUnleashed": "VampPrior-Unleashed",
    "DPVampPrior": "DP-VampPrior",
    "DPVampPriorUnleashed": "DP-VampPrior-Unleashed",
    "VMM": "VMM",
}
