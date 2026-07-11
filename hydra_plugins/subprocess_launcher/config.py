from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class SubprocessLauncherConf:
    """Launcher that runs every sweep job in its own OS process.

    Each job is executed by re-invoking the current script as a fresh
    subprocess, with its concrete Hydra overrides passed on the command line
    exactly as if it had been typed by hand. The only thing that ever crosses
    the process boundary is a list of override strings, so nothing about the
    task (models, dataloaders, loggers, CUDA contexts, ...) needs to be
    picklable, and every job starts from a clean interpreter, so no teardown
    code is needed between jobs.
    """

    _target_: str = (
        "hydra_plugins.subprocess_launcher.subprocess_launcher.SubprocessLauncher"
    )

    # Maximum number of jobs to run concurrently. 1 (the default) runs jobs
    # one at a time. Set to null to launch every job in the batch at once.
    n_jobs: Optional[int] = 1


ConfigStore.instance().store(
    group="hydra/launcher",
    name="subprocess",
    node=SubprocessLauncherConf,
    provider="subprocess_launcher",
)
