import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from hydra.core.utils import (
    JobReturn,
    JobStatus,
    configure_log,
    filter_overrides,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from .config import SubprocessLauncherConf

log = logging.getLogger(__name__)


class SubprocessLauncher(Launcher):
    """Runs each sweep job as an independent OS process.

    Jobs are launched by re-invoking the current script (`sys.executable
    sys.argv[0]`) as a subprocess with the job's concrete overrides passed on
    the command line, i.e. the same way you would launch a single run by
    hand. Because jobs communicate with the parent only via plain argv
    strings (never pickled Python objects), any object graph the task builds
    (models, dataloaders, wandb runs, CUDA state, ...) can be arbitrarily
    unpicklable, and no teardown code is needed between jobs since each one
    gets a brand new interpreter.
    """

    def __init__(self, n_jobs: Optional[int] = SubprocessLauncherConf().n_jobs) -> None:
        self.n_jobs = n_jobs
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def _run_job(self, idx: int, overrides: Sequence[str]) -> JobReturn:
        if self.config is None or self.hydra_context is None:
            raise ValueError("config or hydra_context is not set")

        # Composed only to resolve this job's output directory; the actual
        # config composition (and the run itself) happens inside the child
        # process, so nothing here needs to be picklable.
        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, list(overrides)
        )
        job_dir = Path(str(sweep_config.hydra.sweep.dir)) / str(idx)
        job_dir.mkdir(parents=True, exist_ok=True)

        config_name = self.config.hydra.job.config_name
        cmd: List[str] = [sys.executable, sys.argv[0]]
        if config_name is not None:
            cmd += ["-cn", str(config_name)]
        cmd += list(overrides)
        cmd += [f"hydra.run.dir={job_dir}"]

        overrides_str = " ".join(filter_overrides(overrides))
        log_path = job_dir / "subprocess.log"
        log.info(f"\t#{idx} : {overrides_str}")

        ret = JobReturn()
        ret.overrides = list(overrides)
        with log_path.open("w") as log_file:
            completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        if completed.returncode == 0:
            ret.status = JobStatus.COMPLETED
            ret.return_value = None
            log.info(f"\t#{idx} : COMPLETED (log: {log_path})")
        else:
            ret.status = JobStatus.FAILED
            ret.return_value = RuntimeError(
                f"Job #{idx} failed with return code {completed.returncode} "
                f"(overrides: {overrides_str}). See {log_path} for details."
            )
            log.info(
                f"\t#{idx} : FAILED with return code {completed.returncode} "
                f"(log: {log_path})"
            )
        return ret

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        num_jobs = len(job_overrides)
        n_jobs = num_jobs if self.n_jobs is None else max(1, min(self.n_jobs, num_jobs))
        log.info(
            f"Launching {num_jobs} job(s) as separate subprocesses "
            f"({n_jobs} running concurrently)"
        )

        indexed_overrides: List[Tuple[int, Sequence[str]]] = [
            (initial_job_idx + i, overrides)
            for i, overrides in enumerate(job_overrides)
        ]
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            runs = list(
                executor.map(lambda args: self._run_job(*args), indexed_overrides)
            )
        return runs
