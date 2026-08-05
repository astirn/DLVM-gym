from omegaconf import DictConfig


def checkpoint_path(ckpt_cfg: DictConfig) -> str:
    return f"{ckpt_cfg.dirpath}/{ckpt_cfg.filename}.ckpt"
