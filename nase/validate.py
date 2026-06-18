from typing import Any, Dict, List
import os

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from nase.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> List[Dict[str, Any]]:
    # Keep behavior consistent with training entrypoint (tags, config printing, etc.).
    extras(cfg)

    # Support reading checkpoint path from environment variable to avoid Hydra parsing issues with '=' in paths
    ckpt_path = cfg.get("ckpt_path") or os.getenv("NASE_CKPT_PATH")
    
    if not ckpt_path:
        raise ValueError("ckpt_path is required for checkpoint validation. Pass via 'ckpt_path=...' or set NASE_CKPT_PATH environment variable.")

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info(f"Running validation from checkpoint: {ckpt_path}")
    metrics = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True)
    log.info(f"Validation metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    main()
