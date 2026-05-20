from pathlib import Path

from omegaconf import DictConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from RVT.callbacks.detection import DetectionVizCallback


def get_ckpt_callback(config: DictConfig) -> ModelCheckpoint:
    model_name = config.model.name

    prefix = "val"
    if model_name == "rnndet":
        metric = "AP"
        mode = "max"
    else:
        raise NotImplementedError
    ckpt_callback_monitor = prefix + "/" + metric
    filename_monitor_str = prefix + "_" + metric

    ckpt_filename = (
        "epoch={epoch:03d}-step={step}-"
        + filename_monitor_str
        + "={"
        + ckpt_callback_monitor
        + ":.2f}"
    )
    run_name = config.wandb.get("run_name", None)
    checkpoint_dir = None
    if run_name is not None:
        checkpoint_dir = Path(config.wandb.project_name) / str(run_name) / "checkpoints"

    cktp_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,  # because backslash would create a directory
        save_top_k=1,
        mode=mode,
        every_n_epochs=config.logging.ckpt_every_n_epochs,
        save_last=True,
        verbose=True,
    )
    cktp_callback.CHECKPOINT_NAME_LAST = "last_epoch={epoch:03d}-step={step}"
    return cktp_callback


def get_viz_callback(config: DictConfig) -> Callback:
    model_name = config.model.name

    if model_name == "rnndet":
        return DetectionVizCallback(config=config)
    raise NotImplementedError
