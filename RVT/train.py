import os
from pathlib import Path
from typing import Any

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
import hdf5plugin
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from RVT.callbacks.custom import get_ckpt_callback, get_viz_callback
from RVT.callbacks.gradflow import GradFlowLogCallback
from RVT.config.modifier import dynamically_modify_train_config
from RVT.data.utils.types import DatasetSamplingMode
from RVT.loggers.utils import get_wandb_logger, get_ckpt_path
from RVT.modules.utils.fetch import fetch_data_module, fetch_model_module
from RVT.modules.detection import Module


def load_pretrained_weights(
    module: Module,
    checkpoint_path: str | Path,
    reset_classification_head: bool,
) -> dict[str, Any]:
    """Load transfer-learning weights while keeping the target class head fresh."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")

    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained_state = checkpoint.get("state_dict", checkpoint)
    target_state = module.state_dict()

    compatible_state = {}
    skipped_classification_head = []
    unexpected_keys = []
    mismatched_non_classification_keys = []

    for key, value in pretrained_state.items():
        if reset_classification_head and key.startswith("mdl.yolox_head.cls_preds."):
            skipped_classification_head.append(key)
            continue

        target_value = target_state.get(key)
        if target_value is None:
            unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            mismatched_non_classification_keys.append(
                (key, tuple(value.shape), tuple(target_value.shape))
            )
            continue
        compatible_state[key] = value

    if mismatched_non_classification_keys:
        details = ", ".join(
            f"{key}: {source_shape} -> {target_shape}"
            for key, source_shape, target_shape in mismatched_non_classification_keys
        )
        raise RuntimeError(
            "Pretrained checkpoint does not match the requested model outside the "
            f"classification head: {details}"
        )

    missing_keys, load_unexpected_keys = module.load_state_dict(
        compatible_state,
        strict=False,
    )
    if load_unexpected_keys:
        raise RuntimeError(f"Unexpected keys after filtering pretrained weights: {load_unexpected_keys}")

    print(
        "Loaded "
        f"{len(compatible_state)}/{len(pretrained_state)} pretrained tensors; "
        f"kept {len(skipped_classification_head)} classification-head tensors freshly initialized "
        f"for {module.mdl.yolox_head.num_classes} target class(es)."
    )
    if unexpected_keys:
        print(f"Ignored {len(unexpected_keys)} unexpected checkpoint tensors.")

    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_tensors": len(compatible_state),
        "checkpoint_tensors": len(pretrained_state),
        "skipped_classification_head": tuple(skipped_classification_head),
        "missing_keys": tuple(missing_keys),
        "unexpected_keys": tuple(unexpected_keys),
    }


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (
        DatasetSamplingMode.STREAM,
        DatasetSamplingMode.MIXED,
    )
    if disable_seed_everything:
        print(
            "Disabling PL seed everything because of unresolved issues with shuffling during training on streaming "
            "datasets"
        )
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f"USING pl.seed_everything WITH {seed=}")
        pl.seed_everything(seed=seed, workers=True)

    # ---------------------
    # DDP
    # ---------------------
    accelerator = config.hardware.get("accelerator", "gpu")
    assert accelerator in ("gpu", "cpu"), f"{accelerator=}"
    if accelerator == "gpu":
        gpu_config = config.hardware.gpus
        gpus = (
            OmegaConf.to_container(gpu_config)
            if OmegaConf.is_config(gpu_config)
            else gpu_config
        )
        gpus = gpus if isinstance(gpus, list) else [gpus]
        devices = gpus
        distributed_backend = config.hardware.dist_backend
        assert distributed_backend in ("nccl", "gloo"), f"{distributed_backend=}"
        strategy = (
            DDPStrategy(
                process_group_backend=distributed_backend,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            if len(gpus) > 1
            else "auto"
        )
    else:
        devices = 1
        strategy = "auto"

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    wandb_logger = get_wandb_logger(config)
    csv_log_dir = os.environ.get("RVT_CSV_LOG_DIR", "data/rvt_csv_logs")
    csv_logger = CSVLogger(save_dir=csv_log_dir, name=config.wandb.group_name)
    loggers = [wandb_logger, csv_logger]
    ckpt_path = None
    if config.wandb.artifact_name is not None:
        ckpt_path = get_ckpt_path(wandb_logger, wandb_config=config.wandb)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    pretrained_checkpoint = config.training.get("pretrained_checkpoint", None)
    if ckpt_path is None and pretrained_checkpoint is not None:
        load_pretrained_weights(
            module=module,
            checkpoint_path=pretrained_checkpoint,
            reset_classification_head=bool(
                config.training.pretrained_reset_classification_head
            ),
        )
    if ckpt_path is not None and config.wandb.resume_only_weights:
        print("Resuming only the weights instead of the full training state")
        module = Module.load_from_checkpoint(
            str(ckpt_path), **{"full_config": config}, strict=False
        )

        ckpt_path = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if (
        config.logging.train.high_dim.enable
        or config.logging.validation.high_dim.enable
    ):
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    wandb_logger.watch(
        model=module,
        log="all",
        log_freq=config.logging.train.log_model_every_n_steps,
        log_graph=True,
    )

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=devices,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm="value",
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=loggers,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy == "auto" else True,
        # move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == "__main__":
    main()
