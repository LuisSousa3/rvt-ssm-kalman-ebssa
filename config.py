"""Small public configuration for the project entry point."""

from dataclasses import dataclass

from methods.project_defaults import ProjectDefaults


@dataclass
class DataConfig(ProjectDefaults):
    # Use ("convert", "preprocess", "train", "evaluate") for a full run.
    # Use ("evaluate",) to rerun the final checkpoint analysis.
    pipeline_stages: tuple[str, ...] = ("evaluate",)

    # Leave as None to use the best checkpoint in ebssa_rvt/modelFinal/checkpoints.
    rvt_checkpoint_path: str | None = None

