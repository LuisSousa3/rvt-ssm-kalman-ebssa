"""Entry point for the EBSSA/RVT training and evaluation pipeline."""

from __future__ import annotations

from config import DataConfig
from methods.pipeline import run_pipeline


def main() -> None:
    run_pipeline(DataConfig())


if __name__ == "__main__":
    main()
