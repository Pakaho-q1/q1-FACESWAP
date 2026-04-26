"""
Immutable configuration passed explicitly to each processor.
Replaces direct reads from `core.config` globals, making processors
thread-safe and usable without the CLI config module.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessorConfig:
    # Swapper
    enable_swapper: bool
    swapper_blend: float

    # Restore
    enable_restore: bool
    restore_size: int
    restore_weight: float
    restore_blend: float

    # Parser
    enable_parser: bool
    parser_type: str          # "bisenet" | "segformer"
    parser_mask_blur: int
    preserve_swap_eyes: bool


def build_processor_config_from_run_config(run_config) -> ProcessorConfig:
    """Build a ProcessorConfig from a RunConfig + cfg_module at pipeline start.

    Called once in model_manager / library_api so the rest of the pipeline
    never needs to import core.config again.
    """
    return ProcessorConfig(
        enable_swapper=run_config.enable_swapper,
        swapper_blend=run_config.swapper_blend,
        enable_restore=run_config.enable_restore,
        restore_size=run_config.restore_size,
        restore_weight=run_config.restore_weight,
        restore_blend=run_config.restore_blend,
        enable_parser=run_config.enable_parser,
        parser_type=run_config.parser_type,
        parser_mask_blur=run_config.parser_mask_blur,
        preserve_swap_eyes=run_config.preserve_swap_eyes,
    )
