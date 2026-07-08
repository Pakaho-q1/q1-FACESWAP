from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from core.processor_config import ProcessorConfig, build_processor_config_from_run_config
from core.types import RunConfig


@dataclass
class PipelineQueues:
    detect: queue.Queue
    swap: queue.Queue
    restore: queue.Queue
    parse: queue.Queue
    out: queue.Queue


@dataclass
class PipelineMetrics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    counters: Dict[str, int] = field(default_factory=dict)

    def increment(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.counters)


@dataclass
class PipelineState:
    config: RunConfig
    proc_cfg: ProcessorConfig          # ← explicit, immutable, thread-safe
    queues: PipelineQueues
    stage_concurrency: Dict[str, int]
    stage_active: Dict[str, int]
    model_manager: Any = None
    progress_callback: Optional[Callable[[str, int, int], None]] = None
    write_fps_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    tuner_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    preview_enabled: bool = True
    preview_interval_s: float = 0.4
    slot_cond: threading.Condition = field(default_factory=threading.Condition)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    # Set by io_image / io_video when an unrecoverable error occurs so that
    # writer threads can break out of their loop instead of blocking forever.
    abort_event: threading.Event = field(default_factory=threading.Event)

    def ordered_stages(self):
        stages = ["detect", "swap"]
        if self.config.enable_restore:
            stages.append("restore")
        if self.config.enable_parser:
            stages.append("parse")
        return stages

    def request_abort(self) -> None:
        """Signal cooperative shutdown across all worker/tuner/writer threads."""
        self.abort_event.set()
        with self.slot_cond:
            self.slot_cond.notify_all()


def create_pipeline_state(run_config: RunConfig) -> PipelineState:
    queues = PipelineQueues(
        detect=queue.Queue(maxsize=run_config.worker_queue_size),
        swap=queue.Queue(maxsize=run_config.worker_queue_size),
        restore=queue.Queue(maxsize=run_config.worker_queue_size),
        parse=queue.Queue(maxsize=run_config.worker_queue_size),
        out=queue.Queue(maxsize=run_config.out_queue_size),
    )
    stage_concurrency = {"detect": 2, "swap": 2}
    if run_config.enable_restore:
        stage_concurrency["restore"] = 2
    if run_config.enable_parser:
        stage_concurrency["parse"] = 2
    stage_active = {stage: 0 for stage in stage_concurrency}

    proc_cfg = build_processor_config_from_run_config(run_config)

    return PipelineState(
        config=run_config,
        proc_cfg=proc_cfg,
        queues=queues,
        stage_concurrency=stage_concurrency,
        stage_active=stage_active,
    )
