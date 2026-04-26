from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from contextlib import contextmanager

import insightface
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from safetensors.numpy import load_file

from core.errors import ModelInitError
from core.provider_policy import build_ort_providers, resolve_provider
from core.types import RuntimeContext, RunConfig
from core.ui_log import ui_print


logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class DummySourceFace:
    def __init__(self, embedding):
        self.embedding = embedding
        self.normed_embedding = embedding


def load_source_face(filepath: str) -> DummySourceFace:
    tensors = load_file(filepath)
    if not tensors:
        raise ModelInitError(f"No tensor found in source face file: {filepath}")
    embedding = next(
        (tensor for tensor in tensors.values() if tensor.size == 512),
        max(tensors.values(), key=lambda tensor: tensor.size),
    )
    embedding = embedding.flatten() if embedding.ndim > 1 else embedding
    return DummySourceFace(embedding)


class ModelManager:
    def __init__(self, run_config: RunConfig):
        self.cfg = run_config
        self.lock = threading.RLock()
        self.initialized = False
        self.closed = False

        self.available_ort_providers = set()
        self.app = None
        self.swapper = None
        self.source_face = None
        self.restore_session = None
        self.restore_inputs = None
        self.restore_outputs = None
        self.restore_weight_input_name = None
        self.parser_session = None
        self.parser_inputs = None

    def _cache_dir_for_stage(self, stage: str) -> str:
        mapping = {
            "detect": self.cfg.trt_cache_detect_dir,
            "swap": self.cfg.trt_cache_swap_dir,
            "restore": self.cfg.trt_cache_restore_dir,
            "parse": self.cfg.trt_cache_parser_dir,
        }
        return mapping.get(stage, self.cfg.trt_cache_dir)

    def _ensure_cache_dirs(self) -> None:
        os.makedirs(self.cfg.trt_cache_dir, exist_ok=True)
        os.makedirs(self.cfg.trt_cache_detect_dir, exist_ok=True)
        os.makedirs(self.cfg.trt_cache_swap_dir, exist_ok=True)
        os.makedirs(self.cfg.trt_cache_restore_dir, exist_ok=True)
        os.makedirs(self.cfg.trt_cache_parser_dir, exist_ok=True)

    def _emit_pipeline_banner(self):
        format_text = "IMAGE" if self.cfg.format_is_image else "VIDEO"
        status_list = []
        if self.cfg.enable_swapper:
            status_list.append("swapper=on")
        if self.cfg.enable_restore:
            status_list.append("restore=on")
        if self.cfg.enable_parser:
            status_list.append("parser=on")
        pipeline_status = " | ".join(status_list) if status_list else "all_off"

        ui_print(
            f"\n[Target Face]: {self.cfg.face_name} | [Format]: {format_text}",
            f"\n[Target Face]: {self.cfg.face_name} | [Format]: {format_text}",
        )
        ui_print(f"[Pipeline Status]: {pipeline_status}", f"[Pipeline Status]: {pipeline_status}")
        ui_print(
            "[Providers]: "
            f"default={self.cfg.provider_policy.default} "
            f"detect={self.cfg.provider_policy.detect} "
            f"swap={self.cfg.provider_policy.swap} "
            f"restore={self.cfg.provider_policy.restore} "
            f"parse={self.cfg.provider_policy.parse}",
            "[Providers]: "
            f"default={self.cfg.provider_policy.default} "
            f"detect={self.cfg.provider_policy.detect} "
            f"swap={self.cfg.provider_policy.swap} "
            f"restore={self.cfg.provider_policy.restore} "
            f"parse={self.cfg.provider_policy.parse}",
        )
        ui_print("==========================================")

    def _get_providers(self, cache_prefix: str, stage: str, enable_fp16: bool = True):
        requested = self.cfg.provider_policy.for_stage(stage)
        resolved = resolve_provider(requested, self.available_ort_providers)
        logger.info(
            "provider_selected",
            extra={
                "stage": stage,
                "requested": resolved.requested,
                "selected": resolved.selected,
                "reason": resolved.reason,
            },
        )
        ui_print(
            f"[Provider:{stage}] req={resolved.requested} sel={resolved.selected} reason={resolved.reason}",
            f"[Provider:{stage}] req={resolved.requested} sel={resolved.selected} reason={resolved.reason}",
        )
        return build_ort_providers(
            selected=resolved.selected,
            cache_prefix=cache_prefix,
            trt_cache_dir=self._cache_dir_for_stage(stage),
            enable_fp16=enable_fp16,
        )

    def initialize(self) -> "ModelManager":
        with self.lock:
            if self.initialized:
                return self
            if self.closed:
                raise ModelInitError("ModelManager is closed and cannot be re-initialized")

            try:
                onnxruntime.set_default_logger_severity(3)
                self.available_ort_providers = set(onnxruntime.get_available_providers())
            except Exception:
                self.available_ort_providers = set()

            self._ensure_cache_dirs()
            self._emit_pipeline_banner()
            try:
                ui_print("\nLoading Face Analysis...", "\nLoading Face Analysis...")
                with suppress_stdout():
                    self.app = FaceAnalysis(
                        name="buffalo_l",
                        root=self.cfg.insightface_root,
                        allowed_modules=["detection"],
                        providers=self._get_providers("face_detect", "detect", True),
                    )
                    self.app.prepare(ctx_id=0, det_size=(640, 640))

                if self.cfg.enable_swapper:
                    ui_print("Loading Inswapper_128...", "Loading Inswapper_128...")
                    with suppress_stdout():
                        self.swapper = insightface.model_zoo.get_model(
                            self.cfg.swapper_model,
                            providers=self._get_providers("face_swap", "swap", True),
                        )
                        self.source_face = load_source_face(self.cfg.source_face_path)

                if self.cfg.enable_restore:
                    ui_print(
                        f"Loading {self.cfg.restore_model_name}...",
                        f"Loading {self.cfg.restore_model_name}...",
                    )
                    with suppress_stdout():
                        self.restore_session = onnxruntime.InferenceSession(
                            self.cfg.restore_model_path,
                            providers=self._get_providers(
                                f"restore_{self.cfg.restore_size}",
                                "restore",
                                False,
                            ),
                        )
                        restore_input_defs = self.restore_session.get_inputs()
                        self.restore_inputs = restore_input_defs[0].name
                        self.restore_outputs = self.restore_session.get_outputs()[0].name
                        for inp in restore_input_defs[1:]:
                            shape = inp.shape
                            flat_size = 1
                            if isinstance(shape, (list, tuple)):
                                for dim in shape:
                                    if isinstance(dim, int) and dim > 0:
                                        flat_size *= dim
                            if flat_size == 1:
                                self.restore_weight_input_name = inp.name
                                break

                if self.cfg.enable_parser:
                    parser_model_name = os.path.basename(self.cfg.parser_model)
                    ui_print(f"Loading {parser_model_name}...", f"Loading {parser_model_name}...")
                    with suppress_stdout():
                        self.parser_session = onnxruntime.InferenceSession(
                            self.cfg.parser_model,
                            providers=self._get_providers("face_parser", "parse", True),
                        )
                        self.parser_inputs = self.parser_session.get_inputs()[0].name

                self.initialized = True
                ui_print("All models loaded successfully!\n", "All models loaded successfully!\n")
                return self
            except Exception as exc:
                self._teardown_no_raise()
                raise ModelInitError(str(exc)) from exc

    def _dispose_session(self, session):
        if session is None:
            return
        try:
            # Best effort: flush profiling if enabled.
            if hasattr(session, "end_profiling"):
                session.end_profiling()
        except Exception:
            pass
        try:
            del session
        except Exception:
            pass

    def _dispose_component(self, component, seen_ids=None):
        if component is None:
            return
        if seen_ids is None:
            seen_ids = set()
        comp_id = id(component)
        if comp_id in seen_ids:
            return
        seen_ids.add(comp_id)

        if isinstance(component, onnxruntime.InferenceSession):
            self._dispose_session(component)
            return

        for attr in ("session", "_sess", "sess"):
            try:
                maybe_session = getattr(component, attr, None)
            except Exception:
                maybe_session = None
            if maybe_session is not None:
                self._dispose_component(maybe_session, seen_ids)

        try:
            model_registry = getattr(component, "models", None)
        except Exception:
            model_registry = None

        if isinstance(model_registry, dict):
            for child in model_registry.values():
                self._dispose_component(child, seen_ids)
        elif isinstance(model_registry, (list, tuple)):
            for child in model_registry:
                self._dispose_component(child, seen_ids)

        close_fn = getattr(component, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def _teardown_no_raise(self):
        self._dispose_component(self.restore_session)
        self._dispose_component(self.parser_session)
        self._dispose_component(self.swapper)
        self._dispose_component(self.app)
        self.restore_session = None
        self.parser_session = None
        self.restore_inputs = None
        self.restore_outputs = None
        self.restore_weight_input_name = None
        self.parser_inputs = None
        self.swapper = None
        self.source_face = None
        self.app = None
        gc.collect()

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self._teardown_no_raise()
            self.initialized = False
            self.closed = True

    def _require_ready(self):
        if not self.initialized or self.closed:
            raise ModelInitError("ModelManager is not initialized")

    def detect_faces(self, frame):
        self._require_ready()
        return self.app.get(frame)

    def infer_swap(self, frame, face, source=None, paste_back=True):
        self._require_ready()
        if self.swapper is None:
            return frame
        src = source if source is not None else self.source_face
        return self.swapper.get(frame, face, src, paste_back=paste_back)

    def infer_restore(self, input_face, restore_weight=None):
        self._require_ready()
        if self.restore_session is None:
            return None
        feed = {self.restore_inputs: input_face}
        if self.restore_weight_input_name and restore_weight is not None:
            feed[self.restore_weight_input_name] = np.array([restore_weight], dtype=np.float32)
        return self.restore_session.run([self.restore_outputs], feed)[0]

    def infer_parse(self, input_tensor):
        self._require_ready()
        if self.parser_session is None:
            return None
        return self.parser_session.run(None, {self.parser_inputs: input_tensor})[0]


def init_models(ctx: RuntimeContext) -> ModelManager:
    manager = ModelManager(ctx.config).initialize()
    ctx.models.manager = manager
    ctx.models.app = manager.app
    ctx.models.swapper = manager.swapper
    ctx.models.restore_session = manager.restore_session
    ctx.models.parser_session = manager.parser_session
    return manager
