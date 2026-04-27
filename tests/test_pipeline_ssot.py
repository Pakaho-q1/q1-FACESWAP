import os
import tempfile
import unittest
from dataclasses import replace

from core.io_image import process_images
from core.io_video import process_videos
from core.orchestrator import (
    build_plan,
    build_video_output_paths,
    discover_image_work,
    discover_video_work,
    run_job,
)
from core.pipeline_state import create_pipeline_state
from core.errors import PipelineError
from core.types import ProviderPolicy, RunConfig, RuntimeContext


def _build_run_config(*, format_is_image: bool, input_path: str, output_dir: str) -> RunConfig:
    return RunConfig(
        face_name="unit_face",
        format_is_image=format_is_image,
        input_path=input_path,
        output_dir=output_dir,
        # Pipeline switches
        enable_swapper=False,
        enable_restore=False,
        enable_parser=False,
        # Swapper
        swapper_blend=0.7,
        # Restore
        restore_choice="1",
        restore_size=512,
        restore_model_name="GFPGANv1.4.onnx",
        restore_weight=0.7,
        restore_blend=0.7,
        # Parser
        parser_choice="1",
        parser_type="bisenet",
        parser_mask_blur=21,
        preserve_swap_eyes=True,
        # Runtime
        workers_per_stage=1,
        worker_queue_size=8,
        out_queue_size=8,
        tuner_mode="auto",
        file_sorting="date_modified_newest",
        gpu_target_util=90,
        high_watermark=12,
        low_watermark=4,
        switch_cooldown_s=0.35,
        max_retries=2,
        max_frames=5,
        skip_existing=False,
        output_suffix="_sfx",
        # Paths
        models_dir=output_dir,
        assets_dir=output_dir,
        ffmpeg_cmd="ffmpeg",
        insightface_root=output_dir,
        faces_dir=os.path.join(output_dir, "faces"),
        temp_audio_dir=os.path.join(output_dir, "temp_audio"),
        tensorrt_dir=os.path.join(output_dir, "TensorRT", "bin"),
        source_face_path="",
        swapper_model="",
        restore_model_path="",
        parser_model="",
        trt_cache_dir=os.path.join(output_dir, "trt_cache"),
        trt_cache_detect_dir=os.path.join(output_dir, "trt_cache", "trt_cache_detect"),
        trt_cache_swap_dir=os.path.join(output_dir, "trt_cache", "trt_cache_swap"),
        trt_cache_restore_dir=os.path.join(output_dir, "trt_cache", "trt_cache_restore"),
        trt_cache_parser_dir=os.path.join(output_dir, "trt_cache", "trt_cache_parser"),
        provider_policy=ProviderPolicy(
            default="cpu",
            detect="cpu",
            swap="cpu",
            restore="cpu",
            parse="cpu",
        ),
    )


class PipelineSingleSourceOfTruthTests(unittest.TestCase):
    def test_video_temp_audio_uses_explicit_temp_audio_dir(self):
        with tempfile.TemporaryDirectory() as td:
            temp_audio_dir = os.path.join(td, "assets", "temp_audio")
            paths = build_video_output_paths(
                output_dir=td,
                temp_audio_dir=temp_audio_dir,
                output_name="demo.mp4",
                index=1,
            )
            self.assertTrue(paths.temp_audio.startswith(temp_audio_dir))

    def test_process_images_requires_pipeline_state(self):
        with self.assertRaises(PipelineError):
            process_images(
                get_gpu_utilization=lambda: 0,
                pending_images=[],
                input_path=None,
                output_dir="",
                output_suffix=None,
            )

    def test_process_videos_requires_pipeline_state(self):
        with self.assertRaises(PipelineError):
            process_videos(
                get_gpu_utilization=lambda: 0,
                video_list=[],
                input_path=None,
                output_dir="",
                output_suffix=None,
                temp_audio_dir=None,
                max_frames=None,
                ffmpeg_cmd=None,
            )

    def test_process_images_requires_explicit_paths(self):
        cfg = _build_run_config(format_is_image=True, input_path="in", output_dir="out")
        state = create_pipeline_state(cfg)
        with self.assertRaises(ValueError):
            process_images(
                get_gpu_utilization=lambda: 0,
                pending_images=[],
                input_path=None,
                output_dir="",
                output_suffix=None,
                pipeline_state=state,
            )

    def test_process_videos_requires_explicit_paths(self):
        cfg = _build_run_config(format_is_image=False, input_path="in", output_dir="out")
        state = create_pipeline_state(cfg)
        with self.assertRaises(ValueError):
            process_videos(
                get_gpu_utilization=lambda: 0,
                video_list=[],
                input_path=None,
                output_dir="",
                output_suffix=None,
                temp_audio_dir=None,
                max_frames=None,
                ffmpeg_cmd=None,
                pipeline_state=state,
            )

    def test_image_work_respects_name_desc_sorting(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            out_dir = os.path.join(td, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            for name in ["a.jpg", "c.jpg", "b.jpg"]:
                with open(os.path.join(in_dir, name), "wb") as f:
                    f.write(b"x")
            cfg = _build_run_config(format_is_image=True, input_path=in_dir, output_dir=out_dir)
            cfg = replace(cfg, file_sorting="name_za")
            plan = build_plan(RuntimeContext(config=cfg))
            items = discover_image_work(plan)
            self.assertEqual([item.filename for item in items], ["c.jpg", "b.jpg", "a.jpg"])

    def test_video_work_respects_size_desc_sorting(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            out_dir = os.path.join(td, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            sizes = {"a.mp4": 1, "b.mp4": 3, "c.mp4": 2}
            for name, size in sizes.items():
                with open(os.path.join(in_dir, name), "wb") as f:
                    f.write(b"x" * size)
            cfg = _build_run_config(format_is_image=False, input_path=in_dir, output_dir=out_dir)
            cfg = replace(cfg, file_sorting="size_largest_smallest")
            plan = build_plan(RuntimeContext(config=cfg))
            items = discover_video_work(plan)
            self.assertEqual([item.filename for item in items], ["b.mp4", "c.mp4", "a.mp4"])

    def test_orchestrator_passes_explicit_image_runtime_args(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            out_dir = os.path.join(td, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            image_name = "a.jpg"
            with open(os.path.join(in_dir, image_name), "wb") as f:
                f.write(b"stub")

            cfg = _build_run_config(format_is_image=True, input_path=in_dir, output_dir=out_dir)
            ctx = RuntimeContext(config=cfg)
            state = create_pipeline_state(cfg)

            capture = {}

            def fake_process_images(*args, **kwargs):
                capture["pending_images"] = list(kwargs["pending_images"])
                capture["input_path"] = kwargs["input_path"]
                capture["output_dir"] = kwargs["output_dir"]
                capture["output_suffix"] = kwargs["output_suffix"]
                for item_id in kwargs["pending_images"]:
                    kwargs["on_item_start"](item_id)
                    kwargs["on_item_result"](item_id, None)

            def fake_process_videos(*args, **kwargs):
                self.fail("video path should not be used in image plan")

            run_job(
                ctx=ctx,
                process_images_fn=fake_process_images,
                process_videos_fn=fake_process_videos,
                get_gpu_utilization=lambda: 0,
                runtime_ui=None,
                pipeline_state=state,
            )

            self.assertEqual(capture["pending_images"], [image_name])
            self.assertEqual(capture["input_path"], in_dir)
            self.assertEqual(capture["output_dir"], out_dir)
            self.assertEqual(capture["output_suffix"], "_sfx")

    def test_orchestrator_passes_explicit_video_runtime_args(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            out_dir = os.path.join(td, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            video_name = "a.mp4"
            with open(os.path.join(in_dir, video_name), "wb") as f:
                f.write(b"stub")

            cfg = _build_run_config(format_is_image=False, input_path=in_dir, output_dir=out_dir)
            ctx = RuntimeContext(config=cfg)
            state = create_pipeline_state(cfg)

            capture = {}

            def fake_process_images(*args, **kwargs):
                self.fail("image path should not be used in video plan")

            def fake_process_videos(*args, **kwargs):
                capture["video_list"] = list(kwargs["video_list"])
                capture["input_path"] = kwargs["input_path"]
                capture["output_dir"] = kwargs["output_dir"]
                capture["output_suffix"] = kwargs["output_suffix"]
                capture["temp_audio_dir"] = kwargs["temp_audio_dir"]
                capture["max_frames"] = kwargs["max_frames"]
                capture["ffmpeg_cmd"] = kwargs["ffmpeg_cmd"]

            run_job(
                ctx=ctx,
                process_images_fn=fake_process_images,
                process_videos_fn=fake_process_videos,
                get_gpu_utilization=lambda: 0,
                runtime_ui=None,
                pipeline_state=state,
            )

            self.assertEqual(capture["video_list"], [(video_name, 4)])
            self.assertEqual(capture["input_path"], in_dir)
            self.assertEqual(capture["output_dir"], out_dir)
            self.assertEqual(capture["output_suffix"], "_sfx")
            self.assertEqual(capture["temp_audio_dir"], os.path.join(out_dir, "temp_audio"))
            self.assertEqual(capture["max_frames"], 5)
            self.assertEqual(capture["ffmpeg_cmd"], "ffmpeg")

    def test_orchestrator_emits_item_events(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = os.path.join(td, "in")
            out_dir = os.path.join(td, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            image_name = "evt.jpg"
            with open(os.path.join(in_dir, image_name), "wb") as f:
                f.write(b"stub")

            cfg = _build_run_config(format_is_image=True, input_path=in_dir, output_dir=out_dir)
            ctx = RuntimeContext(config=cfg)
            events = []
            ctx.hooks.on_event = lambda name, payload: events.append((name, dict(payload)))
            state = create_pipeline_state(cfg)

            def fake_process_images(*args, **kwargs):
                for item_id in kwargs["pending_images"]:
                    kwargs["on_item_start"](item_id)
                    kwargs["on_item_result"](item_id, None)

            run_job(
                ctx=ctx,
                process_images_fn=fake_process_images,
                process_videos_fn=lambda *a, **k: None,
                get_gpu_utilization=lambda: 0,
                runtime_ui=None,
                pipeline_state=state,
            )

            self.assertTrue(any(name == "item_started" for name, _ in events))
            self.assertTrue(any(name == "item_failed" for name, _ in events))

    def test_proc_cfg_is_frozen_and_decoupled(self):
        """ProcessorConfig must be immutable and not reference global config."""
        with tempfile.TemporaryDirectory() as td:
            cfg = _build_run_config(format_is_image=True, input_path=td, output_dir=td)
            state = create_pipeline_state(cfg)
            proc_cfg = state.proc_cfg

            # Frozen dataclass — mutation must raise
            with self.assertRaises((AttributeError, TypeError)):
                proc_cfg.swapper_blend = 0.99  # type: ignore[misc]

            # Values must reflect RunConfig, not some global default
            self.assertEqual(proc_cfg.swapper_blend, cfg.swapper_blend)
            self.assertEqual(proc_cfg.restore_blend, cfg.restore_blend)
            self.assertEqual(proc_cfg.parser_mask_blur, cfg.parser_mask_blur)

    def test_abort_event_unblocks_image_writer(self):
        """image_writer must exit when abort_event is set instead of blocking."""
        import threading
        cfg = _build_run_config(format_is_image=True, input_path="/tmp", output_dir="/tmp")
        state = create_pipeline_state(cfg)

        results = []

        def run_writer():
            from core.io_image import image_writer
            image_writer(
                total_images=10,
                output_dir="/tmp",
                output_suffix="",
                pipeline_state=state,
            )
            results.append("exited")

        t = threading.Thread(target=run_writer)
        t.start()
        # Signal abort immediately — writer must not hang
        state.abort_event.set()
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive(), "image_writer did not exit after abort_event was set")
        self.assertEqual(results, ["exited"])


if __name__ == "__main__":
    unittest.main()
