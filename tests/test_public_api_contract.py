import unittest

import core
from core.errors import ConfigError
from core.types import ProviderPolicy, RunConfig, validate_run_config


class PublicApiContractTests(unittest.TestCase):
    def test_public_api_exports(self):
        self.assertTrue(callable(core.run_pipeline))
        self.assertTrue(callable(core.run_image_job))
        self.assertTrue(callable(core.run_video_job))
        self.assertTrue(callable(core.resume_pipeline_job))
        self.assertTrue(hasattr(core, "RunConfig"))
        self.assertTrue(hasattr(core, "ConfigError"))

    def test_validate_run_config_rejects_invalid(self):
        cfg = RunConfig(
            face_name="",
            format_is_image=True,
            input_path="",
            output_dir="",
            enable_swapper=False,
            enable_restore=False,
            enable_parser=False,
            swapper_blend=1.2,
            restore_choice="1",
            restore_size=512,
            restore_model_name="GFPGANv1.4.onnx",
            restore_weight=1.2,
            restore_blend=1.2,
            parser_choice="1",
            parser_type="bisenet",
            parser_mask_blur=20,
            preserve_swap_eyes=True,
            workers_per_stage=0,
            worker_queue_size=1,
            out_queue_size=1,
            tuner_mode="auto",
            file_sorting="date_modified_newest",
            gpu_target_util=10,
            high_watermark=1,
            low_watermark=2,
            switch_cooldown_s=0.1,
            max_retries=0,
            max_frames=0,
            skip_existing=False,
            output_suffix="",
            models_dir="",
            assets_dir="",
            ffmpeg_cmd="",
            insightface_root="",
            faces_dir="",
            temp_audio_dir="",
            tensorrt_dir="",
            source_face_path="",
            swapper_model="",
            restore_model_path="",
            parser_model="",
            trt_cache_dir="",
            trt_cache_detect_dir="",
            trt_cache_swap_dir="",
            trt_cache_restore_dir="",
            trt_cache_parser_dir="",
            provider_policy=ProviderPolicy(
                default="cpu",
                detect="cpu",
                swap="cpu",
                restore="cpu",
                parse="cpu",
            ),
        )
        with self.assertRaises(ConfigError):
            validate_run_config(cfg)


if __name__ == "__main__":
    unittest.main()
