import os
import unittest
from core.config import _resolve_input_face, _resolve_input_target
from core.orchestrator import JobPlan, discover_image_work, discover_video_work

class SmartInputsTests(unittest.TestCase):
    def test_resolve_input_face_bare_name(self):
        faces_dir = "D:\\mock\\faces"
        face_name, src_path, is_image = _resolve_input_face("alice", faces_dir)
        self.assertEqual(face_name, "alice")
        self.assertEqual(src_path, "D:\\mock\\faces\\alice.safetensors")
        self.assertFalse(is_image)

    def test_resolve_input_face_safetensors_path(self):
        faces_dir = "D:\\mock\\faces"
        face_name, src_path, is_image = _resolve_input_face("D:\\custom\\bob.safetensors", faces_dir)
        self.assertEqual(face_name, "bob")
        self.assertEqual(src_path, "D:\\custom\\bob.safetensors")
        self.assertFalse(is_image)

    def test_resolve_input_face_image_path(self):
        faces_dir = "D:\\mock\\faces"
        face_name, src_path, is_image = _resolve_input_face("C:\\images\\charlie.png", faces_dir)
        self.assertEqual(face_name, "charlie")
        self.assertEqual(src_path, "C:\\images\\charlie.png")
        self.assertTrue(is_image)

    def test_resolve_input_target_video(self):
        # Even if the file does not exist, check that it resolves based on extension
        input_dir, single_file, fmt_override = _resolve_input_target("D:\\inputs\\clip.mp4")
        self.assertEqual(input_dir, "D:\\inputs")
        self.assertEqual(single_file, "clip.mp4")
        self.assertFalse(fmt_override)

    def test_resolve_input_target_image(self):
        input_dir, single_file, fmt_override = _resolve_input_target("D:\\inputs\\photo.jpg")
        self.assertEqual(input_dir, "D:\\inputs")
        self.assertEqual(single_file, "photo.jpg")
        self.assertTrue(fmt_override)

    def test_discover_work_respects_single_file(self):
        # Create a mock JobPlan with input_single_file
        plan = JobPlan(
            kind="image",
            input_path="mock_in",
            output_path="mock_out",
            output_suffix="_swapped",
            models_dir="mock_models",
            temp_audio_dir="mock_temp",
            ffmpeg_cmd="ffmpeg",
            file_sorting="name_az",
            skip_existing=False,
            max_frames=0,
            face_name="alice",
            enable_swapper=True,
            enable_restore=False,
            enable_parser=False,
            workers_per_stage=1,
            input_single_file="test_image.jpg"
        )
        
        work_items = discover_image_work(plan)
        self.assertEqual(len(work_items), 1)
        self.assertEqual(work_items[0].filename, "test_image.jpg")
        self.assertEqual(work_items[0].output_name, "test_image_swapped.jpg")

if __name__ == "__main__":
    unittest.main()
