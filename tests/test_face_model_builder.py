import os
import tempfile
import unittest

import cv2
import numpy as np
from safetensors.numpy import load_file

from core.face_model_builder import FaceModelBuildError, build_face_model


class _FakeExtractor:
    def __init__(self, mapping: dict[int, list[np.ndarray]]) -> None:
        self._mapping = mapping

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        marker = int(image_bgr[0, 0, 0])
        return list(self._mapping.get(marker, []))


def _write_marker_image(path: str, marker: int) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, :] = int(marker)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise RuntimeError(f"failed writing test image: {path}")


class FaceModelBuilderTests(unittest.TestCase):
    def test_build_face_model_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_dir = os.path.join(td, "input")
            faces_dir = os.path.join(td, "faces")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(faces_dir, exist_ok=True)

            _write_marker_image(os.path.join(input_dir, "a.png"), 10)
            _write_marker_image(os.path.join(input_dir, "b.png"), 11)
            _write_marker_image(os.path.join(input_dir, "c.png"), 12)

            mapping = {
                10: [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
                11: [np.array([0.9, 0.1, 0.0], dtype=np.float32)],
                12: [np.array([0.8, 0.2, 0.0], dtype=np.float32)],
            }
            result = build_face_model(
                face_name="unit_face",
                input_dir=input_dir,
                faces_dir=faces_dir,
                extractor=_FakeExtractor(mapping),
            )

            self.assertEqual(result.accepted_images, 3)
            self.assertTrue(result.output_path.endswith("unit_face.safetensors"))
            payload = load_file(result.output_path)
            self.assertIn("embedding", payload)
            emb = np.asarray(payload["embedding"], dtype=np.float32).reshape(-1)
            self.assertEqual(emb.size, 3)
            self.assertGreater(float(np.linalg.norm(emb)), 0.99)
            self.assertLessEqual(float(np.linalg.norm(emb)), 1.01)

    def test_build_face_model_requires_valid_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FaceModelBuildError):
                build_face_model(
                    face_name="../bad name",
                    input_dir=td,
                    faces_dir=td,
                    extractor=_FakeExtractor({}),
                )

    def test_build_face_model_handles_no_face_and_multi_face(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_dir = os.path.join(td, "input")
            faces_dir = os.path.join(td, "faces")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(faces_dir, exist_ok=True)

            _write_marker_image(os.path.join(input_dir, "no_face.png"), 21)
            _write_marker_image(os.path.join(input_dir, "multi_face.png"), 22)
            _write_marker_image(os.path.join(input_dir, "valid.png"), 23)

            mapping = {
                21: [],
                22: [
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                ],
                23: [np.array([0.7, 0.3], dtype=np.float32)],
            }
            result = build_face_model(
                face_name="unit_face2",
                input_dir=input_dir,
                faces_dir=faces_dir,
                extractor=_FakeExtractor(mapping),
                min_accepted_images=1,
            )
            self.assertEqual(result.scanned_images, 3)
            self.assertEqual(result.accepted_images, 1)
            self.assertEqual(result.skipped_no_face, 1)
            self.assertEqual(result.skipped_multi_face, 1)

    def test_build_face_model_overwrite_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_dir = os.path.join(td, "input")
            faces_dir = os.path.join(td, "faces")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(faces_dir, exist_ok=True)
            _write_marker_image(os.path.join(input_dir, "a.png"), 42)

            mapping = {42: [np.array([1.0, 0.0], dtype=np.float32)]}
            first = build_face_model(
                face_name="dup_face",
                input_dir=input_dir,
                faces_dir=faces_dir,
                extractor=_FakeExtractor(mapping),
            )
            self.assertTrue(os.path.isfile(first.output_path))

            with self.assertRaises(FaceModelBuildError):
                build_face_model(
                    face_name="dup_face",
                    input_dir=input_dir,
                    faces_dir=faces_dir,
                    extractor=_FakeExtractor(mapping),
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
