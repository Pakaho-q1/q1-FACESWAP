from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from gui.services.output_index_service import OutputIndexService


class OutputIndexServiceTests(unittest.TestCase):
    def _wait_index_ready(
        self,
        service: OutputIndexService,
        output_dir: Path,
        *,
        timeout_s: float = 8.0,
        min_total: int = 1,
        exact_total: int | None = None,
    ) -> dict[str, object]:
        end = time.time() + timeout_s
        payload: dict[str, object] = {}
        while time.time() < end:
            payload = service.get_page(str(output_dir), page=1, page_size=50, force_refresh=True)
            total = int(payload.get("total", 0))
            if exact_total is not None and total == exact_total:
                return payload
            if exact_total is None and total >= min_total:
                return payload
            time.sleep(0.05)
        return payload

    def test_index_pagination_50_per_page(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            db_path = root / "runtime" / "idx.sqlite3"
            for i in range(120):
                (out_dir / f"img_{i:04d}.jpg").write_bytes(b"x")
            service = OutputIndexService(str(db_path), stale_seconds=1.0)
            try:
                first = self._wait_index_ready(service, out_dir, min_total=120)
                self.assertEqual(int(first.get("total", 0)), 120)
                self.assertEqual(len(list(first.get("rows", []))), 50)
                page3 = service.get_page(str(out_dir), page=3, page_size=50, force_refresh=False)
                self.assertEqual(len(list(page3.get("rows", []))), 20)
            finally:
                service.shutdown()

    def test_index_refresh_tracks_delete_and_add(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            db_path = root / "runtime" / "idx.sqlite3"
            for i in range(10):
                (out_dir / f"v_{i:04d}.mp4").write_bytes(b"video")
            service = OutputIndexService(str(db_path), stale_seconds=1.0)
            try:
                payload = self._wait_index_ready(service, out_dir, min_total=10)
                self.assertEqual(int(payload.get("total", 0)), 10)
                (out_dir / "v_0000.mp4").unlink()
                (out_dir / "v_0001.mp4").unlink()
                (out_dir / "v_new.mp4").write_bytes(b"video")
                refreshed = self._wait_index_ready(service, out_dir, exact_total=9)
                self.assertEqual(int(refreshed.get("total", 0)), 9)
            finally:
                service.shutdown()


if __name__ == "__main__":
    unittest.main()
