from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from gui.services.output_index_service import OutputIndexService


RUN_STRESS = os.environ.get("Q1_RUN_STRESS_TESTS", "").strip() == "1"


@unittest.skipUnless(RUN_STRESS, "Set Q1_RUN_STRESS_TESTS=1 to run stress tests")
class OutputIndexStressTests(unittest.TestCase):
    def _wait_total(
        self,
        service: OutputIndexService,
        output_dir: Path,
        expected_total: int,
        timeout_s: float = 60.0,
    ) -> dict[str, object]:
        end = time.time() + timeout_s
        payload: dict[str, object] = {}
        while time.time() < end:
            payload = service.get_page(str(output_dir), page=1, page_size=50, force_refresh=True)
            if int(payload.get("total", 0)) >= expected_total:
                return payload
            time.sleep(0.1)
        return payload

    def _run_case(self, count: int) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            db_path = root / "runtime" / "idx.sqlite3"
            for i in range(count):
                (out_dir / f"case_{i:06d}.jpg").write_bytes(b"x")
            service = OutputIndexService(str(db_path), stale_seconds=1.0, max_scan_seconds=180.0)
            try:
                payload = self._wait_total(service, out_dir, expected_total=count, timeout_s=180.0)
                self.assertEqual(int(payload.get("total", 0)), count)
                self.assertEqual(len(list(payload.get("rows", []))), 50)
            finally:
                service.shutdown()

    def test_stress_10k(self) -> None:
        self._run_case(10_000)

    def test_stress_50k(self) -> None:
        self._run_case(50_000)

    def test_stress_100k(self) -> None:
        self._run_case(100_000)


if __name__ == "__main__":
    unittest.main()
