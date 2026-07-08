import unittest

from core.provider_policy import build_ort_providers, resolve_provider


class ProviderPolicyTests(unittest.TestCase):
    def test_trt_falls_back_to_cuda_when_trt_unavailable(self):
        resolution = resolve_provider("trt", {"CUDAExecutionProvider", "CPUExecutionProvider"})
        self.assertEqual(resolution.selected, "cuda")
        self.assertEqual(resolution.reason, "fallback_trt_to_cuda")

    def test_cuda_falls_back_to_cpu_when_cuda_unavailable(self):
        resolution = resolve_provider("cuda", {"CPUExecutionProvider"})
        self.assertEqual(resolution.selected, "cpu")
        self.assertEqual(resolution.reason, "fallback_cuda_to_cpu")

    def test_build_ort_providers_for_trt_includes_fallback_chain(self):
        providers = build_ort_providers(
            selected="trt",
            cache_prefix="unit",
            trt_cache_dir="cache",
            enable_fp16=True,
        )
        self.assertEqual(providers[1], "CUDAExecutionProvider")
        self.assertEqual(providers[2], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
