from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union


@dataclass(frozen=True)
class ProviderResolution:
    requested: str
    selected: str
    reason: str


def _normalize(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"cpu", "cuda", "trt"}:
        return "trt"
    return normalized


def resolve_provider(requested: str, available_provider_names: Iterable[str]) -> ProviderResolution:
    req = _normalize(requested)
    available = set(available_provider_names)

    matrix = {
        "trt": ["trt", "cuda", "cpu"],
        "cuda": ["cuda", "cpu"],
        "cpu": ["cpu"],
    }
    ort_name = {
        "trt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    for candidate in matrix[req]:
        if ort_name[candidate] in available:
            if candidate == req:
                return ProviderResolution(requested=req, selected=candidate, reason="requested_available")
            return ProviderResolution(
                requested=req,
                selected=candidate,
                reason=f"fallback_{req}_to_{candidate}",
            )

    # Defensive fallback if ORT reports an unusual provider set.
    return ProviderResolution(requested=req, selected="cpu", reason="forced_cpu_fallback")


def build_ort_providers(
    selected: str,
    cache_prefix: str,
    trt_cache_dir: str,
    enable_fp16: bool,
) -> List[Union[str, Tuple[str, dict]]]:
    if selected == "cpu":
        return ["CPUExecutionProvider"]
    if selected == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
                "trt_fp16_enable": enable_fp16,
                "trt_engine_cache_prefix": cache_prefix,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
