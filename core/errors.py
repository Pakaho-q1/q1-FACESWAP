class FaceSwapError(Exception):
    """Base exception for library consumers."""


class ConfigError(FaceSwapError):
    """Raised when runtime configuration is invalid."""


class ModelInitError(FaceSwapError):
    """Raised when model initialization fails."""


class PipelineError(FaceSwapError):
    """Raised when pipeline orchestration fails."""


class RecoveryError(PipelineError):
    """Raised when recovery or resume paths fail."""
