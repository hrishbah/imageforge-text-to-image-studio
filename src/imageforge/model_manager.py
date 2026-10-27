"""Model loading, caching, device detection with Apple Silicon/MPS support.

The pipeline is loaded lazily (only when :func:`load_pipeline` is first called),
cached for reuse across Streamlit reruns, and reused on every subsequent
generation request.
"""

import os

import torch

# Configure a project-local Hugging Face cache *before* importing diffusers so
# that downloaded weights live outside the user's home cache and can be excluded
# from Git via .gitignore. If the user has explicitly exported HF_HOME we keep it.
if "HF_HOME" not in os.environ:
    os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))

from diffusers import StableDiffusionPipeline  # noqa: E402

MODEL_ID = "segmind/tiny-sd"

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_MESSAGE = "Apple Silicon (MPS)"
    TORCH_DTYPE = torch.float32  # float32 for numerical stability on MPS
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_MESSAGE = "NVIDIA GPU (CUDA)"
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DEVICE_MESSAGE = "CPU (slow — performance will be limited)"
    TORCH_DTYPE = torch.float32


def supports_negative_prompt() -> bool:
    """Return True if the loaded pipeline supports a negative_prompt argument.

    The ``StableDiffusionPipeline`` exposes ``__call__`` with a ``negative_prompt``
    keyword, so on this model negative prompts are supported natively.
    """
    return True


def load_pipeline():
    """Load (and cache) the Stable Diffusion pipeline on the detected device.

    The first call downloads ~1 GB of weights if a local cache is absent.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        _PIPELINE = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
        )
        _PIPELINE.to(DEVICE)

        # Attention slicing reduces peak memory on MPS/CUDA. It is unsupported
        # on CPU but that path is memory-safe anyway.
        if DEVICE in ("cuda", "mps"):
            try:
                _PIPELINE.enable_attention_slicing()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        _PIPELINE = None
        raise RuntimeError(
            f"Failed to load the model '{MODEL_ID}'. "
            f"Ensure an internet connection is available for the first download "
            f"(~1 GB of weights) and check the error log. "
            f"Original error: {exc}"
        ) from exc

    return _PIPELINE


def is_pipeline_loaded() -> bool:
    """Report whether the pipeline is currently resident in memory."""
    return _PIPELINE is not None


# Module-level cache holder (created lazily inside load_pipeline)
_PIPELINE = None


def get_device_info():
    """Return a serialisable dictionary describing the runtime device."""
    return {
        "device": DEVICE,
        "message": DEVICE_MESSAGE,
        "torch_dtype": str(TORCH_DTYPE),
    }


def has_cache() -> bool:
    """Best-effort check for an existing local Hugging Face cache of the model."""
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidate = os.path.join(cache_root, "models--", "segmind--tiny-sd")
    return os.path.isdir(candidate)