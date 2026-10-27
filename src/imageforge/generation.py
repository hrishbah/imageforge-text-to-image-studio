"""Image generation orchestration: validation and pipeline invocation."""

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image


@dataclass
class GenerationRequest:
    """Validated, typed container for image generation parameters."""

    prompt: str
    negative_prompt: Optional[str]  # None if user left it blank
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]  # None means random

    def __post_init__(self):
        # Canonicalise empty strings to None for negative_prompt
        if self.negative_prompt is not None and not self.negative_prompt.strip():
            self.negative_prompt = None


def validate_prompt(prompt: str) -> str:
    """Validate that prompt is non-empty and non-whitespace.

    Returns the stripped prompt on success.
    Raises :class:`ValueError` with a user-friendly message on failure.
    """
    stripped = prompt.strip()
    if not stripped:
        raise ValueError("Prompt cannot be empty. Please enter a description of the image you want to create.")
    return stripped


def validate_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Validate that dimensions are multiples of 8 and within supported bounds.

    Stable Diffusion requires dimensions divisible by 8 (latent space constraints).
    """
    from .config import MAX_WIDTH, MAX_HEIGHT, DEFAULT_WIDTH, DEFAULT_HEIGHT

    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")

    width = max(64, min(MAX_WIDTH, width))
    height = max(64, min(MAX_HEIGHT, height))

    return width, height


def validate_inference_steps(steps: int) -> int:
    """Cap inference steps to a memory-conscious range."""
    from .config import MAX_STEPS, DEFAULT_STEPS

    if steps < 1:
        raise ValueError("Inference steps must be at least 1.")
    return min(steps, MAX_STEPS)


def validate_guidance_scale(scale: float) -> float:
    """Validate guidance scale (classifier-free guidance weight)."""
    if scale < 0:
        raise ValueError("Guidance scale must be non-negative.")
    return scale


def build_request(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: Optional[int],
) -> GenerationRequest:
    """Build a validated :class:`GenerationRequest` from raw form inputs."""
    return GenerationRequest(
        prompt=validate_prompt(prompt),
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_inference_steps=validate_inference_steps(steps),
        guidance_scale=validate_guidance_scale(guidance),
        seed=seed,
    )


def _is_valid_image(image: Image.Image, min_mean: float = 2.0, max_uniform_ratio: float = 0.99) -> bool:
    """Reject clearly degenerate outputs (uniform black, zero-variance, etc.).

    Uses mean brightness and pixel-stdev heuristics rather than requiring
    perfect photographic quality.
    """
    try:
        import numpy as np
        arr = np.array(image.convert("L"))
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        # Near-black with almost no variance is a failed diffusion step
        if mean_val < min_mean and std_val < 1.0:
            return False
        # If >99% of pixels are identical, treat as uniform failure
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        max_bin = float(hist.max()) / float(arr.size)
        if max_bin > max_uniform_ratio:
            return False
        return True
    except Exception:
        # If analytics fail, prefer permissive pass rather than hidden failure
        return True


def generate_image(request: GenerationRequest):
    """Run the actual generation using the cached pipeline.

    Returns a tuple of (PIL.Image, generation_config_dict) or raises RuntimeError.
    """
    from .model_manager import load_pipeline, DEVICE

    try:
        pipe = load_pipeline()
    except Exception as exc:  # noqa: BLE001 - surface a clean error message
        raise RuntimeError(f"Failed to load the model pipeline: {exc}") from exc

    generator = None
    if request.seed is not None:
        import torch
        generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

    kwargs = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "generator": generator,
    }

    try:
        result = pipe(**kwargs)
        image: Image.Image = result.images[0]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    if not _is_valid_image(image):
        raise RuntimeError(
            "Image generation returned a uniform/black output. "
            "This usually indicates a numerical failure (e.g. float16 overflow) "
            "or scheduler mismatch. Try a different seed, fewer steps, or float32."
        )

    config = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
        "device": DEVICE,
    }

    return image, config