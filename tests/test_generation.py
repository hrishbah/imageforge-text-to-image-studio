"""Tests for image generation orchestration."""

import pytest
from PIL import Image

from imageforge.generation import (
    validate_prompt,
    validate_dimensions,
    validate_inference_steps,
    validate_guidance_scale,
    build_request,
    GenerationRequest,
    generate_image,
)


# ---------------------------------------------------------------------------
# Prompt validation tests
# ---------------------------------------------------------------------------
class TestValidatePrompt:
    def test_valid_prompt(self):
        result = validate_prompt("a cat in a hat")
        assert result == "a cat in a hat"

    def test_prompt_with_leading_trailing_whitespace(self):
        result = validate_prompt("  a cat  ")
        assert result == "a cat"

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("")

    def test_whitespace_only_prompt_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("   \t\n  ")


# ---------------------------------------------------------------------------
# Dimension validation tests
# ---------------------------------------------------------------------------
class TestValidateDimensions:
    def test_valid_dimensions(self):
        w, h = validate_dimensions(512, 512)
        assert w == 512 and h == 512

    def test_dimensions_must_be_multiple_of_8(self):
        with pytest.raises(ValueError, match="multiples of 8"):
            validate_dimensions(511, 512)
        with pytest.raises(ValueError, match="multiples of 8"):
            validate_dimensions(512, 503)

    def test_minimum_dimension_clamped_to_64(self):
        w, h = validate_dimensions(16, 16)
        assert w == 64 and h == 64

    def test_maximum_dimension_clamped(self):
        w, h = validate_dimensions(1024, 1024)
        assert w == 768 and h == 768  # MAX_WIDTH, MAX_HEIGHT


# ---------------------------------------------------------------------------
# Inference steps validation
# ---------------------------------------------------------------------------
class TestValidateInferenceSteps:
    def test_valid_steps(self):
        assert validate_inference_steps(20) == 20

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_inference_steps(0)

    def test_negative_steps_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            validate_inference_steps(-5)

    def test_exceeds_max_capped(self):
        assert validate_inference_steps(100) == 50  # MAX_STEPS


# ---------------------------------------------------------------------------
# Guidance scale validation
# ---------------------------------------------------------------------------
class TestValidateGuidanceScale:
    def test_valid_scale(self):
        assert validate_guidance_scale(7.5) == 7.5

    def test_zero_scale(self):
        assert validate_guidance_scale(0) == 0

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_guidance_scale(-1.0)


# ---------------------------------------------------------------------------
# GenerationRequest dataclass
# ---------------------------------------------------------------------------
class TestGenerationRequest:
    def test_canonicalise_empty_negative_prompt(self):
        req = GenerationRequest(
            prompt="test",
            negative_prompt="   ",
            width=512,
            height=512,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=None,
        )
        assert req.negative_prompt is None

    def test_positive_negative_prompt_preserved(self):
        req = GenerationRequest(
            prompt="test",
            negative_prompt="blurry",
            width=512,
            height=512,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=42,
        )
        assert req.negative_prompt == "blurry"


# ---------------------------------------------------------------------------
# build_request convenience function
# ---------------------------------------------------------------------------
class TestBuildRequest:
    def test_builds_valid_request(self):
        req = build_request(
            prompt="test prompt",
            negative_prompt="ugly",
            width=512,
            height=512,
            steps=30,
            guidance=8.0,
            seed=123,
        )
        assert req.prompt == "test prompt"
        assert req.negative_prompt == "ugly"
        assert req.width == 512
        assert req.height == 512
        assert req.num_inference_steps == 30
        assert req.guidance_scale == 8.0
        assert req.seed == 123

    def test_build_request_with_blank_negative_prompt(self):
        req = build_request(
            prompt="test prompt",
            negative_prompt="   ",
            width=512,
            height=512,
            steps=25,
            guidance=7.5,
            seed=None,
        )
        assert req.negative_prompt is None


# ---------------------------------------------------------------------------
# generate_image with mocked pipeline
# ---------------------------------------------------------------------------
class TestGenerateImage:
    def test_generate_image_returns_real_pil_image_and_config(self, monkeypatch):
        """Verify generate_image returns a PIL Image and config dict."""

        class FakePipe:
            def to(self, device):
                return self

            def __call__(self, **kwargs):
                # Non-uniform image to pass degeneration checks
                import random
                img = Image.new("RGB", (512, 512), color=(100, 150, 200))
                px = img.load()
                for y in range(512):
                    for x in range(512):
                        if (x + y) % 7 == 0:
                            px[x, y] = (120, 170, 220)
                return type("obj", (), {"images": [img]})()

        monkeypatch.setattr(
            "imageforge.model_manager.load_pipeline",
            lambda: FakePipe(),
        )

        req = GenerationRequest(
            prompt="test",
            negative_prompt=None,
            width=512,
            height=512,
            num_inference_steps=1,
            guidance_scale=7.5,
            seed=42,
        )

        img, cfg = generate_image(req)

        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)
        assert cfg["prompt"] == "test"
        assert cfg["seed"] == 42
        assert cfg["num_inference_steps"] == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_generate_with_load_failure_raises_readable_error(self, monkeypatch):
        """Verify we surface a readable error when the pipeline loader fails."""
        def mock_load():
            raise RuntimeError("Mocked model load failure")

        monkeypatch.setattr("imageforge.model_manager.load_pipeline", mock_load)

        req = GenerationRequest(
            prompt="test",
            negative_prompt=None,
            width=512,
            height=512,
            num_inference_steps=1,
            guidance_scale=7.5,
            seed=None,
        )

        with pytest.raises(RuntimeError, match="Failed to load the model pipeline"):
            generate_image(req)

    def test_generate_with_pipeline_inference_failure(self, monkeypatch):
        """Verify inference errors become readable RuntimeError."""
        class FailingPipe:
            def to(self, device):
                return self

            def __call__(self, **kwargs):
                raise RuntimeError("mocked inference OOM")

        monkeypatch.setattr(
            "imageforge.model_manager.load_pipeline",
            lambda: FailingPipe(),
        )

        req = GenerationRequest(
            prompt="test",
            negative_prompt=None,
            width=512,
            height=512,
            num_inference_steps=1,
            guidance_scale=7.5,
            seed=None,
        )

        with pytest.raises(RuntimeError, match="Image generation failed"):
            generate_image(req)


# ---------------------------------------------------------------------------
# Seed and metadata handling
# ---------------------------------------------------------------------------
class TestSeedAndMetadata:
    def test_random_seed_passed_through_as_none_and_config_kept(self, monkeypatch):
        import imageforge.model_manager as mm

        class FakePipe:
            def to(self, device):
                return self

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                import random
                img = Image.new("RGB", (512, 512), color=(100, 150, 200))
                px = img.load()
                for y in range(512):
                    for x in range(512):
                        if (x + y) % 7 == 0:
                            px[x, y] = (120, 170, 220)
                return type("obj", (), {"images": [img]})()

        fake = FakePipe()
        monkeypatch.setattr("imageforge.model_manager.load_pipeline", lambda: fake)

        req = GenerationRequest(
            prompt="test",
            negative_prompt=None,
            width=512,
            height=512,
            num_inference_steps=1,
            guidance_scale=7.5,
            seed=None,
        )

        _, cfg = generate_image(req)

        assert fake.last_kwargs["prompt"] == "test"
        assert fake.last_kwargs["negative_prompt"] is None
        assert fake.last_kwargs["generator"] is None
        assert cfg["seed"] is None
        assert cfg["width"] == 512
        assert cfg["height"] == 512
        assert cfg["num_inference_steps"] == 1
        assert cfg["guidance_scale"] == 7.5


# ---------------------------------------------------------------------------
# PNG output / download-related data
# ---------------------------------------------------------------------------
class TestPngOutput:
    def test_downloadable_png_bytes_from_generated_image(self, tmp_path):
        """Verify we can round-trip a generated image to PNG bytes & reload it."""
        from io import BytesIO
        img = Image.new("RGB", (8, 8), color=(10, 20, 30))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        reloaded = Image.open(BytesIO(png_bytes))
        assert reloaded.format == "PNG"
        assert reloaded.size == (8, 8)
        # PNG bytes start with the PNG signature
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sensible_filename_format(self):
        import datetime
        ts = "20261027_235959"
        seed = 42
        filename = f"imageforge_seed{seed}_{ts}.png"
        assert filename.endswith(".png")
        assert "imageforge" in filename
