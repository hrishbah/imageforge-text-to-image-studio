"""Tests for model manager: device detection, pipeline loading/caching."""

import os
from unittest import mock

import pytest

import imageforge.model_manager as mm


# ---------------------------------------------------------------------------
# Device selection logic
# ---------------------------------------------------------------------------
class TestDeviceSelection:
    def test_device_is_one_of_supported(self):
        assert mm.DEVICE in ("mps", "cuda", "cpu")

    def test_dtype_compatible_with_device(self):
        import torch
        if mm.DEVICE == "mps":
            assert mm.TORCH_DTYPE is torch.float32  # float32 for MPS stability
        elif mm.DEVICE == "cuda":
            assert mm.TORCH_DTYPE is torch.float16
        else:
            assert mm.TORCH_DTYPE is torch.float32

    def test_device_message_is_nonempty(self):
        assert isinstance(mm.DEVICE_MESSAGE, str) and mm.DEVICE_MESSAGE

    def test_device_info_dict(self):
        info = mm.get_device_info()
        assert info["device"] == mm.DEVICE
        assert info["message"] == mm.DEVICE_MESSAGE
        assert info["torch_dtype"] == str(mm.TORCH_DTYPE)


# ---------------------------------------------------------------------------
# Mocked hardware availability for device logic
# ---------------------------------------------------------------------------
class TestMockedDeviceSelection:
    def test_cpu_fallback_when_no_mps_and_no_cuda(self, monkeypatch):
        import importlib
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

        importlib.reload(mm)
        assert mm.DEVICE == "cpu"

    def test_mps_preferred_over_cpu(self, monkeypatch):
        import importlib
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

        importlib.reload(mm)
        assert mm.DEVICE == "mps"

    def test_cuda_preferred_when_available(self, monkeypatch):
        import importlib
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)

        importlib.reload(mm)
        assert mm.DEVICE == "cuda"


# ---------------------------------------------------------------------------
# Pipeline loading / caching with mocked loader
# ---------------------------------------------------------------------------
class TestPipelineLoading:
    def test_load_pipeline_uses_stable_diffusion_pipeline(self, monkeypatch):
        """Verify StableDiffusionPipeline.from_pretrained is called once."""
        calls = []

        class FakePipe:
            def __init__(self):
                calls.append("created")

            def to(self, device):
                return self

        def fake_from_pretrained(model_id, **kwargs):
            assert model_id == "segmind/tiny-sd"
            return FakePipe()

        monkeypatch.setattr(mm.StableDiffusionPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
        calls.clear()

        pipe = mm.load_pipeline()
        pipe2 = mm.load_pipeline()

        assert pipe is pipe2  # cached
        assert len(calls) == 1  # loaded once

    def test_is_pipeline_loaded_before_and_after(self, monkeypatch):
        import imageforge.model_manager as m

        # reset state
        m._PIPELINE = None
        assert m.is_pipeline_loaded() is False

        class FakePipe:
            def to(self, device):
                return self

        monkeypatch.setattr(m.StableDiffusionPipeline, "from_pretrained", staticmethod(lambda *a, **k: FakePipe()))
        m.load_pipeline()
        assert m.is_pipeline_loaded() is True

    def test_load_pipeline_failure_surfaces_readable_error(self, monkeypatch):
        """Verify a readable RuntimeError when model loading fails."""
        import imageforge.model_manager as m

        m._PIPELINE = None

        monkeypatch.setattr(
            m.StableDiffusionPipeline,
            "from_pretrained",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mocked download failure"))),
        )

        with pytest.raises(RuntimeError, match="Failed to load the model"):
            m.load_pipeline()
        assert m.is_pipeline_loaded() is False

    def test_has_cache_returns_bool(self):
        result = mm.has_cache()
        assert isinstance(result, bool)

    def test_has_cache_runs_without_raising(self):
        """has_cache must run regardless of cache presence, no personal path leaks."""
        mm.has_cache()
