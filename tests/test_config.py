"""Tests for configuration constants."""

from imageforge import config


def test_model_id():
    assert config.MODEL_ID == "segmind/tiny-sd"


def test_default_dimensions():
    assert config.DEFAULT_WIDTH == 512
    assert config.DEFAULT_HEIGHT == 512
    assert config.DEFAULT_WIDTH % 8 == 0


def test_max_dimensions():
    assert config.MAX_WIDTH == 768
    assert config.MAX_HEIGHT == 768
    assert config.MAX_WIDTH % 8 == 0


def test_step_bounds():
    assert config.DEFAULT_STEPS == 25
    assert config.MAX_STEPS == 50
    assert 1 <= config.DEFAULT_STEPS <= config.MAX_STEPS


def test_output_dir():
    assert config.OUTPUT_DIR == "outputs"


def test_no_absolute_paths():
    """Ensure no hardcoded absolute personal paths leak into config constants.

    We deliberately inspect only the public configuration values, not module
    dunders such as ``__file__`` (which legitimately carries the install path).
    """
    public_keys = {
        "MODEL_ID",
        "DEFAULT_WIDTH",
        "DEFAULT_HEIGHT",
        "MAX_WIDTH",
        "MAX_HEIGHT",
        "DEFAULT_STEPS",
        "MAX_STEPS",
        "DEFAULT_GUIDANCE",
        "OUTPUT_DIR",
        "PRESETS",
    }
    for key in public_keys:
        value = getattr(config, key)
        if isinstance(value, str):
            assert "/Users/" not in value, f"Absolute path found in {key}: {value}"
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str):
                    assert "/Users/" not in v, f"Absolute path found in {key}: {v}"