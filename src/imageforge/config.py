"""ImageForge configuration constants."""

MODEL_ID = "segmind/tiny-sd"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
MAX_WIDTH = 768
MAX_HEIGHT = 768
DEFAULT_STEPS = 25
MAX_STEPS = 50
DEFAULT_GUIDANCE = 7.5

OUTPUT_DIR = "outputs"

# Presets applied to the prompt text area
PRESETS = {
    "Photorealistic": "photorealistic, highly detailed, sharp focus, 8k, professional lighting",
    "Digital Art": "digital illustration, vibrant colors, stylized, fantasy art",
    "Watercolor": "watercolor painting, soft brush strokes, dreamy atmosphere",
}