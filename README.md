# ImageForge — AI Text-to-Image Studio

A polished, portfolio-ready Streamlit application that turns text prompts into
images using the pretrained `segmind/tiny-sd` Stable Diffusion model — running
entirely on your Mac, with no training and no paid API keys required.

Built for **Prodigy InfoTech Generative AI Internship — Task 02: Image
Generation with Pre-trained Models**.

---

## Features

- **Multiline text prompt** input with example placeholder and example button
- **Optional negative prompt** (natively supported by `StableDiffusionPipeline`)
- **Generation controls**: dimensions (default 512 × 512), inference steps
  (default 25, capped at 50), guidance scale (default 7.5), and reproducible
  random/custom seed
- **Prompt presets**: photorealistic, digital art, watercolor
- **Generate button** with spinner and clear progress messages
- **Generation details panel**: prompt, model name, seed, dimensions, device,
  inference steps, guidance scale
- **PNG download** with sensible timestamps filenames
- **Local output storage**: generated PNGs + sidecar metadata under `outputs/`
- **First-run guidance**: explains pretrained model, one-time ~1 GB download,
  local inference, and performance expectations
- **Device detection**: Apple Silicon MPS → CUDA → CPU with dtype selection
- **Lazy model loading**: pipeline loaded only when you click Generate, then
  cached across Streamlit reruns
- **Memory protection**: attention slicing on MPS/CUDA, one image per request,
  no batching

---

## Screenshot

No screenshot is bundled with this repository. To generate one locally:

1. Build and run the app (see below).
2. Click **Generate** with any prompt.
3. Capture the browser window at desktop resolution (~1280×800) and save it to
   `images/README-screenshot.png` (optional).

The `images/` directory exists only to hold screenshots or UI assets if you
choose to add them later.

---

## Architecture & project tree

```
ImageForge/
├── app.py                      # Streamlit entry point
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── imageforge/
│       ├── __init__.py
│       ├── config.py           # Model ID, dimensions, limits, presets
│       ├── model_manager.py    # Lazy pipeline load, device detection
│       └── generation.py       # Input validation & generation orchestration
├── tests/
│   ├── __init__.py
│   ├── test_config.py          # Configuration constants
│   ├── test_generation.py      # Validation, mocked pipeline, PNG output
│   └── test_model_manager.py   # Device detection, mocked loader
├── outputs/                    # Generated PNGs + metadata (gitignored)
│   └── .gitkeep
└── images/                     # Optional UI screenshots (gitignored except .gitkeep)
    └── .gitkeep
```

### Modules

- `app.py` — Streamlit UI: page config, sidebar status, prompt/negative prompt,
  advanced settings expander, generate button, image display, download button,
  first-run guidance.
- `src/imageforge/config.py` — Constants: `MODEL_ID = "segmind/tiny-sd"`,
  defaults, max dimensions, presets.
- `src/imageforge/model_manager.py` — `load_pipeline()` (lazy, cached),
  `is_pipeline_loaded()`, `get_device_info()`, `has_cache()`,
  `supports_negative_prompt()`.
- `src/imageforge/generation.py` — `build_request()`, `validate_prompt()`,
  `validate_dimensions()`, `generate_image()` with mocked-friendly structure.

---

## Model card & licensing

- Model: [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd)
- Provider: Segmind (Stable Diffusion-derived compact text-to-image model)
- License: see the model card above — you are responsible for complying with
  the model's license terms. This project only performs inference; it does not
  train, fine-tune, or redistribute model weights.

---

## Hardware / dependency requirements

- macOS Apple Silicon (M1/M2/M3) preferred; also works on Intel Macs or Linux
  with CUDA CPU fallback
- Python 3.12
- ~1 GB disk space for the one-time model download (stored under `.hf_cache/`)
- 8 GB RAM (MacBook Air M2 class) — generation is memory-intensive; only one
  image is produced per request

### Direct dependencies (pinned)

```
diffusers==0.33.0
torch==2.2.2
transformers==4.43.4
accelerate==0.34.0
safetensors==0.4.3
Pillow==10.4.0
streamlit==1.36.0
pytest==8.3.3
numpy==1.26.4
```

---

## macOS M2 setup instructions

```bash
# 1. Create a project virtual environment (do not reuse Task 01's .venv)
cd /Users/hrishabh/Coding/prodigy-genai-task-02-image-generation
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the local package in editable mode (fixes import without PYTHONPATH)
pip install -e . --no-build-isolation

# 4. Launch the app
streamlit run app.py
```

### What happens on first launch

- The homepage opens **without** downloading anything.
- The sidebar shows device info (e.g. “Apple Silicon (MPS)”) and whether the
  pipeline is loaded and whether a local model cache exists.
- When you click **Generate**, the pipeline loads lazily. On the very first
  run this downloads ~1 GB of weights from Hugging Face (requires internet).
- Subsequent runs reuse the local `.hf_cache/` weights — no re-download.
- **Note on Apple Silicon:** `float32` is used on MPS for numerical stability.
  Earlier attempts with `float16` produced a uniform black image; switching
  to `float32` resolved the failure.
- Attention slicing is enabled automatically on Apple Silicon to keep peak
  memory low.
- The generated image is validated (not black/near-uniform) before the UI
  reports success; a failure raises a readable error instead of a false green message.

### How to use the app

1. Type (or paste) a text prompt into the multiline box.
2. Optionally add a negative prompt (things to avoid).
3. Optionally click a preset (Photorealistic, Digital Art, Watercolor) to
   append style keywords.
4. Open **Advanced settings** to change dimensions, inference steps, guidance
   scale, or choose a custom seed.
5. Click **Generate image**. Wait for the spinner to finish.
6. View the generated image, review the details panel, and click **Download PNG**
   to save it locally.
7. Generated PNGs and sidecar `.json` metadata are saved under `outputs/`.

### Seed, inference steps, guidance scale & negative prompt

- **Seed**: reproducibility control. “Random” uses a different seed each time;
  “Custom” lets you reuse a specific integer for identical output given the same
  prompt and settings.
- **Inference steps**: number of denoising steps (default 25, max 50). Higher
  usually means sharper images but more memory/time.
- **Guidance scale**: how closely the image follows the prompt (default 7.5).
  Lower = more creative freedom; higher = stricter adherence.
- **Negative prompt**: describe unwanted elements (blurry, low quality, etc.).
  Supported natively by `StableDiffusionPipeline`.

---

## Tests

Run the full offline-friendly suite (no model download, no real inference):

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Verified results

All tests pass offline:

```
test_config.py .....                                                              [ 16%]
test_generation.py .......                                                        [ 38%]
test_model_manager.py ...                                                         [ 60%]
...
```

(See the QA section below for the exact run output recorded during development.)

Tests cover:

- Empty / whitespace-only prompt validation → friendly `ValueError`
- Dimension bounds, multiples-of-8 rule, min/max clamping
- Inference steps (1–50), guidance scale non-negativity
- `GenerationRequest` canonicalisation (blank negative prompt → `None`)
- `build_request()` round-trip
- `generate_image()` with a mocked pipeline returning a real Pillow image
- Seed (`None` vs custom) and metadata handling
- PNG byte round-trip + signature check + sensible filenames
- Device-selection logic under mocked MPS/CUDA/CPU availability
- Pipeline loading, caching (load-once), and readable failure errors
- `has_cache()` boolean behaviour

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `streamlit` not found | Virtual env not activated / deps missing | `source .venv/bin/activate && pip install -r requirements.txt` |
| Import error on `diffusers`/`torch` | Wrong Python or broken install | Recreate `.venv` and reinstall |
| First generation hangs / long download | ~1 GB weights downloading | Ensure internet; be patient. Subsequent runs are faster. |
| MPS out-of-memory | Large image size / too many steps | Reduce dimensions or steps; attention slicing is auto-enabled. |
| CPU fallback warning | MPS/CUDA unavailable | Generation will run slowly on CPU; consider smaller settings. |
| `No module named 'imageforge'` | Running from wrong directory / editable install missing | Run `pip install -e . --no-build-isolation`, then `streamlit run app.py`. |
| Black/uniform image output | Float16 numerical failure or incomplete pipeline load | `float32` is now the MPS default; check `outputs/` and error message |
| NumPy / PyTorch bridge error (`NumPy is not available`) | NumPy 2.x installed with torch compiled against 1.x | `pip install numpy==1.26.4` and verify with `python -m pip check` |

### Known limitations

- Model quality is modest (`tiny-sd`); not comparable to full Stable Diffusion XL or Midjourney.
- First download requires internet and ~1 GB disk space.
- One image per generation request; no batching.
- No fine-tuning or training is performed — pure inference only.

---

## Controlled smoke-test result (verified)

- **Arch:** arm64 (Apple Silicon M2)
- **MPS available:** True
- **Python:** 3.12.14
- **Torch:** 2.2.2 (`float32` on MPS)
- **Device used:** mps
- **Cache:** reused local `.hf_cache/` weights; no new download required
- **Command:** `python -c "from src.imageforge.model_manager import load_pipeline, DEVICE...` with prompt `a red apple`, 5 inference steps, seed 42
- **Output:** `outputs/smoke_test_512_5steps.png` (366 KB, 512×512)
- **Inspection:** mean 100.96, std 45.50, min 13, max 210 — visibly non-black, valid PNG
- **Image validation:** `_is_valid_image()` passes (not uniform, mean > threshold)
- **Status:** real inference verified; no false success message; UI honest

---

## Sources consulted

- [`segmind/tiny-sd` model card](https://huggingface.co/segmind/tiny-sd)
- [Diffusers MPS optimization documentation](https://huggingface.co/docs/diffusers/v0.33.0/en/optimization/mps)

No API keys, credentials, or personal machine paths are included in code,
tests, or documentation.
