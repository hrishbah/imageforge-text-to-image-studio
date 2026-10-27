"""ImageForge — AI Text-to-Image Studio (Streamlit entry point)."""

import os
import json
import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st

from imageforge.generation import (
    build_request,
    validate_prompt,
    generate_image,
)
from imageforge.model_manager import (
    load_pipeline,
    is_pipeline_loaded,
    supports_negative_prompt,
    get_device_info,
    has_cache,
)
from imageforge import config

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ImageForge — AI Text-to-Image Studio",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Title and subtitle
st.markdown("## ImageForge")
st.markdown(
    "AI Text-to-Image Studio — generate images from text prompts using a "
    "pretrained Stable Diffusion model, running entirely on your machine."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar: status information (non-blocking; no model download on homepage)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Runtime status")
    device_info = get_device_info()
    st.write(f"Device: {device_info['message']} ({device_info['device']})")
    st.write(f"Pipeline loaded: {'Yes' if is_pipeline_loaded() else 'No'}")
    st.write(f"Model cache available: {'Yes' if has_cache() else 'No — will download on first generate'}")

    st.caption(
        "Model: ``segmind/tiny-sd`` (pretrained — no training required). "
        "The model is downloaded from Hugging Face only when you first click "
        "Generate and then reused from the local cache."
    )

# ---------------------------------------------------------------------------
# Prompt controls
# ---------------------------------------------------------------------------
st.subheader("What would you like to create?")

prompt = st.text_area(
    "Prompt",
    placeholder="e.g. A cinematic portrait of a forest at golden hour, detailed, 8k",
    height=140,
)

example_prompt = st.button("Use example prompt")
if example_prompt:
    st.session_state["example_applied"] = True

if "example_applied" in st.session_state:
    prompt = "A cinematic portrait of a forest at golden hour, detailed, 8k"
    st.session_state.pop("example_applied")

st.caption("Example: 'A cinematic portrait of a forest at golden hour, detailed, 8k'")

st.markdown("**Quick presets**")
cols = st.columns(3)
for idx, (preset_name, preset_text) in enumerate(config.PRESETS.items()):
    if cols[idx].button(preset_name, use_container_width=True):
        prompt = f"{prompt.strip()} {preset_text}" if prompt else preset_text

# Negative prompt (supported natively by StableDiffusionPipeline)
negative_prompt = ""
if supports_negative_prompt():
    negative_prompt = st.text_input(
        "Negative prompt (optional)",
        placeholder="e.g. blurry, low quality, distorted, watermark",
        help="Describe elements you want to avoid in the image.",
    )
else:
    st.info("Negative prompts are not supported by the selected model.")

# ---------------------------------------------------------------------------
# Advanced generation controls
# ---------------------------------------------------------------------------
with st.expander("Advanced settings"):
    width = st.number_input("Width (px)", min_value=64, max_value=config.MAX_WIDTH,
                            value=config.DEFAULT_WIDTH, step=64)
    height = st.number_input("Height (px)", min_value=64, max_value=config.MAX_HEIGHT,
                             value=config.DEFAULT_HEIGHT, step=64)
    steps = st.slider("Inference steps", min_value=1, max_value=config.MAX_STEPS,
                      value=config.DEFAULT_STEPS,
                      help="More steps usually means higher quality, at the cost of memory and time.")
    guidance = st.number_input("Guidance scale", min_value=0.0, max_value=50.0,
                               value=config.DEFAULT_GUIDANCE, step=0.5,
                               help="How closely the image should follow the prompt.")

    seed_mode = st.radio("Seed", ["Random", "Custom"], horizontal=True)
    custom_seed = None
    if seed_mode == "Custom":
        custom_seed = st.number_input("Custom seed", min_value=0, value=42, step=1)

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
st.markdown("---")
generate_btn = st.button("Generate image", type="primary", use_container_width=True)

if generate_btn:
    try:
        req = build_request(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance=float(guidance),
            seed=int(custom_seed) if seed_mode == "Custom" else None,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not is_pipeline_loaded():
        st.info(
            "Loading the model pipeline for the first time. "
            "This may download about 1 GB of weights and can take several minutes "
            "on Apple Silicon. Please be patient."
        )

    with st.spinner("Generating your image… this can take a while on first run"):
        try:
            image, gen_config = generate_image(req)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.success("Image generated successfully.")

    st.subheader("Generated image")
    st.image(image, caption=f"{req.width}×{req.height}")

    with st.expander("Generation details"):
        st.write(f"Prompt: {gen_config['prompt']}")
        if gen_config["negative_prompt"]:
            st.write(f"Negative prompt: {gen_config['negative_prompt']}")
        st.write(f"Model: segmind/tiny-sd")
        st.write(f"Device: {gen_config['device']}")
        st.write(f"Seed: {gen_config['seed'] if gen_config['seed'] is not None else 'random'}")
        st.write(f"Dimensions: {gen_config['width']} × {gen_config['height']}")
        st.write(f"Inference steps: {gen_config['num_inference_steps']}")
        st.write(f"Guidance scale: {gen_config['guidance_scale']}")

    # Download button
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"_seed{gen_config['seed']}" if gen_config["seed"] is not None else "_seed-random"
    filename = f"imageforge{seed_part}_{ts}.png"
    st.download_button(
        label="Download PNG",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
    )

    # Save outputs locally
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "wb") as f:
        f.write(buffer.getvalue())

    # Save metadata sidecar
    meta_path = output_dir / f"{filename}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)

# ---------------------------------------------------------------------------
# First-run guidance footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.info(
    "**First run:** The first generation downloads the pretrained model (~1 GB) "
    "from Hugging Face and requires an internet connection. Subsequent generations "
    "reuse the cached weights locally. No model training happens here — this app "
    "only performs inference with ``segmind/tiny-sd``."
)
