import torch
import numpy as np
from typing import Callable, Optional
from PIL import Image

from modules.smart_edit_config import (
    SMART_EDIT_MODEL_ID,
    DEFAULT_NEGATIVE_PROMPT,
    OPTIMAL_RESOLUTION,
    VRAM_THRESHOLD_FOR_OFFLOAD_GB,
)

_pipeline = None
_loaded = False


def _get_available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_mem, _ = torch.cuda.mem_get_info(0)
    return free_mem / (1024 ** 3)


def load_smart_edit_model() -> None:
    global _pipeline, _loaded
    if _loaded and _pipeline is not None:
        return

    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    print("[Smart Edit] Loading InstructPix2Pix model...")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        SMART_EDIT_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )

    _pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        _pipeline.scheduler.config
    )

    if torch.cuda.is_available():
        available_vram = _get_available_vram_gb()
        if available_vram < VRAM_THRESHOLD_FOR_OFFLOAD_GB:
            print(f"[Smart Edit] Low VRAM ({available_vram:.1f} GB), enabling CPU offload")
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline.to("cuda")
    else:
        print("[Smart Edit] No CUDA available, running on CPU (slow)")

    _loaded = True
    print("[Smart Edit] Model loaded successfully")


def unload_smart_edit_model() -> None:
    global _pipeline, _loaded

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Smart Edit] Model unloaded, VRAM freed")


def _numpy_to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.uint8)


def _resize_for_model(img: Image.Image) -> Image.Image:
    w, h = img.size
    target = OPTIMAL_RESOLUTION
    aspect = w / h

    if aspect >= 1.0:
        new_w = target
        new_h = int(target / aspect)
    else:
        new_h = target
        new_w = int(target * aspect)

    new_w = max(64, (new_w // 8) * 8)
    new_h = max(64, (new_h // 8) * 8)

    return img.resize((new_w, new_h), Image.LANCZOS)


def run_smart_edit(
    image: np.ndarray,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    steps: int = 20,
    seed: int = -1,
    callback: Optional[Callable] = None,
) -> np.ndarray:
    load_smart_edit_model()

    pil_image = _numpy_to_pil(image)
    original_size = pil_image.size

    pil_image = _resize_for_model(pil_image)

    generator = None
    if seed >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[Smart Edit] Running: '{prompt}' | steps={steps} | img_cfg={image_guidance_scale} | txt_cfg={guidance_scale}")

    result = _pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        num_inference_steps=steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)

    return _pil_to_numpy(result)
