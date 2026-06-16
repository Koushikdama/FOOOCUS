import os
import torch
import numpy as np
from typing import Callable, Optional
from PIL import Image

from modules.video_config import (
    VIDEO_MODELS,
    VIDEO_RESOLUTIONS,
    VideoQuality,
    DEFAULT_VIDEO_NEGATIVE_PROMPT,
    DEFAULT_VIDEO_STEPS,
    VRAM_THRESHOLD_FOR_OFFLOAD_GB,
)

_pipeline = None
_loaded_quality: Optional[VideoQuality] = None


def _get_available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_mem, _ = torch.cuda.mem_get_info(0)
    return free_mem / (1024 ** 3)


def load_video_model(quality: VideoQuality = VideoQuality.STANDARD) -> None:
    global _pipeline, _loaded_quality

    if _loaded_quality == quality and _pipeline is not None:
        return

    unload_video_model()

    from diffusers import WanImageToVideoPipeline

    model_id = VIDEO_MODELS[quality]
    print(f"[Video Gen] Loading {model_id} ...")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _pipeline = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    if torch.cuda.is_available():
        available_vram = _get_available_vram_gb()
        if available_vram < VRAM_THRESHOLD_FOR_OFFLOAD_GB:
            print(f"[Video Gen] Low VRAM ({available_vram:.1f} GB), enabling sequential CPU offload")
            _pipeline.enable_sequential_cpu_offload()
        else:
            _pipeline.enable_model_cpu_offload()
    else:
        print("[Video Gen] No CUDA available, running on CPU (extremely slow)")

    _loaded_quality = quality
    print(f"[Video Gen] Model loaded ({quality.value})")


def unload_video_model() -> None:
    global _pipeline, _loaded_quality

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _loaded_quality = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Video Gen] Model unloaded, VRAM freed")


def _numpy_to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), Image.LANCZOS)


def _export_to_mp4(frames: list, fps: int, output_path: str) -> str:
    import imageio.v3 as iio

    frame_arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame_arrays.append(np.array(frame))
        elif isinstance(frame, np.ndarray):
            frame_arrays.append(frame)

    stacked = np.stack(frame_arrays)
    iio.imwrite(output_path, stacked, fps=fps, codec="libx264", plugin="pyav")

    print(f"[Video Gen] Saved video to {output_path}")
    return output_path


def generate_video(
    image: np.ndarray,
    prompt: str,
    output_dir: str,
    negative_prompt: str = DEFAULT_VIDEO_NEGATIVE_PROMPT,
    quality: VideoQuality = VideoQuality.STANDARD,
    num_frames: int = 81,
    fps: int = 16,
    guidance_scale: float = 5.0,
    steps: int = DEFAULT_VIDEO_STEPS,
    seed: int = -1,
    callback: Optional[Callable] = None,
) -> str:
    load_video_model(quality)

    pil_image = _numpy_to_pil(image)

    target_w, target_h = VIDEO_RESOLUTIONS[quality]
    pil_image = _resize_image(pil_image, target_w, target_h)

    generator = None
    if seed >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[Video Gen] Generating: '{prompt}' | frames={num_frames} | fps={fps} | quality={quality.value}")

    output = _pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )

    frames = output.frames[0]

    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = int(time.time())
    actual_seed = seed if seed >= 0 else "random"
    output_filename = f"video_{timestamp}_{actual_seed}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    return _export_to_mp4(frames, fps, output_path)
