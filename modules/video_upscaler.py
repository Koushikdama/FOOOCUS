"""
Frame-by-frame video super-resolution for Fooocus.

Pipeline per frame:  decode (ffmpeg) -> SR model (tiled, resident on GPU) -> resize to target
-> color fix (keeps the source colors) -> optional face restoration -> temporal blend -> encode (ffmpeg).

Only `torch`, `numpy` and the ldm_patched backend are imported at module level. `modules.config`
(which parses Fooocus command-line flags) is imported lazily, so this module can be used from
the CLI and the tests without going through Fooocus's launcher.
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

import modules.flags as flags
from modules.video_io import (VideoInfo, FrameWriter, FFmpegError, ffmpeg_available, iter_frames,
                              probe_video)

DEFAULT_MODEL_NAME = 'realesr-general-x4v3.pth'          # Real-ESRGAN Compact: fast enough for video
FOOOCUS_MODEL_NAME = 'fooocus_upscaler_s409985e5.bin'    # bundled ESRGAN (RRDB): high quality, ~20x slower
BUILTIN_MODELS = [DEFAULT_MODEL_NAME, FOOOCUS_MODEL_NAME]

SUPPORTED_ARCHS = ('ESRGAN/RRDB (incl. Real-ESRGAN x2/x4), Real-ESRGAN Compact, SwinIR, Swin2SR, HAT, DAT, '
                   'OmniSR, SCUNet, SPSR, SwiftSRGAN')

ProgressCallback = Callable[[int, Optional[int], Optional[float], Optional[np.ndarray]], None]


class VideoUpscaleError(RuntimeError):
    pass


class VideoUpscaleCancelled(Exception):
    pass


@dataclass
class VideoUpscaleOptions:
    model_name: str = DEFAULT_MODEL_NAME
    target: str = '4K'                  # one of modules.flags.video_upscale_targets
    color_fix: str = 'wavelet'          # one of modules.flags.video_color_fix_options
    temporal_blend: float = 0.25        # 0 disables; 0.2-0.3 removes most per-frame shimmer
    scene_cut_threshold: float = 0.25   # mean |diff| of gray thumbnails (0..1) that resets the blend
    face_enhance: bool = False
    face_strength: float = 0.7          # blend of restored vs. original face crop
    codec: str = 'h264'                 # one of modules.flags.video_codecs
    crf: int = 18
    preset: str = 'fast'                # x264/x265 speed preset
    fp16: Optional[bool] = None         # None -> decided by model_management.should_use_fp16
    tile: int = 512                     # 0 = whole frame; halved automatically on out-of-memory
    tile_overlap: int = 32
    device: Optional[str] = None        # None -> model_management.get_torch_device()
    force_sr: bool = False              # run the model even if the input is already >= target
    preview_every: float = 2.0          # seconds between preview frames passed to progress_cb


# ----------------------------------------------------------------------------- models

def list_sr_models() -> list:
    import modules.config
    names = list(BUILTIN_MODELS)
    for name in modules.config.get_model_filenames(modules.config.path_upscale_models,
                                                   ['.pth', '.safetensors', '.bin']):
        if name not in names:
            names.append(name)
    return names


def resolve_model_path(model_name: str) -> str:
    import modules.config
    if model_name == DEFAULT_MODEL_NAME:
        return modules.config.downloading_realesr_general_x4v3()
    if model_name == FOOOCUS_MODEL_NAME:
        return modules.config.downloading_upscale_model()
    if os.path.isfile(model_name):
        return model_name
    path = os.path.join(modules.config.path_upscale_models, model_name)
    if not os.path.isfile(path):
        raise VideoUpscaleError(f'Upscale model not found: {path}')
    return path


_sr_cache = {'key': None, 'model': None}


def unload_sr_model():
    _sr_cache['key'] = None
    _sr_cache['model'] = None


def load_sr_model(model_name: str, device, fp16: bool):
    """Load (and cache) an upscale model; returns (model, integer scale factor)."""
    key = (model_name, str(device), bool(fp16))
    if _sr_cache['key'] == key:
        return _sr_cache['model'], int(_sr_cache['model'].scale)
    unload_sr_model()

    import ldm_patched.modules.utils as ldm_utils
    from ldm_patched.pfn import model_loading

    path = resolve_model_path(model_name)
    sd = ldm_utils.load_torch_file(path, safe_load=True)
    if 'module.layers.0.residual_group.blocks.0.norm1.weight' in sd:
        sd = ldm_utils.state_dict_prefix_replace(sd, {'module.': ''})
    # The bundled Fooocus upscaler uses a different key spelling (see modules/upscaler.py).
    sd = OrderedDict((k.replace('residual_block_', 'RDB'), v) for k, v in sd.items())
    try:
        model = model_loading.load_state_dict(sd).eval()
    except model_loading.UnsupportedModel:
        raise VideoUpscaleError(f'Unsupported upscale model architecture: {os.path.basename(path)}. '
                                f'Supported: {SUPPORTED_ARCHS}.')
    for p in model.parameters():
        p.requires_grad_(False)
    if fp16 and getattr(model, 'supports_fp16', True):
        model = model.half()
    model = model.to(device)
    print(f'[Video Upscale] Loaded {os.path.basename(path)} ({type(model).__name__}, {int(model.scale)}x, '
          f'{"fp16" if next(model.parameters()).dtype == torch.float16 else "fp32"}) on {device}')
    _sr_cache['key'] = key
    _sr_cache['model'] = model
    return model, int(model.scale)


# ----------------------------------------------------------------------------- geometry

def compute_target(in_w: int, in_h: int, target: str, force_sr: bool = False):
    """Return (out_w, out_h, run_sr). Output is even-sized for 4:2:0 encoding."""
    if target == '4x':
        s = 4.0
    elif target == '2x':
        s = 2.0
    elif target == '4K':
        s = min(3840 / in_w, 2160 / in_h)
    elif target == '1080p':
        s = min(1920 / in_w, 1080 / in_h)
    else:
        raise VideoUpscaleError(f'Unknown target {target!r}; expected one of {flags.video_upscale_targets}')
    out_w = max(2, int(round(in_w * s)) // 2 * 2)
    out_h = max(2, int(round(in_h * s)) // 2 * 2)
    return out_w, out_h, (s > 1.0 or force_sr)


def _antialias_ok(x: torch.Tensor) -> bool:
    # Anti-aliased resizing is not implemented for every backend (e.g. MPS on torch 2.1).
    return x.device.type in ('cpu', 'cuda')


def resize_to_target(x: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    if x.shape[-2] == out_h and x.shape[-1] == out_w:
        return x
    downscaling = out_w < x.shape[-1] or out_h < x.shape[-2]
    return F.interpolate(x, size=(out_h, out_w), mode='bicubic', align_corners=False,
                         antialias=downscaling and _antialias_ok(x)).clamp_(0, 1)


def _match_size(ref: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if ref.shape[-2:] == like.shape[-2:]:
        return ref
    downscaling = ref.shape[-1] > like.shape[-1]
    return F.interpolate(ref, size=like.shape[-2:], mode='bicubic', align_corners=False,
                         antialias=downscaling and _antialias_ok(ref))


# ----------------------------------------------------------------------------- super-resolution

class TileState:
    def __init__(self, tile: int, overlap: int):
        self.tile = max(0, int(tile))
        self.overlap = max(0, int(overlap))


def sr_frame(model, scale: int, x: torch.Tensor, state: TileState) -> torch.Tensor:
    """x: [1,3,h,w] float 0..1 on the model's device -> [1,3,h*scale,w*scale] float32 0..1."""
    import ldm_patched.modules.model_management as model_management
    import ldm_patched.modules.utils as ldm_utils

    dtype = next(model.parameters()).dtype
    while True:
        tile = state.tile
        whole_frame = tile <= 0 or (tile >= x.shape[-1] and tile >= x.shape[-2])
        try:
            if whole_frame:
                out = model(x.to(dtype)).float()
            else:
                overlap = min(state.overlap, tile // 4)
                out = ldm_utils.tiled_scale(x.to(dtype), lambda a: model(a).float(), tile_x=tile, tile_y=tile,
                                            overlap=overlap, upscale_amount=scale, output_device=x.device)
            return out.clamp(0, 1)   # not in-place: tiled_scale returns an inference-mode tensor
        except model_management.OOM_EXCEPTION:
            model_management.soft_empty_cache(force=True)
            new_tile = (tile if tile > 0 else max(x.shape[-2:])) // 2
            if new_tile < 128:
                raise VideoUpscaleError('Out of memory even with 128px tiles. Use a smaller target or model.')
            state.tile = new_tile
            print(f'[Video Upscale] Out of memory, retrying with tile size {new_tile}')


# ----------------------------------------------------------------------------- color fix

_gauss_kernels = {}


def _gaussian_kernel(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    if key not in _gauss_kernels:
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=x.device, dtype=x.dtype) / 16.0
        _gauss_kernels[key] = k.expand(3, 1, 3, 3).contiguous()
    return _gauss_kernels[key]


def _blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    # A dilated 3x3 Gaussian is one level of an "a trous" wavelet decomposition.
    x = F.pad(x, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(x, _gaussian_kernel(x), groups=3, dilation=radius)


def _decompose(x: torch.Tensor, levels: int = 5):
    """Split into (high-frequency detail, low-frequency color) using blur radii 1,2,4,...."""
    high = torch.zeros_like(x)
    for i in range(levels):
        low = _blur(x, 2 ** i)
        high += x - low
        x = low
    return high, x


def wavelet_color_fix(sr: torch.Tensor, ref: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """Keep the detail of `sr` but the colors of `ref` (any size; resized to match `sr`)."""
    ref = _match_size(ref, sr)
    sr_high, _ = _decompose(sr, levels)
    _, ref_low = _decompose(ref, levels)
    return (sr_high + ref_low).clamp_(0, 1)


def adain_color_fix(sr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Match the per-channel mean and standard deviation of `sr` to `ref`."""
    ref = _match_size(ref, sr)
    sr_mean, sr_std = sr.mean((2, 3), keepdim=True), sr.std((2, 3), keepdim=True) + 1e-5
    ref_mean, ref_std = ref.mean((2, 3), keepdim=True), ref.std((2, 3), keepdim=True) + 1e-5
    return ((sr - sr_mean) / sr_std * ref_std + ref_mean).clamp_(0, 1)


def apply_color_fix(mode: str, sr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mode == 'wavelet':
        return wavelet_color_fix(sr, ref)
    if mode == 'adain':
        return adain_color_fix(sr, ref)
    if mode in ('none', '', None):
        return sr
    raise VideoUpscaleError(f'Unknown color fix {mode!r}; expected one of {flags.video_color_fix_options}')


# ----------------------------------------------------------------------------- temporal blend

class TemporalBlender:
    """Blend each output frame with the previous one where nothing moved; reset on scene cuts."""

    def __init__(self, alpha: float, cut_threshold: float = 0.25, motion_threshold: float = 0.08):
        self.alpha = float(alpha)
        self.cut_threshold = float(cut_threshold)
        self.motion_threshold = float(motion_threshold)
        self.prev_out = None
        self.prev_thumb = None
        self.cuts = 0

    @staticmethod
    def _thumbnail(src: torch.Tensor) -> torch.Tensor:
        h, w = src.shape[-2:]
        th = max(1, round(64 * h / w))
        # Tiny tensor; computed on the CPU because 'area' resizing is not implemented everywhere (MPS).
        return F.interpolate(src.mean(1, keepdim=True).cpu(), size=(th, 64), mode='area')

    def __call__(self, out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        thumb = self._thumbnail(src)
        is_cut = self.prev_thumb is not None and float((thumb - self.prev_thumb).abs().mean()) > self.cut_threshold
        if is_cut:
            self.cuts += 1
        if self.prev_out is not None and not is_cut and self.alpha > 0:
            diff = (out - self.prev_out).abs().mean(1, keepdim=True)
            diff = F.avg_pool2d(diff, 5, stride=1, padding=2, count_include_pad=False)
            weight = self.alpha * (1.0 - diff / self.motion_threshold).clamp_(0, 1)
            out = out * (1.0 - weight) + self.prev_out * weight
        self.prev_out = out
        self.prev_thumb = thumb
        return out


# ----------------------------------------------------------------------------- face restoration

class FaceEnhancer:
    """GFPGAN v1.4 face restoration using the bundled architecture and facexlib helper."""

    def __init__(self, device, strength: float = 0.7, detect_size: int = 640):
        import modules.config
        import ldm_patched.modules.utils as ldm_utils
        from ldm_patched.pfn import model_loading
        from extras.facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.device = device
        self.strength = float(max(0.0, min(1.0, strength)))
        self.detect_size = detect_size
        self.helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
            use_parse=True, device=device, model_rootpath=modules.config.path_controlnet)
        sd = ldm_utils.load_torch_file(modules.config.downloading_gfpgan_model(), safe_load=True)
        self.model = model_loading.load_state_dict(sd).eval().to(device)   # GFPGANv1Clean, fp32
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.faces_restored = 0

    @torch.inference_mode()
    def _restore_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        x = torch.from_numpy(np.ascontiguousarray(crop_bgr[:, :, ::-1])).permute(2, 0, 1)[None]
        x = (x.to(self.device).float() / 255.0 - 0.5) / 0.5
        # randomize_noise=False keeps the StyleGAN noise fixed, otherwise faces sparkle between frames.
        y = self.model(x, return_rgb=False, randomize_noise=False)[0]
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        y = np.ascontiguousarray(y[:, :, ::-1])
        if self.strength < 1.0:
            y = cv2.addWeighted(y, self.strength, crop_bgr, 1.0 - self.strength, 0)
        return y

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        h = self.helper
        h.clean_all()
        h.read_image(bgr)
        h.get_face_landmarks_5(only_center_face=False, resize=self.detect_size, eye_dist_threshold=5)
        if len(h.all_landmarks_5) == 0:
            return frame_rgb
        h.align_warp_face()
        for crop in h.cropped_faces:
            h.add_restored_face(self._restore_crop(crop))
            self.faces_restored += 1
        h.get_inverse_affine()
        out_bgr = h.paste_faces_to_input_image(upsample_img=bgr)
        return np.ascontiguousarray(out_bgr[:, :, ::-1])


# ----------------------------------------------------------------------------- main entry point

def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x[0].permute(1, 2, 0) * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


def _from_uint8(frame: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).to(device).permute(2, 0, 1)[None].float().div_(255.0)


def _preview(x: torch.Tensor, max_side: int = 512) -> np.ndarray:
    h, w = x.shape[-2:]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        x = F.interpolate(x, size=(max(1, round(h * s)), max(1, round(w * s))), mode='bilinear',
                          align_corners=False)
    return _to_uint8(x)


def describe_run(info: VideoInfo, options: VideoUpscaleOptions, out_w: int, out_h: int) -> str:
    total = info.estimated_total_frames
    return (f'{info.width}x{info.height} @ {float(info.fps):.3f} fps, {total if total else "?"} frames'
            f' -> {out_w}x{out_h}, model {options.model_name}, color fix {options.color_fix},'
            f' temporal blend {options.temporal_blend}, face enhance {options.face_enhance}, {options.codec} crf {options.crf}')


def upscale_video(input_path: str, output_path: str, options: VideoUpscaleOptions,
                  progress_cb: Optional[ProgressCallback] = None,
                  should_stop: Optional[Callable[[], bool]] = None,
                  sr_model=None) -> str:
    """Upscale `input_path` to `output_path`. `sr_model` lets tests inject a model (needs `.scale`)."""
    if not ffmpeg_available():
        raise VideoUpscaleError('ffmpeg/ffprobe not found on PATH. Install ffmpeg (apt install ffmpeg / '
                                'brew install ffmpeg / winget install ffmpeg) and restart.')
    import ldm_patched.modules.model_management as model_management

    info = probe_video(input_path)
    if info.is_hdr:
        raise VideoUpscaleError(
            'HDR input (PQ/HLG or BT.2020) is not supported yet. Tone-map it to SDR first, e.g.\n'
            'ffmpeg -i in.mp4 -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable,'
            'zscale=t=bt709:m=bt709:r=tv,format=yuv420p" -c:a copy sdr.mp4')
    if info.has_alpha:
        print('[Video Upscale] Warning: alpha channel will be dropped.')
    if info.is_vfr_hint:
        print(f'[Video Upscale] Warning: variable frame rate input; output is constant {float(info.fps):.3f} fps.')

    device = torch.device(options.device) if options.device else model_management.get_torch_device()
    fp16 = options.fp16 if options.fp16 is not None else bool(model_management.should_use_fp16(device=device))

    if sr_model is not None:
        model, scale = sr_model.to(device).eval(), int(sr_model.scale)
    else:
        model, scale = load_sr_model(options.model_name, device, fp16)

    out_w, out_h, run_sr = compute_target(info.width, info.height, options.target, options.force_sr)
    if not run_sr:
        print(f'[Video Upscale] Input {info.width}x{info.height} already covers target {options.target}; '
              f'resizing/re-encoding only (use force_sr to run the model anyway).')
    print(f'[Video Upscale] {describe_run(info, options, out_w, out_h)}')

    blender = TemporalBlender(options.temporal_blend, options.scene_cut_threshold) if options.temporal_blend > 0 else None
    faces = FaceEnhancer(device, options.face_strength) if options.face_enhance else None
    tile_state = TileState(options.tile, options.tile_overlap)
    total = info.estimated_total_frames

    start = time.perf_counter()
    last_preview = 0.0
    frames = iter_frames(input_path, info)
    try:
        with FrameWriter(output_path, out_w, out_h, info.fps,
                         audio_source=input_path if info.has_audio else None, audio_codec=info.audio_codec,
                         codec=options.codec, crf=options.crf, preset=options.preset) as writer, \
                torch.inference_mode():
            for i, frame in enumerate(frames):
                if should_stop is not None and should_stop():
                    raise VideoUpscaleCancelled()

                src = _from_uint8(frame, device)
                if run_sr:
                    out = sr_frame(model, scale, src, tile_state)
                    out = resize_to_target(out, out_w, out_h)
                    out = apply_color_fix(options.color_fix, out, src)
                else:
                    out = resize_to_target(src, out_w, out_h)
                if faces is not None:
                    out = _from_uint8(faces(_to_uint8(out)), device)
                if blender is not None:
                    out = blender(out, src)

                writer.write(_to_uint8(out))

                if progress_cb is not None:
                    done = i + 1
                    elapsed = time.perf_counter() - start
                    eta = (elapsed / done) * (total - done) if total and total > done else None
                    preview = None
                    if elapsed - last_preview >= options.preview_every or done == 1:
                        preview = _preview(out)
                        last_preview = elapsed
                    progress_cb(done, total, eta, preview)
    except VideoUpscaleCancelled:
        print('[Video Upscale] Cancelled.')
        raise
    finally:
        frames.close()

    elapsed = time.perf_counter() - start
    print(f'[Video Upscale] Done: {writer.frames_written} frames in {elapsed:.1f}s '
          f'({writer.frames_written / max(elapsed, 1e-6):.2f} fps) -> {output_path}')
    return output_path
