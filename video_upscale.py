#!/usr/bin/env python
"""
Upscale a video frame by frame with Fooocus's super-resolution models (standalone CLI).

Examples:
    python video_upscale.py input.mp4
    python video_upscale.py input.mp4 --out out_4k.mp4 --target 4K --face-enhance
    python video_upscale.py old_tape.mp4 --color-restore 0.7 --start 60 --duration 20   # test 20 s of a faded tape
    python video_upscale.py input.mp4 --model RealESRGAN_x2plus.pth --tile 0 --preset fast -- --always-high-vram
    python video_upscale.py --list-models

Everything after a bare "--" is passed to Fooocus itself (e.g. --always-cpu, --always-high-vram,
--output-path). Run `python launch.py` once (or `pip install -r requirements_versions.txt`) before
using this script so the Python packages are installed; ffmpeg/ffprobe must be on PATH.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Frame-by-frame video upscaling (Real-ESRGAN family) with color fix, '
                    'temporal smoothing and optional GFPGAN face restoration.',
        epilog='Flags after "--" are passed to Fooocus (e.g. "-- --always-cpu").')
    p.add_argument('input', nargs='?', help='input video file')
    p.add_argument('--out', help='output file (default: <input>_<target>.mp4 next to the input)')
    p.add_argument('--model', default='realesr-general-x4v3.pth',
                   help='model file name in models/upscale_models (or an absolute path); see --list-models')
    p.add_argument('--list-models', action='store_true', help='list available upscale models and exit')
    p.add_argument('--target', default='4K', choices=['4K', '1080p', '2x', '4x'])
    p.add_argument('--color-fix', default='wavelet', choices=['wavelet', 'adain', 'none'],
                   help='keep the colors of the source frames (default: wavelet)')
    p.add_argument('--color-restore', type=float, default=0.0, metavar='STRENGTH',
                   help='0..1 restore faded/old footage: white balance, levels and saturation, temporally '
                        'smoothed (0 = keep the source colors; try 0.7 for old camcorder/VHS tapes)')
    p.add_argument('--start', type=float, default=0.0, metavar='SECONDS', help='skip the first SECONDS of the video')
    p.add_argument('--duration', type=float, default=0.0, metavar='SECONDS',
                   help='process only SECONDS of video (0 = to the end); handy to test settings on a short section')
    p.add_argument('--repair-glitches', action='store_true',
                   help='damaged disc/tape: replace corrupted blocks and garbage frames using the neighbouring frames')
    p.add_argument('--repair-freezes', action='store_true',
                   help='motion-interpolate short runs of repeated (stuck) frames; slow for HD sources')
    p.add_argument('--temporal-blend', type=float, default=0.25,
                   help='0..1 blend with the previous output frame on static pixels to reduce flicker (0 = off)')
    p.add_argument('--scene-cut-threshold', type=float, default=0.25)
    p.add_argument('--face-enhance', action='store_true', help='restore faces with GFPGAN v1.4')
    p.add_argument('--face-strength', type=float, default=0.7)
    p.add_argument('--codec', default='h265', choices=['h265', 'h264'],
                   help='h265 = 10-bit HEVC (smaller, best quality), h264 = plays everywhere')
    p.add_argument('--crf', type=int, default=18, help='quality, lower is better/larger (default 18)')
    p.add_argument('--preset', default='fast', help='x264/x265 speed preset (default: fast)')
    p.add_argument('--tile', type=int, default=512, help='tile size in pixels, 0 = whole frame (default 512)')
    p.add_argument('--tile-overlap', type=int, default=32)
    p.add_argument('--fp16', dest='fp16', action='store_true', default=None, help='force half precision')
    p.add_argument('--fp32', dest='fp16', action='store_false', help='force full precision')
    p.add_argument('--cpu', action='store_true', help='run on the CPU (same as "-- --always-cpu")')
    p.add_argument('--force-sr', action='store_true', help='run the model even if the input already covers the target')
    return p


def format_eta(seconds):
    if seconds is None:
        return '--:--'
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def main() -> int:
    argv = sys.argv[1:]
    if '--' in argv:
        split = argv.index('--')
        cli_argv, fooocus_argv = argv[:split], argv[split + 1:]
    else:
        cli_argv, fooocus_argv = argv, []

    ns = build_parser().parse_args(cli_argv)
    if ns.cpu and '--always-cpu' not in fooocus_argv:
        fooocus_argv.append('--always-cpu')

    # Fooocus parses sys.argv when modules.config is imported; hand it only its own flags.
    sys.argv = [sys.argv[0]] + fooocus_argv
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

    import modules.config  # noqa: F401  (must come before ldm_patched.modules.model_management)
    from modules.video_io import ffmpeg_available
    from modules.video_upscaler import (VideoUpscaleCancelled, VideoUpscaleError, VideoUpscaleOptions,
                                        list_sr_models, upscale_video)

    if ns.list_models:
        for name in list_sr_models():
            print(name)
        return 0
    if not ns.input:
        build_parser().print_usage()
        print('error: input video is required', file=sys.stderr)
        return 2
    if not ffmpeg_available():
        print('error: ffmpeg/ffprobe not found on PATH. Install ffmpeg: apt-get install -y ffmpeg | '
              'brew install ffmpeg | winget install ffmpeg', file=sys.stderr)
        return 1

    input_path = os.path.abspath(ns.input)
    if ns.out:
        output_path = os.path.abspath(ns.out)
    else:
        stem, _ = os.path.splitext(input_path)
        output_path = f'{stem}_{ns.target}.mp4'

    options = VideoUpscaleOptions(
        model_name=ns.model, target=ns.target, color_fix=ns.color_fix, temporal_blend=ns.temporal_blend,
        scene_cut_threshold=ns.scene_cut_threshold, face_enhance=ns.face_enhance, face_strength=ns.face_strength,
        codec=ns.codec, crf=ns.crf, preset=ns.preset, fp16=ns.fp16, tile=ns.tile, tile_overlap=ns.tile_overlap,
        force_sr=ns.force_sr, color_restore=ns.color_restore, start=ns.start,
        duration=ns.duration if ns.duration > 0 else None,
        repair_glitches=ns.repair_glitches, repair_freezes=ns.repair_freezes)

    def progress(done, total, eta, preview):
        if total:
            line = f'[Video Upscale] {done}/{total} frames ({100 * done / total:5.1f}%)  ETA {format_eta(eta)}'
        else:
            line = f'[Video Upscale] {done} frames'
        print('\r' + line.ljust(70), end='', flush=True)

    try:
        upscale_video(input_path, output_path, options, progress_cb=progress)
        print()
        print(f'Saved {output_path}')
        return 0
    except (KeyboardInterrupt, VideoUpscaleCancelled):
        print('\nCancelled; partial output removed.')
        return 130
    except VideoUpscaleError as e:
        print(f'\nerror: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
