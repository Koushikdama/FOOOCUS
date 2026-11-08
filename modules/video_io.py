"""
ffmpeg/ffprobe layer for the Video Upscale feature.

This module has no torch or Fooocus imports on purpose: it can be imported by
the CLI and the tests without triggering Fooocus's argument parsing or model
loading.  Frames flow through two ffmpeg subprocess pipes (raw RGB in both
directions), the same approach Real-ESRGAN's video script uses, so a video of
any length only ever needs two frames in memory.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional

import numpy as np


class FFmpegError(RuntimeError):
    pass


FFMPEG = 'ffmpeg'
FFPROBE = 'ffprobe'

# Audio codecs that can be copied into an mp4 container without re-encoding.
MP4_SAFE_AUDIO_CODECS = {'aac', 'mp3', 'ac3', 'eac3', 'alac'}

HDR_TRANSFERS = {'smpte2084', 'arib-std-b67'}
RGBA_PIX_FMTS = {'rgba', 'bgra', 'argb', 'abgr', 'rgba64le', 'rgba64be', 'bgra64le', 'bgra64be', 'pal8'}

SCALE_FLAGS = 'spline+accurate_rnd+full_chroma_int'


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


@dataclass
class VideoInfo:
    width: int                      # square-pixel display size, after applying rotation metadata
    height: int
    fps: Fraction
    nb_frames: Optional[int]
    duration: Optional[float]
    codec_name: str = ''
    pix_fmt: str = ''
    color_range: str = ''           # 'tv' | 'pc' | ''
    color_space: str = ''           # 'bt709' | 'smpte170m' | 'bt2020nc' | ...
    color_primaries: str = ''
    color_transfer: str = ''
    rotation: int = 0               # 0 | 90 | 180 | 270
    has_audio: bool = False
    audio_codec: str = ''
    is_vfr_hint: bool = False
    stored_width: int = 0           # coded frame size (before rotation / pixel aspect correction)
    stored_height: int = 0
    sar: Fraction = Fraction(1)     # sample (pixel) aspect ratio; != 1 for VCD/DVD/DV style anamorphic footage

    @property
    def has_non_square_pixels(self) -> bool:
        return self.sar != 1

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS or self.color_primaries == 'bt2020'

    @property
    def has_alpha(self) -> bool:
        p = self.pix_fmt
        return p.startswith('yuva') or p.startswith('gbrap') or p.startswith('ya') or p in RGBA_PIX_FMTS

    @property
    def estimated_total_frames(self) -> Optional[int]:
        if self.nb_frames:
            return self.nb_frames
        if self.duration and self.fps > 0:
            return max(1, round(self.duration * float(self.fps)))
        return None

    def trimmed_duration(self, start: float = 0.0, duration: Optional[float] = None) -> Optional[float]:
        """Seconds that will be processed when decoding from `start` for at most `duration` seconds."""
        if self.duration is None:
            return duration if duration and duration > 0 else None
        remaining = max(0.0, self.duration - max(0.0, start))
        if duration and duration > 0:
            remaining = min(remaining, duration)
        return remaining

    def estimated_frames(self, start: float = 0.0, duration: Optional[float] = None) -> Optional[int]:
        if start <= 0 and not (duration and duration > 0):
            return self.estimated_total_frames
        seconds = self.trimmed_duration(start, duration)
        if seconds is None or self.fps <= 0:
            return None
        return max(1, round(seconds * float(self.fps)))


def _parse_fraction(text) -> Fraction:
    try:
        f = Fraction(str(text))
        return f if f > 0 else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def _parse_sar(text) -> Fraction:
    """ffprobe prints the sample aspect ratio as 'num:den'; '0:1', 'N/A' and missing mean square pixels."""
    if not text or text in ('N/A', '0:1', '1:1'):
        return Fraction(1)
    try:
        num, den = str(text).split(':')
        sar = Fraction(int(num), int(den))
    except (ValueError, ZeroDivisionError):
        return Fraction(1)
    # Guard against garbage tags; real anamorphic content stays well inside this window.
    return sar if 0.5 <= sar <= 2.0 and sar > 0 else Fraction(1)


def _parse_rotation(stream: dict) -> int:
    rotation = None
    for side in stream.get('side_data_list', []) or []:
        if 'rotation' in side:
            rotation = side['rotation']
            break
    if rotation is None:
        rotation = (stream.get('tags') or {}).get('rotate')
    try:
        rotation = int(round(float(rotation))) % 360
    except (TypeError, ValueError):
        return 0
    return rotation if rotation in (0, 90, 180, 270) else 0


def probe_video(path: str) -> VideoInfo:
    if not os.path.isfile(path):
        raise FFmpegError(f'Input video not found: {path}')
    cmd = [
        FFPROBE, '-v', 'error',
        '-show_entries',
        'stream=index,codec_type,codec_name,width,height,sample_aspect_ratio,r_frame_rate,avg_frame_rate,nb_frames,'
        'pix_fmt,color_range,color_space,color_primaries,color_transfer'
        ':stream_tags=rotate:stream_side_data=rotation:format=duration',
        '-of', 'json', path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f'ffprobe failed for {path}:\n{result.stderr.strip()}')
    data = json.loads(result.stdout or '{}')

    video = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    audio = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if video is None:
        raise FFmpegError(f'No video stream found in {path}')

    avg = _parse_fraction(video.get('avg_frame_rate', '0/0'))
    real = _parse_fraction(video.get('r_frame_rate', '0/0'))
    fps = avg if avg > 0 else real
    if fps <= 0:
        raise FFmpegError(f'Could not determine frame rate of {path}')

    try:
        duration = float(data.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        duration = None
    try:
        nb_frames = int(video.get('nb_frames'))
        if nb_frames <= 0:
            nb_frames = None
    except (TypeError, ValueError):
        nb_frames = None

    stored_width, stored_height = int(video['width']), int(video['height'])
    sar = _parse_sar(video.get('sample_aspect_ratio'))
    # Non-square pixels (e.g. 352x288 VCD with SAR 12:11 displays as 4:3): process square pixels so the
    # output keeps the shape players show for the source.
    width = stored_width if sar == 1 else max(2, int(round(stored_width * sar / 2)) * 2)
    height = stored_height
    rotation = _parse_rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    def tag(key):
        value = video.get(key)
        return '' if value in (None, 'unknown', 'N/A') else str(value)

    return VideoInfo(
        width=width, height=height, fps=fps, nb_frames=nb_frames, duration=duration,
        codec_name=tag('codec_name'), pix_fmt=tag('pix_fmt'),
        color_range=tag('color_range'), color_space=tag('color_space'),
        color_primaries=tag('color_primaries'), color_transfer=tag('color_transfer'),
        rotation=rotation,
        has_audio=audio is not None, audio_codec=(audio or {}).get('codec_name', '') or '',
        is_vfr_hint=(avg > 0 and real > 0 and avg != real),
        stored_width=stored_width, stored_height=stored_height, sar=sar,
    )


def _input_color_matrix(info: VideoInfo) -> str:
    cs = info.color_space
    if cs in ('bt709',):
        return 'bt709'
    if cs in ('smpte170m', 'bt470bg', 'bt601', 'smpte240m'):
        return 'smpte170m'
    if cs.startswith('bt2020'):
        return 'bt2020'
    # Unspecified: HD content is almost always BT.709, SD content BT.601.
    return 'bt709' if info.height >= 720 else 'smpte170m'


def _input_color_range(info: VideoInfo) -> str:
    return 'pc' if info.color_range == 'pc' else 'tv'


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _stderr_text(stderr_file) -> str:
    try:
        stderr_file.seek(0)
        return stderr_file.read().decode('utf-8', errors='replace').strip()
    except Exception:
        return ''


def _trim_args(start: float = 0.0, duration: Optional[float] = None) -> list:
    """Input-side seek/limit options (placed before -i) for the video and the audio source alike."""
    args = []
    if start and start > 0:
        args += ['-ss', f'{float(start):.6f}']
    if duration and duration > 0:
        args += ['-t', f'{float(duration):.6f}']
    return args


def freeze_repair_filter(fps: Fraction, max_freeze_seconds: float = 0.5) -> str:
    """ffmpeg filters that replace short runs of repeated frames (damaged disc/tape read errors) with
    motion-interpolated frames. `mpdecimate` drops frames identical to the previous one, at most
    `max_freeze_seconds` in a row, so a long freeze stays a freeze instead of a slow morph; `minterpolate`
    fills the gaps at the original frame rate, so the duration and the audio sync do not change."""
    max_drop = max(1, int(round(float(fps) * max_freeze_seconds)))
    rate = f'{fps.numerator}/{fps.denominator}'
    return (f'mpdecimate=max={max_drop},'
            f'minterpolate=fps={rate}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1')


def iter_frames(path: str, info: Optional[VideoInfo] = None, start: float = 0.0,
                duration: Optional[float] = None, repair_freezes: bool = False) -> Iterator[np.ndarray]:
    """Yield every frame as an HxWx3 RGB uint8 array (full range, square pixels), at a constant frame rate.

    `start`/`duration` (seconds) process only a section of the video; the audio is trimmed to match by
    passing the same values to FrameWriter. `repair_freezes` motion-interpolates short runs of repeated
    frames (see freeze_repair_filter)."""
    if info is None:
        info = probe_video(path)
    w, h = info.width, info.height
    vf = (
        (freeze_repair_filter(info.fps) + ',' if repair_freezes else '') +
        f'fps={info.fps.numerator}/{info.fps.denominator},'
        f'scale=w={w}:h={h}:in_range={_input_color_range(info)}:in_color_matrix={_input_color_matrix(info)}'
        f':out_range=pc:flags={SCALE_FLAGS},setsar=1,format=rgb24'
    )
    cmd = [
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin',
        *_trim_args(start, duration),
        '-i', path, '-map', '0:v:0', '-an', '-sn', '-dn',
        '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
    ]
    frame_bytes = w * h * 3
    stderr = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, bufsize=frame_bytes * 4)
    try:
        while True:
            buf = _read_exact(proc.stdout, frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise FFmpegError(f'ffmpeg decoder exited with code {rc}:\n{_stderr_text(stderr)}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        stderr.close()


_encoder_cache = {}


def has_encoder(name: str) -> bool:
    if name not in _encoder_cache:
        try:
            out = subprocess.run([FFMPEG, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
        except OSError:
            out = ''
        _encoder_cache[name] = any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())
    return _encoder_cache[name]


def choose_video_encoder(codec: str) -> str:
    """Map 'h265'/'h264' to an available ffmpeg encoder, falling back to libx264."""
    if codec == 'h265':
        if has_encoder('libx265'):
            return 'libx265'
        print('[Video Upscale] libx265 not available in this ffmpeg build, falling back to libx264.')
    return 'libx264'


class FrameWriter:
    """Encodes RGB frames written one at a time; audio is muxed from `audio_source`."""

    def __init__(self, output_path: str, width: int, height: int, fps: Fraction, *,
                 audio_source: Optional[str] = None, audio_codec: str = '',
                 codec: str = 'h264', crf: int = 18, preset: str = 'fast',
                 start: float = 0.0, duration: Optional[float] = None):
        assert width % 2 == 0 and height % 2 == 0, 'output size must be even for 4:2:0 encoding'
        self.output_path = os.path.abspath(output_path)
        base, ext = os.path.splitext(self.output_path)
        self.part_path = f'{base}.part{ext or ".mp4"}'
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3
        self.frames_written = 0
        self.encoder = choose_video_encoder(codec)

        cmd = [
            FFMPEG, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', f'{fps.numerator}/{fps.denominator}', '-i', 'pipe:0',
        ]
        if audio_source:
            cmd += [*_trim_args(start, duration), '-i', audio_source, '-map', '0:v:0', '-map', '1:a:0?']
            if audio_codec in MP4_SAFE_AUDIO_CODECS:
                cmd += ['-c:a', 'copy']
            else:
                cmd += ['-c:a', 'aac', '-b:a', '192k']
        else:
            cmd += ['-map', '0:v:0']

        # Frame-level tags via setparams are what the encoders actually record; the -color_* output
        # options alone are ignored for some inputs (observed with ffmpeg 9 + libx264).
        pix_fmt = 'yuv420p10le' if self.encoder == 'libx265' else 'yuv420p'
        vf = (f'scale=out_color_matrix=bt709:out_range=tv:flags={SCALE_FLAGS},format={pix_fmt},'
              f'setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv')
        if self.encoder == 'libx265':
            cmd += ['-vf', vf, '-c:v', 'libx265', '-preset', preset, '-crf', str(int(crf)), '-tag:v', 'hvc1',
                    '-x265-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:log-level=error']
        else:
            cmd += ['-vf', vf, '-c:v', 'libx264', '-preset', preset, '-crf', str(int(crf)), '-profile:v', 'high']
        cmd += ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
                '-movflags', '+faststart', self.part_path]

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr, bufsize=self.frame_bytes * 4)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        assert frame.dtype == np.uint8 and frame.shape == (self.height, self.width, 3), \
            f'expected {(self.height, self.width, 3)} uint8 frame, got {frame.shape} {frame.dtype}'
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise FFmpegError(f'ffmpeg encoder stopped unexpectedly:\n{_stderr_text(self._stderr)}')
        self.frames_written += 1

    def close(self, success: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if success:
                self._proc.stdin.close()
                rc = self._proc.wait()
                if rc != 0:
                    raise FFmpegError(f'ffmpeg encoder exited with code {rc}:\n{_stderr_text(self._stderr)}')
                os.replace(self.part_path, self.output_path)
            else:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
                if self._proc.poll() is None:
                    self._proc.kill()
                self._proc.wait()
        finally:
            self._stderr.close()
            if not success and os.path.exists(self.part_path):
                try:
                    os.remove(self.part_path)
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(success=exc_type is None)
        return False


def make_synthetic_video(path: str, width: int = 64, height: int = 64, frames: int = 10, fps: int = 10,
                         audio: bool = True, source: str = 'testsrc2', codec: str = 'libx264',
                         sar: Optional[str] = None) -> str:
    """Create a small test clip. `source` is any lavfi video source, e.g. 'testsrc2' or 'color=c=red'.
    `sar` (e.g. '12/11') tags the clip with non-square pixels."""
    duration = frames / fps
    sep = ':' if '=' in source else '='   # lavfi: first option after '=', the rest after ':'
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'lavfi', '-i', f'{source}{sep}size={width}x{height}:rate={fps}']
    if audio:
        cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=44100:duration={duration}']
    cmd += ['-frames:v', str(frames)]
    sar_vf = f',setsar={sar}' if sar else ''
    if codec == 'libx264':
        cmd += ['-vf', f'scale=out_color_matrix=bt709:out_range=tv:flags={SCALE_FLAGS},format=yuv420p,'
                       f'setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv{sar_vf}',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '12',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    else:
        cmd += ['-c:v', codec, '-pix_fmt', 'rgb24']
        if sar:
            cmd += ['-vf', f'setsar={sar}']
    if audio:
        cmd += ['-c:a', 'aac', '-b:a', '64k', '-shortest']
    cmd += [path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip())
    return path
