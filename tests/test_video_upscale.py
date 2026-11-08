import os
import shutil
import tempfile
import unittest
from fractions import Fraction

import numpy as np

from modules import video_io

HAS_FFMPEG = shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None

try:
    import torch
    import torch.nn.functional as F
    from modules import video_upscaler as vu
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard requirement of Fooocus itself
    HAS_TORCH = False


if HAS_TORCH:
    class IdentityX4(torch.nn.Module):
        """Stand-in for a real SR model: nearest-neighbour 4x upsampling."""
        scale = 4
        supports_fp16 = True

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return F.interpolate(x, scale_factor=4, mode='nearest')

    def _options(**kwargs):
        base = dict(target='4x', color_fix='none', temporal_blend=0.0, codec='h264', crf=10,
                    preset='ultrafast', device='cpu', fp16=False, tile=0)
        base.update(kwargs)
        return vu.VideoUpscaleOptions(**base)


def _moving_square_frames(n: int = 30, size: int = 64):
    """Gray background with a gradient (so the scene has texture) and a bright square moving 2 px per frame."""
    frames = []
    yy, xx = np.mgrid[0:size, 0:size]
    background = np.stack([40 + xx * 2, 80 + yy * 2, np.full_like(xx, 120)], -1).astype(np.uint8)
    for i in range(n):
        f = background.copy()
        x0 = 4 + 2 * i
        f[8:24, x0:x0 + 16] = (250, 250, 250)
        frames.append(f)
    return frames


def _write_frames(path: str, frames, fps: int = 10) -> str:
    h, w = frames[0].shape[:2]
    with video_io.FrameWriter(path, w, h, Fraction(fps), codec='h264', crf=1, preset='ultrafast') as writer:
        for f in frames:
            writer.write(f)
    return path


@unittest.skipUnless(HAS_FFMPEG, 'ffmpeg/ffprobe not on PATH')
class TestVideoIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='fooocus_video_io_')
        cls.clip = video_io.make_synthetic_video(os.path.join(cls.tmp, 'in.mp4'), 64, 64, 10, 10, audio=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_probe(self):
        info = video_io.probe_video(self.clip)
        self.assertEqual((info.width, info.height), (64, 64))
        self.assertEqual(info.fps, Fraction(10))
        self.assertTrue(info.has_audio)
        self.assertEqual(info.audio_codec, 'aac')
        self.assertIn(info.nb_frames, (9, 10, 11))
        self.assertEqual(info.estimated_total_frames, info.nb_frames)
        self.assertFalse(info.is_hdr)
        self.assertFalse(info.has_alpha)

    def test_read_frames(self):
        frames = list(video_io.iter_frames(self.clip))
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0].shape, (64, 64, 3))
        self.assertEqual(frames[0].dtype, np.uint8)

    def test_color_round_trip(self):
        # A solid colour must survive decode (YUV->RGB) and encode (RGB->YUV) without a shift.
        for name, expected in [('color=c=red', (255, 0, 0)), ('color=c=0x808080', (128, 128, 128))]:
            src = video_io.make_synthetic_video(os.path.join(self.tmp, 'solid.mp4'), 64, 64, 4, 10,
                                                audio=False, source=name)
            frame = next(iter(video_io.iter_frames(src)))
            np.testing.assert_allclose(frame.reshape(-1, 3).mean(0), expected, atol=2)
            out = os.path.join(self.tmp, 'solid_out.mp4')
            with video_io.FrameWriter(out, 64, 64, Fraction(10), codec='h264', crf=1, preset='ultrafast') as w:
                w.write(frame)
            frame2 = next(iter(video_io.iter_frames(out)))
            np.testing.assert_allclose(frame2.reshape(-1, 3).mean(0), expected, atol=2)

    def test_writer_tags_and_audio(self):
        info = video_io.probe_video(self.clip)
        out = os.path.join(self.tmp, 'out.mp4')
        with video_io.FrameWriter(out, 128, 128, info.fps, audio_source=self.clip, audio_codec=info.audio_codec,
                                  codec='h264', crf=20, preset='ultrafast') as w:
            for frame in video_io.iter_frames(self.clip, info):
                w.write(np.repeat(np.repeat(frame, 2, 0), 2, 1))
        self.assertFalse(os.path.exists(w.part_path))
        out_info = video_io.probe_video(out)
        self.assertEqual((out_info.width, out_info.height), (128, 128))
        self.assertEqual(out_info.codec_name, 'h264')
        self.assertEqual(out_info.color_space, 'bt709')
        self.assertEqual(out_info.color_primaries, 'bt709')
        self.assertEqual(out_info.color_transfer, 'bt709')
        self.assertEqual(out_info.color_range, 'tv')
        self.assertTrue(out_info.has_audio)
        self.assertIn(out_info.nb_frames, (9, 10, 11))

    def test_non_square_pixels_are_decoded_square(self):
        # 352x288 VCD-style footage carries SAR 12:11 and displays as 4:3; we process square pixels.
        src = video_io.make_synthetic_video(os.path.join(self.tmp, 'sar.mp4'), 88, 72, 3, 10, audio=False,
                                            sar='12/11')
        info = video_io.probe_video(src)
        self.assertEqual(info.sar, Fraction(12, 11))
        self.assertTrue(info.has_non_square_pixels)
        self.assertEqual((info.stored_width, info.stored_height), (88, 72))
        self.assertEqual((info.width, info.height), (96, 72))           # 88 * 12/11 = 96 -> 4:3
        frames = list(video_io.iter_frames(src, info))
        self.assertEqual(frames[0].shape, (72, 96, 3))
        self.assertFalse(video_io.probe_video(self.clip).has_non_square_pixels)

    def test_trim_start_and_duration(self):
        src = video_io.make_synthetic_video(os.path.join(self.tmp, 'long.mp4'), 64, 64, 30, 10, audio=True)
        info = video_io.probe_video(src)
        self.assertEqual(info.estimated_frames(1.0, 1.0), 10)
        self.assertEqual(info.estimated_frames(2.5, None), 5)
        self.assertEqual(info.estimated_frames(0, 0), info.estimated_total_frames)
        frames = list(video_io.iter_frames(src, info, start=1.0, duration=1.0))
        self.assertIn(len(frames), (9, 10, 11))
        full = list(video_io.iter_frames(src, info))
        # testsrc2 draws a frame counter, so frame 10 of the full clip must match frame 0 of the trimmed one.
        self.assertLess(np.abs(full[10].astype(np.float32) - frames[0].astype(np.float32)).mean(), 4.0)
        out = os.path.join(self.tmp, 'trimmed.mp4')
        with video_io.FrameWriter(out, 64, 64, info.fps, audio_source=src, audio_codec=info.audio_codec,
                                  codec='h264', crf=20, preset='ultrafast', start=1.0, duration=1.0) as w:
            for frame in frames:
                w.write(frame)
        out_info = video_io.probe_video(out)
        self.assertTrue(out_info.has_audio)
        self.assertAlmostEqual(out_info.duration, 1.0, delta=0.25)       # audio trimmed to the video section

    def test_freeze_repair_interpolates_stuck_frames(self):
        frames = _moving_square_frames(30)
        frozen = frames[:10] + [frames[9]] * 4 + frames[14:]          # frames 10..13 stuck on frame 9
        src = _write_frames(os.path.join(self.tmp, 'frozen.mp4'), frozen)
        info = video_io.probe_video(src)
        plain = list(video_io.iter_frames(src, info))
        repaired = list(video_io.iter_frames(src, info, repair_freezes=True))
        self.assertEqual(len(repaired), len(plain))                     # duration / audio sync unchanged
        diff = lambda a, b: np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
        self.assertLess(diff(plain[11], plain[9]), 1.0)                 # really frozen in the source
        self.assertGreater(diff(repaired[11], repaired[9]), 1.0)        # ... and moving again after repair
        self.assertLess(diff(repaired[11], frames[11]), diff(plain[11], frames[11]))   # closer to the truth
        self.assertLess(diff(repaired[20], frames[20]), 2.0)            # untouched frames stay intact

    def test_writer_failure_removes_partial_file(self):
        out = os.path.join(self.tmp, 'cancel.mp4')
        w = video_io.FrameWriter(out, 64, 64, Fraction(10))
        w.write(np.zeros((64, 64, 3), np.uint8))
        w.close(success=False)
        self.assertFalse(os.path.exists(w.part_path))
        self.assertFalse(os.path.exists(out))


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class TestVideoUpscalerPure(unittest.TestCase):
    def test_compute_target(self):
        self.assertEqual(vu.compute_target(1920, 1080, '4K'), (3840, 2160, True))
        self.assertEqual(vu.compute_target(640, 360, '4K'), (3840, 2160, True))
        self.assertEqual(vu.compute_target(640, 480, '4K'), (2880, 2160, True))
        self.assertEqual(vu.compute_target(3840, 2160, '4K'), (3840, 2160, False))
        self.assertEqual(vu.compute_target(3840, 2160, '4K', force_sr=True), (3840, 2160, True))
        self.assertEqual(vu.compute_target(63, 65, '2x'), (126, 130, True))
        self.assertEqual(vu.compute_target(1920, 1080, '1080p'), (1920, 1080, False))
        with self.assertRaises(vu.VideoUpscaleError):
            vu.compute_target(10, 10, '8K')

    def test_wavelet_color_fix_keeps_detail_and_fixes_color(self):
        torch.manual_seed(0)
        ref = vu._blur(vu._blur(torch.rand(1, 3, 64, 64), 4), 8)                 # smooth source frame
        cast = torch.tensor([0.15, -0.10, 0.05]).view(1, 3, 1, 1)
        noise = (torch.rand(1, 3, 64, 64) - 0.5) * 0.2                            # fake "detail"
        sr = (ref + cast + noise).clamp(0, 1)
        out = vu.wavelet_color_fix(sr, ref)
        self.assertTrue(torch.allclose(out.mean((2, 3)), ref.mean((2, 3)), atol=0.01))

        def hf(x):
            return (x - vu._blur(x, 1)).flatten()
        corr = torch.corrcoef(torch.stack([hf(out), hf(sr)]))[0, 1]
        self.assertGreater(float(corr), 0.9)

    def test_wavelet_color_fix_resizes_reference(self):
        ref = torch.rand(1, 3, 16, 16)
        sr = torch.rand(1, 3, 64, 64)
        self.assertEqual(tuple(vu.wavelet_color_fix(sr, ref).shape), (1, 3, 64, 64))

    def test_adain_color_fix(self):
        ref = torch.rand(1, 3, 32, 32) * 0.5 + 0.25
        sr = torch.rand(1, 3, 32, 32)
        out = vu.adain_color_fix(sr, ref)
        self.assertTrue(torch.allclose(out.mean((2, 3)), ref.mean((2, 3)), atol=0.02))

    def test_temporal_blender_blends_static_and_resets_on_cut(self):
        torch.manual_seed(0)
        blender = vu.TemporalBlender(alpha=0.5, cut_threshold=0.25, motion_threshold=0.08)
        a = torch.rand(1, 3, 32, 32) * 0.9
        first = blender(a.clone(), a)
        self.assertTrue(torch.equal(first, a))                       # nothing to blend with yet
        a2 = a + 0.01                                                 # tiny uniform change, below motion threshold
        second = blender(a2.clone(), a2)
        weight = 0.5 * (1 - 0.01 / 0.08)                              # blend weight shrinks with motion
        expected = a2 * (1 - weight) + a * weight
        self.assertTrue(torch.allclose(second, expected, atol=1e-5))
        b = 1.0 - a                                                   # scene cut: completely different frame
        third = blender(b.clone(), b)
        self.assertTrue(torch.equal(third, b))
        self.assertEqual(blender.cuts, 1)

    def test_temporal_blender_skips_moving_pixels(self):
        blender = vu.TemporalBlender(alpha=0.5, cut_threshold=0.9)
        a = torch.zeros(1, 3, 32, 32)
        blender(a, a)
        b = torch.zeros(1, 3, 32, 32)
        b[:, :, :, 16:] = 0.5                                         # right half moved a lot
        out = blender(b.clone(), b)
        self.assertTrue(torch.equal(out[:, :, :, 24:], b[:, :, :, 24:]))   # moving region untouched

    @staticmethod
    def _faded(x):
        # Simulate an old tape: yellow cast, lifted blacks / dull whites, low saturation.
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = y + (x - y) * 0.5
        x = x * torch.tensor([1.15, 1.05, 0.80]).view(1, 3, 1, 1)
        return (x * 0.5 + 0.2).clamp(0, 1)

    def test_color_restorer_fixes_cast_contrast_and_saturation(self):
        torch.manual_seed(0)
        clean = torch.rand(1, 3, 48, 64)
        faded = self._faded(clean)
        restorer = vu.ColorRestorer(strength=1.0)
        fixed = restorer(faded, faded)
        # Color cast: channel means move much closer together.
        cast_before = float(faded.mean((2, 3)).std())
        cast_after = float(fixed.mean((2, 3)).std())
        self.assertLess(cast_after, cast_before * 0.5)
        # Contrast: the luminance range is stretched back toward full scale.
        self.assertGreater(float(fixed.max() - fixed.min()), float(faded.max() - faded.min()) + 0.3)
        # Saturation: per-pixel chroma goes up.
        chroma = lambda t: float((t.amax(1) - t.amin(1)).mean())
        self.assertGreater(chroma(fixed), chroma(faded) * 1.3)
        # Strength 0 is the identity and the restorer never produces out-of-range values.
        self.assertTrue(torch.equal(vu.ColorRestorer(strength=0.0)(faded, faded), faded))
        self.assertTrue(bool((fixed >= 0).all() and (fixed <= 1).all()))

    def test_color_restorer_is_temporally_smooth_and_resets_on_cut(self):
        torch.manual_seed(1)
        a = self._faded(torch.rand(1, 3, 32, 32))
        restorer = vu.ColorRestorer(strength=1.0, smoothing=0.1)
        first = restorer(a, a)
        # Same source again -> identical parameters -> identical output (no flicker on a still shot).
        self.assertTrue(torch.allclose(restorer(a, a), first, atol=1e-5))
        # A slightly different frame changes the parameters only a little (EMA), not all at once.
        b = (a * 0.9).clamp(0, 1)
        white_before = restorer.white
        restorer(b, b)
        _, _, white_b = vu.ColorRestorer._measure(b)
        self.assertLess(abs(restorer.white - white_before), abs(white_b - white_before))
        # A scene cut re-measures from scratch.
        dark = torch.full((1, 3, 32, 32), 0.05)
        restorer(dark, dark)
        self.assertEqual(restorer.cuts, 1)
        _, _, white_dark = vu.ColorRestorer._measure(dark)
        self.assertAlmostEqual(restorer.white, white_dark, places=5)

    def test_color_restorer_applies_source_stats_to_large_output(self):
        torch.manual_seed(2)
        src = self._faded(torch.rand(1, 3, 16, 16))
        big = F.interpolate(src, scale_factor=4, mode='nearest')
        restorer = vu.ColorRestorer(strength=0.7)
        out = restorer(big, src)
        self.assertEqual(out.shape, big.shape)
        self.assertFalse(torch.allclose(out, big))

    @staticmethod
    def _t(frame):
        return torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255.0

    def test_glitch_repair_fixes_blocks_and_garbage_frames_but_keeps_flashes(self):
        frames = _moving_square_frames(8)
        prev, cur, nxt = (self._t(f) for f in frames[2:5])
        repair = vu.GlitchRepair()
        # Clean frames pass through untouched.
        self.assertTrue(torch.equal(repair(prev, cur.clone(), nxt), cur))
        # A corrupted 16x16 block in the static background is replaced by the neighbours' content.
        damaged = cur.clone()
        damaged[..., 40:56, 40:56] = torch.tensor([0.0, 0.8, 0.0]).view(1, 3, 1, 1)   # green macroblock garbage
        fixed = repair(prev, damaged, nxt)
        self.assertLess(float((fixed - cur)[..., 40:56, 40:56].abs().mean()), 0.01)
        self.assertLess(float((fixed - cur).abs().mean()), 0.002)      # the moving square was left alone
        # A garbage frame (noise) is replaced entirely.
        torch.manual_seed(0)
        garbage = torch.rand_like(cur)
        fixed = repair(prev, garbage, nxt)
        self.assertEqual(repair.frames_replaced, 1)
        self.assertLess(float((fixed - 0.5 * (prev + nxt)).abs().max()), 1e-6)
        # A camera flash (whole frame brighter for one frame) is not damage and must survive.
        flash = (cur * 1.6).clamp(0, 1)
        self.assertTrue(torch.allclose(repair(prev, flash, nxt), flash))
        # First / last frame have no neighbour on one side: unchanged.
        self.assertTrue(torch.equal(repair(None, damaged, nxt), damaged))
        self.assertTrue(torch.equal(repair(prev, damaged, None), damaged))

    def test_with_lookahead(self):
        self.assertEqual(list(vu._with_lookahead([])), [])
        self.assertEqual(list(vu._with_lookahead([1])), [(1, None)])
        self.assertEqual(list(vu._with_lookahead([1, 2, 3])), [(1, 2), (2, 3), (3, None)])

    def test_sr_frame_tiled_matches_whole_frame(self):
        model = IdentityX4()
        x = torch.rand(1, 3, 70, 100)
        whole = vu.sr_frame(model, 4, x, vu.TileState(0, 0))
        tiled = vu.sr_frame(model, 4, x, vu.TileState(48, 16))
        self.assertEqual(tuple(whole.shape), (1, 3, 280, 400))
        self.assertTrue(torch.allclose(whole, tiled, atol=1e-5))

    def test_resize_to_target(self):
        x = torch.rand(1, 3, 40, 60)
        self.assertEqual(tuple(vu.resize_to_target(x, 30, 20).shape), (1, 3, 20, 30))
        self.assertIs(vu.resize_to_target(x, 60, 40), x)


@unittest.skipUnless(HAS_FFMPEG and HAS_TORCH, 'needs ffmpeg and torch')
class TestVideoUpscalerPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='fooocus_video_sr_')
        cls.clip = video_io.make_synthetic_video(os.path.join(cls.tmp, 'in.mp4'), 64, 64, 10, 10, audio=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pipeline_4x_h264(self):
        out = os.path.join(self.tmp, 'out_4x.mp4')
        progress = []
        result = vu.upscale_video(self.clip, out, _options(), sr_model=IdentityX4(),
                                  progress_cb=lambda *a: progress.append(a))
        self.assertEqual(result, out)
        info = video_io.probe_video(out)
        self.assertEqual((info.width, info.height), (256, 256))
        self.assertEqual(info.codec_name, 'h264')
        self.assertEqual(info.color_space, 'bt709')
        self.assertEqual(info.color_range, 'tv')
        self.assertTrue(info.has_audio)
        self.assertIn(info.nb_frames, (9, 10, 11))
        self.assertEqual(len(progress), 10)
        self.assertEqual(progress[-1][0], 10)
        self.assertIsNotNone(progress[0][3])                          # first call carries a preview image

        # Content check: output ~= nearest-neighbour upscale of the input (only codec loss).
        src = list(video_io.iter_frames(self.clip))
        dst = list(video_io.iter_frames(out))
        expected = np.repeat(np.repeat(src[5], 4, 0), 4, 1).astype(np.float32)
        self.assertLess(np.abs(dst[5].astype(np.float32) - expected).mean(), 6.0)

    def test_pipeline_with_color_fix_blend_and_h265(self):
        out = os.path.join(self.tmp, 'out_h265.mp4')
        vu.upscale_video(self.clip, out, _options(color_fix='wavelet', temporal_blend=0.25, codec='h265', crf=24),
                         sr_model=IdentityX4())
        info = video_io.probe_video(out)
        self.assertEqual((info.width, info.height), (256, 256))
        self.assertIn(info.codec_name, ('hevc', 'h264'))              # h264 when libx265 is not built in
        if info.codec_name == 'hevc':
            self.assertEqual(info.pix_fmt, 'yuv420p10le')

    def test_odd_input_size_and_no_audio(self):
        src = video_io.make_synthetic_video(os.path.join(self.tmp, 'odd.mp4'), 63, 65, 4, 10, audio=False,
                                            codec='libx264rgb')
        info = video_io.probe_video(src)
        self.assertFalse(info.has_audio)
        out = os.path.join(self.tmp, 'odd_out.mp4')
        vu.upscale_video(src, out, _options(target='2x'), sr_model=IdentityX4())
        out_info = video_io.probe_video(out)
        exp_w, exp_h, _ = vu.compute_target(info.width, info.height, '2x')
        self.assertEqual((out_info.width, out_info.height), (exp_w, exp_h))
        self.assertEqual(out_info.width % 2 + out_info.height % 2, 0)
        self.assertFalse(out_info.has_audio)

    def test_resize_only_when_input_already_large(self):
        src = video_io.make_synthetic_video(os.path.join(self.tmp, 'big.mp4'), 1920, 1080, 2, 10, audio=False)
        out = os.path.join(self.tmp, 'big_out.mp4')
        vu.upscale_video(src, out, _options(target='1080p'), sr_model=IdentityX4())
        info = video_io.probe_video(out)
        self.assertEqual((info.width, info.height), (1920, 1080))

    def test_trim_color_restore_and_non_square_pixels(self):
        src = video_io.make_synthetic_video(os.path.join(self.tmp, 'tape.mp4'), 88, 72, 30, 10, audio=True,
                                            sar='12/11')
        out = os.path.join(self.tmp, 'tape_out.mp4')
        progress = []
        vu.upscale_video(src, out, _options(target='2x', color_restore=0.7, start=1.0, duration=1.0),
                         sr_model=IdentityX4(), progress_cb=lambda *a: progress.append(a))
        info = video_io.probe_video(out)
        self.assertEqual((info.width, info.height), (192, 144))          # 96x72 square pixels, 2x -> 4:3
        self.assertIn(info.nb_frames, (9, 10, 11))
        self.assertEqual(progress[0][1], 10)                              # total frames reflects the trim
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 1.0, delta=0.25)

    def test_pipeline_repairs_damaged_source(self):
        frames = _moving_square_frames(12)
        damaged = [f.copy() for f in frames]
        damaged[5][40:56, 40:56] = (0, 200, 0)                           # corrupted block
        damaged[8] = frames[7]                                           # one stuck frame
        src = _write_frames(os.path.join(self.tmp, 'damaged.mp4'), damaged)
        out = os.path.join(self.tmp, 'damaged_out.mp4')
        vu.upscale_video(src, out, _options(target='2x', repair_glitches=True, repair_freezes=True),
                         sr_model=IdentityX4())
        result = list(video_io.iter_frames(out))
        self.assertEqual(len(result), 12)
        truth = [np.repeat(np.repeat(f, 2, 0), 2, 1).astype(np.float32) for f in frames]
        diff = lambda a, b: np.abs(a.astype(np.float32) - b).mean()
        self.assertLess(diff(result[5][80:112, 80:112], truth[5][80:112, 80:112]), 6.0)   # block repaired
        self.assertLess(diff(result[8], truth[8]), diff(np.repeat(np.repeat(damaged[8], 2, 0), 2, 1), truth[8]))

    def test_start_beyond_end_fails_cleanly(self):
        out = os.path.join(self.tmp, 'empty.mp4')
        with self.assertRaises(vu.VideoUpscaleError):
            vu.upscale_video(self.clip, out, _options(start=100.0), sr_model=IdentityX4())
        self.assertFalse(os.path.exists(out))
        self.assertFalse(any(name.endswith('.part.mp4') for name in os.listdir(self.tmp)))

    def test_cancel_cleans_up(self):
        out = os.path.join(self.tmp, 'cancelled.mp4')
        calls = {'n': 0}

        def should_stop():
            calls['n'] += 1
            return calls['n'] >= 3

        with self.assertRaises(vu.VideoUpscaleCancelled):
            vu.upscale_video(self.clip, out, _options(), sr_model=IdentityX4(), should_stop=should_stop)
        self.assertFalse(os.path.exists(out))
        self.assertFalse(any(name.endswith('.part.mp4') for name in os.listdir(self.tmp)))

    def test_hdr_input_is_refused(self):
        src = os.path.join(self.tmp, 'hdr.mp4')
        import subprocess
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc2=size=64x64:rate=10', '-frames:v', '2',
                        '-vf', 'setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,'
                               'format=yuv420p10le',
                        '-c:v', 'libx264', src], check=True)
        self.assertTrue(video_io.probe_video(src).is_hdr)
        with self.assertRaises(vu.VideoUpscaleError):
            vu.upscale_video(src, os.path.join(self.tmp, 'hdr_out.mp4'), _options(), sr_model=IdentityX4())


if __name__ == '__main__':
    unittest.main()
