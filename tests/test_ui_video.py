import os
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction

from modules.video_io import VideoInfo

try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

if HAS_GRADIO:
    _argv = sys.argv
    sys.argv = sys.argv[:1]          # modules.config parses Fooocus flags on import
    try:
        from modules import ui_video
    finally:
        sys.argv = _argv


@unittest.skipUnless(HAS_GRADIO, 'gradio not installed')
class TestUiVideoHelpers(unittest.TestCase):
    def test_format_eta(self):
        self.assertEqual(ui_video.format_eta(None), '--:--')
        self.assertEqual(ui_video.format_eta(0), '00:00')
        self.assertEqual(ui_video.format_eta(83), '01:23')
        self.assertEqual(ui_video.format_eta(3723), '1:02:03')

    def test_estimate_output_size_mb(self):
        base = ui_video.estimate_output_size_mb(3840, 2160, 30, 60, 'h264', 18)
        self.assertGreater(base, 100)
        self.assertLess(ui_video.estimate_output_size_mb(3840, 2160, 30, 60, 'h265', 18), base)
        self.assertAlmostEqual(ui_video.estimate_output_size_mb(3840, 2160, 30, 60, 'h264', 24), base / 2, places=3)
        self.assertEqual(ui_video.estimate_output_size_mb(3840, 2160, 30, None, 'h264', 18), 0)

    def test_format_probe_html(self):
        info = VideoInfo(width=1920, height=1080, fps=Fraction(30000, 1001), nb_frames=300, duration=10.0,
                         has_audio=True, audio_codec='aac')
        text = ui_video.format_probe_html(info, 3840, 2160, True, 42.0)
        self.assertIn('1920x1080', text)
        self.assertIn('3840x2160', text)
        self.assertIn('300 frames', text)
        self.assertIn('42 MB', text)
        self.assertNotIn('re-encoded', text)
        self.assertIn('re-encoded', ui_video.format_probe_html(info, 1920, 1080, False, 1.0))

    def test_make_error_html_escapes(self):
        self.assertIn('&lt;b&gt;', ui_video.make_error_html('<b>'))

    def test_build_tab_and_bind_events(self):
        import gradio as gr
        with gr.Blocks():
            generate_button = gr.Button(value='Generate')
            with gr.Tabs():
                tab, ctrls = ui_video.build_video_tab()
            ui_video.bind_events(ctrls, generate_button)
        self.assertEqual(ctrls['target'].value, '4K')
        self.assertEqual(ctrls['codec'].value, 'h264')
        self.assertIn('realesr-general-x4v3.pth', ctrls['model'].choices)

    def test_run_without_video_reports_error(self):
        outputs = list(ui_video.run_video_upscale(None, 'm', '4K', 'wavelet', 0.25, False, 0.7, 'h264', 18))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 5)
        self.assertIn('upload a video', outputs[0][0]['value'])

    @unittest.skipUnless(os.environ.get('FOOOCUS_TEST_SR_MODEL') and shutil.which('ffmpeg'),
                         'set FOOOCUS_TEST_SR_MODEL=/path/to/model.pth to run the end-to-end UI test')
    def test_run_end_to_end(self):
        import modules.config
        from modules import video_io
        tmp = tempfile.mkdtemp(prefix='fooocus_ui_video_')
        original_outputs = modules.config.path_outputs
        modules.config.path_outputs = tmp
        try:
            clip = video_io.make_synthetic_video(os.path.join(tmp, 'in.mp4'), 96, 64, 5, 10, audio=True)
            outputs = list(ui_video.run_video_upscale(clip, os.environ['FOOOCUS_TEST_SR_MODEL'], '2x', 'wavelet',
                                                      0.25, False, 0.7, 'h264', 20))
            final = outputs[-1]
            self.assertEqual(final[0]['visible'], False)              # progress hidden
            self.assertEqual(final[2]['visible'], True)               # h264 result is playable
            result_path = final[2]['value']
            self.assertTrue(result_path.startswith(tmp))
            info = video_io.probe_video(result_path)
            self.assertEqual((info.width, info.height), (192, 128))
            self.assertIn('Download', final[4]['value'])
            self.assertTrue(any('Frame' in o[0].get('value', '') for o in outputs[1:-1]))   # progress updates
        finally:
            modules.config.path_outputs = original_outputs
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
