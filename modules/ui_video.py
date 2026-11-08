"""
Gradio UI for the Video Upscale tab.

The job runs in its own thread straight from the click handler (like the Describe tab), not through
the image-generation worker, so nothing here touches the `ctrls` list consumed by AsyncTask.
"""

import html
import os
import queue
import sys
import threading
import time
import traceback

import gradio as gr

import modules.config
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.html
from modules.util import generate_temp_filename
from modules.video_io import ffmpeg_available, probe_video

_cancel_event = threading.Event()
_running = threading.Event()

FFMPEG_MISSING = ('ffmpeg/ffprobe not found on PATH. Install ffmpeg (brew install ffmpeg / '
                  'apt install ffmpeg / winget install ffmpeg) and restart Fooocus.')


# ----------------------------------------------------------------------------- pure helpers

def format_eta(seconds) -> str:
    if seconds is None:
        return '--:--'
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def estimate_output_size_mb(width: int, height: int, fps: float, duration: float, codec: str, crf: int) -> float:
    """Rough file size estimate from bits-per-pixel at CRF 18, halved every 6 CRF steps."""
    bits_per_pixel = 0.06 if codec == 'h265' else 0.10
    bits_per_pixel *= 2 ** ((18 - int(crf)) / 6.0)
    return width * height * float(fps) * float(duration or 0) * bits_per_pixel / 8 / 1e6


def format_probe_html(info, out_w: int, out_h: int, run_sr: bool, size_mb: float,
                      start: float = 0.0, duration=None) -> str:
    total = info.estimated_frames(start, duration)
    seconds = info.trimmed_duration(start, duration)
    duration_text = f'{seconds:.1f} s' if seconds is not None else '? s'
    source = f'{info.width}x{info.height}'
    if info.has_non_square_pixels:
        source = f'{info.stored_width}x{info.stored_height} (non-square pixels, processed as {source})'
    text = (f'{source} @ {float(info.fps):.2f} fps, {total if total else "?"} frames, {duration_text}'
            f' &rarr; <b>{out_w}x{out_h}</b>, estimated output ~{size_mb:.0f} MB.')
    if start or (duration and duration > 0):
        text += f' Processing a section only (from {start:g} s' + (f' for {duration:g} s' if duration and duration > 0 else '') + ').'
    if not run_sr:
        text += ' Input already covers the target: it will only be re-encoded.'
    if info.is_hdr:
        text += ' <b>HDR input is not supported.</b>'
    if info.is_vfr_hint:
        text += ' Variable frame rate detected; output will use a constant frame rate.'
    return f'<div style="opacity:0.85">{text}</div>'


def make_error_html(text: str) -> str:
    return f'<div class="loader-container"><span style="white-space:pre-wrap">{html.escape(text)}</span></div>'


def list_video_models() -> list:
    try:
        from modules.video_upscaler import list_sr_models
        return list_sr_models()
    except Exception as e:
        print(f'[Video Upscale] Could not list models: {e}')
        return []


# ----------------------------------------------------------------------------- tab

def build_video_tab():
    """Call inside the input-image `gr.Tabs` block. Returns (tab, components dict)."""
    model_choices = list_video_models()
    default_model = modules.config.default_video_upscale_model
    if default_model not in model_choices and len(model_choices) > 0:
        default_model = model_choices[0]

    with gr.Tab(label='Video Upscale', id='video_upscale_tab') as video_tab:
        with gr.Row():
            with gr.Column():
                # gr.File instead of gr.Video: the video component decodes the whole file in the browser for a
                # preview, which crashes the tab ("Aw, Snap") for large videos.
                video_input = gr.File(label='Video file', file_types=['video'], file_count='single', interactive=True)
                video_path_box = gr.Textbox(label='...or path of a video already on the server',
                                            placeholder='/content/drive/MyDrive/clip.mp4  (for large files: upload via the Colab file sidebar or Google Drive)',
                                            lines=1)
                video_info = gr.HTML(value='', elem_id='video_probe_info')
                video_progress = gr.HTML(value='', visible=False, elem_classes='progress-bar')
                video_preview = grh.Image(label='Frame Preview', show_label=True, visible=False, interactive=False)
                video_output = gr.Video(label='Result', show_label=True, visible=False, interactive=False)
                video_file = gr.File(label='Download', visible=False, interactive=False)
                video_download_html = gr.HTML(value='')
            with gr.Column():
                with gr.Row():
                    video_model = gr.Dropdown(label='Upscale Model', choices=model_choices, value=default_model, scale=5)
                    video_model_refresh = gr.Button(value='\U0001f504', scale=1, variant='secondary',
                                                    elem_classes='refresh_button')
                video_target = gr.Radio(label='Target Resolution', choices=flags.video_upscale_targets,
                                        value=modules.config.default_video_upscale_target)
                video_color_fix = gr.Dropdown(label='Color Fix', choices=flags.video_color_fix_options,
                                              value=modules.config.default_video_color_fix,
                                              info='Keeps the colors of the original frames.')
                video_color_restore = gr.Slider(label='Restore Faded Colors', minimum=0.0, maximum=1.0, step=0.05,
                                                value=modules.config.default_video_color_restore,
                                                info='For old camcorder/VHS/VCD footage: removes the color cast, '
                                                     'restores contrast and saturation (0 = keep the original colors, '
                                                     'try 0.7).')
                video_temporal_blend = gr.Slider(label='Temporal Blend', minimum=0.0, maximum=1.0, step=0.05,
                                                 value=modules.config.default_video_temporal_blend,
                                                 info='Blends still areas with the previous frame to reduce flicker.')
                video_face_enhance = gr.Checkbox(label='Face Enhance (GFPGAN)',
                                                 value=modules.config.default_video_face_enhance)
                video_face_strength = gr.Slider(label='Face Enhance Strength', minimum=0.0, maximum=1.0, step=0.05,
                                                value=modules.config.default_video_face_strength,
                                                visible=modules.config.default_video_face_enhance)
                video_codec = gr.Radio(label='Codec', choices=flags.video_codecs,
                                       value=modules.config.default_video_codec,
                                       info='h264 plays in the browser; h265 is 10-bit and smaller (download only).')
                video_crf = gr.Slider(label='Quality (CRF, lower is better)', minimum=0, maximum=51, step=1,
                                      value=modules.config.default_video_crf)
                with gr.Row():
                    video_repair_glitches = gr.Checkbox(label='Repair Glitches',
                                                        value=False, info='Damaged disc/tape: fix corrupted blocks.')
                    video_repair_freezes = gr.Checkbox(label='Repair Freezes',
                                                       value=False, info='Motion-interpolate short stuck frames.')
                with gr.Row():
                    video_start = gr.Number(label='Start (seconds)', value=0, minimum=0, precision=2,
                                            info='Test your settings on a short section first.')
                    video_duration = gr.Number(label='Duration (seconds, 0 = to the end)', value=0, minimum=0,
                                               precision=2)
                with gr.Row():
                    video_btn = gr.Button(value='Upscale Video', variant='primary', elem_id='video_upscale_button')
                    video_stop_btn = gr.Button(value='Stop', elem_id='video_stop_button', visible=False)
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">'
                        '\U0001F4D4 Documentation</a>')

    components = dict(
        tab=video_tab, input=video_input, path=video_path_box, info=video_info, progress=video_progress, preview=video_preview,
        output=video_output, file=video_file, download_html=video_download_html,
        model=video_model, model_refresh=video_model_refresh, target=video_target, color_fix=video_color_fix,
        temporal_blend=video_temporal_blend, color_restore=video_color_restore,
        face_enhance=video_face_enhance, face_strength=video_face_strength,
        codec=video_codec, crf=video_crf, start=video_start, duration=video_duration,
        repair_glitches=video_repair_glitches, repair_freezes=video_repair_freezes,
        btn=video_btn, stop_btn=video_stop_btn)
    return video_tab, components


# ----------------------------------------------------------------------------- handlers

def resolve_input_path(upload, path_text) -> str:
    """The server path typed by the user wins; otherwise the uploaded file (gr.File object or path)."""
    if path_text and str(path_text).strip():
        return os.path.expanduser(str(path_text).strip())
    if upload is None:
        return ''
    if isinstance(upload, (list, tuple)):
        upload = upload[0] if upload else None
    if isinstance(upload, dict):
        return str(upload.get('name') or upload.get('path') or '')
    return str(getattr(upload, 'name', upload) or '')


def _trim_values(start, duration):
    try:
        start = max(0.0, float(start or 0))
    except (TypeError, ValueError):
        start = 0.0
    try:
        duration = float(duration or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return start, (duration if duration > 0 else None)


def trigger_video_probe(upload, path_text, target, codec, crf, start=0, duration=0):
    video_path = resolve_input_path(upload, path_text)
    if not video_path:
        return gr.update(value='')
    if not ffmpeg_available():
        return gr.update(value=make_error_html(FFMPEG_MISSING))
    if not os.path.isfile(video_path):
        return gr.update(value=make_error_html(f'File not found on the server: {video_path}'))
    try:
        from modules.video_upscaler import compute_target
        info = probe_video(video_path)
        start, duration = _trim_values(start, duration)
        out_w, out_h, run_sr = compute_target(info.width, info.height, target)
        size_mb = estimate_output_size_mb(out_w, out_h, info.fps, info.trimmed_duration(start, duration), codec, crf)
        return gr.update(value=format_probe_html(info, out_w, out_h, run_sr, size_mb, start, duration))
    except Exception as e:
        traceback.print_exc()
        return gr.update(value=make_error_html(f'Could not read video: {e}'))


def stop_video_clicked():
    _cancel_event.set()
    return gr.update(interactive=False)


def run_video_upscale(upload, path_text, model_name, target, color_fix, temporal_blend,
                      face_enhance, face_strength, codec, crf, color_restore=0.0, start=0, duration=0,
                      repair_glitches=False, repair_freezes=False):
    video_path = resolve_input_path(upload, path_text)
    start, duration = _trim_values(start, duration)
    # outputs = [progress, preview, output video, download file, download link]
    def ui(progress=None, preview=None, output=None, file=None, link=None):
        return (progress if progress is not None else gr.update(),
                preview if preview is not None else gr.update(),
                output if output is not None else gr.update(),
                file if file is not None else gr.update(),
                link if link is not None else gr.update())

    def error(text):
        return ui(progress=gr.update(visible=True, value=make_error_html(text)), preview=gr.update(visible=False))

    if not video_path:
        yield error('Please upload a video first (or enter the path of a video on the server).')
        return
    if not os.path.isfile(video_path):
        yield error(f'File not found on the server: {video_path}')
        return
    if not ffmpeg_available():
        yield error(FFMPEG_MISSING)
        return
    if _running.is_set():
        yield error('A video upscale is already running.')
        return

    worker = sys.modules.get('modules.async_worker')   # only present inside the full web UI
    if worker is not None and len(worker.async_tasks) > 0:
        yield error('Please wait for the current image generation to finish.')
        return

    _running.set()
    _cancel_event.clear()
    events = queue.Queue()
    result = {}
    start_time = time.perf_counter()

    def progress_cb(done, total, eta, preview):
        pct = int(100 * done / total) if total else 0
        total_text = f'/{total}' if total else ''
        events.put(('preview', (pct, f'Frame {done}{total_text}, ETA {format_eta(eta)} ...', preview)))

    def job():
        try:
            from modules.video_upscaler import VideoUpscaleOptions, VideoUpscaleCancelled, upscale_video
            _, output_path, _ = generate_temp_filename(folder=modules.config.path_outputs, extension='mp4')
            options = VideoUpscaleOptions(
                model_name=model_name, target=target, color_fix=color_fix, temporal_blend=float(temporal_blend),
                face_enhance=bool(face_enhance), face_strength=float(face_strength), codec=codec, crf=int(crf),
                color_restore=float(color_restore or 0.0), start=start, duration=duration,
                repair_glitches=bool(repair_glitches), repair_freezes=bool(repair_freezes),
                preset=modules.config.default_video_encoder_preset,
                fp16=None if modules.config.default_video_fp16 else False)
            result['path'] = upscale_video(video_path, output_path, options,
                                           progress_cb=progress_cb, should_stop=_cancel_event.is_set)
        except VideoUpscaleCancelled:
            result['cancelled'] = True
        except Exception as e:
            traceback.print_exc()
            result['error'] = e
        finally:
            events.put(('finish', None))

    yield ui(progress=gr.update(visible=True, value=modules.html.make_progress_html(1, 'Loading models ...')),
             preview=gr.update(visible=True, value=None), output=gr.update(visible=False, value=None),
             file=gr.update(visible=False, value=None), link=gr.update(value=''))

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    finished = False
    try:
        while not finished:
            time.sleep(0.01)
            try:
                flag, product = events.get_nowait()
            except queue.Empty:
                continue
            if flag == 'preview':
                if not events.empty():   # skip stale previews when the browser is behind
                    continue
                pct, text, image = product
                yield ui(progress=gr.update(visible=True, value=modules.html.make_progress_html(pct, text)),
                         preview=gr.update(visible=True, value=image) if image is not None else None)
            elif flag == 'finish':
                finished = True
        thread.join()
    finally:
        if not finished:            # generator closed early (client disconnected): stop the job
            _cancel_event.set()
            thread.join(timeout=30)
        _running.clear()

    if result.get('cancelled'):
        yield error('Video upscale stopped.')
        return
    if 'error' in result:
        yield error(f'Video upscale failed: {result["error"]}\nSee the console for details.')
        return

    output_path = result['path']
    print(f'[Video Upscale] Finished in {time.perf_counter() - start_time:.1f}s: {output_path}')
    name = os.path.basename(output_path)
    link = (f'<a href="file={output_path}" download="{name}" target="_blank">'
            f'\U0001F4BE Download {html.escape(name)}</a>')
    playable = codec == 'h264'
    yield ui(progress=gr.update(visible=False), preview=gr.update(visible=False),
             output=gr.update(visible=playable, value=output_path if playable else None),
             file=gr.update(visible=True, value=output_path), link=gr.update(value=link))


def bind_events(c: dict, generate_button):
    c['face_enhance'].change(lambda x: gr.update(visible=x), inputs=c['face_enhance'],
                             outputs=c['face_strength'], queue=False, show_progress=False)
    c['model_refresh'].click(lambda: gr.update(choices=list_video_models()), outputs=c['model'],
                             queue=False, show_progress=False)

    probe_inputs = [c['input'], c['path'], c['target'], c['codec'], c['crf'], c['start'], c['duration']]
    c['input'].upload(trigger_video_probe, inputs=probe_inputs, outputs=c['info'], queue=False, show_progress=False)
    c['input'].clear(trigger_video_probe, inputs=probe_inputs, outputs=c['info'], queue=False, show_progress=False)
    c['path'].submit(trigger_video_probe, inputs=probe_inputs, outputs=c['info'], queue=False, show_progress=False)
    c['path'].blur(trigger_video_probe, inputs=probe_inputs, outputs=c['info'], queue=False, show_progress=False)
    for component in (c['target'], c['codec'], c['crf'], c['start'], c['duration']):
        component.change(trigger_video_probe, inputs=probe_inputs, outputs=c['info'], queue=False, show_progress=False)

    c['btn'].click(lambda: (gr.update(visible=False, interactive=False), gr.update(visible=True, interactive=True),
                            gr.update(interactive=False)),
                   outputs=[c['btn'], c['stop_btn'], generate_button], queue=False, show_progress=False) \
        .then(run_video_upscale,
              inputs=[c['input'], c['path'], c['model'], c['target'], c['color_fix'], c['temporal_blend'],
                      c['face_enhance'], c['face_strength'], c['codec'], c['crf'], c['color_restore'],
                      c['start'], c['duration'], c['repair_glitches'], c['repair_freezes']],
              outputs=[c['progress'], c['preview'], c['output'], c['file'], c['download_html']],
              show_progress=False, queue=True) \
        .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False),
                       gr.update(interactive=True)),
              outputs=[c['btn'], c['stop_btn'], generate_button], queue=False, show_progress=False) \
        .then(fn=lambda: None, _js='playNotification')

    c['stop_btn'].click(stop_video_clicked, outputs=c['stop_btn'], queue=False, show_progress=False)
