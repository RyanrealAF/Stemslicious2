# ENTRY POINT: app.py — StemToMIDI (Polyphonic AMT)
# Engine: piano_transcription_inference (Kong et al. 2020)
#   - CRNN frame-level estimator: polyphonic, velocity-aware, pedal-aware
#   - F1=0.9677 on MAPS dataset
#   - Model: ~165MB, downloaded on first run to ~/piano_transcription_inference_data/
# Preprocessing: HPSS transient suppression via librosa
# Compatible: Gradio 6.x, Python 3.10, CPU inference

import os
import tempfile
from typing import Any

import numpy as np
import gradio as gr
import librosa
import librosa.decompose
import torch

from piano_transcription_inference import PianoTranscription, sample_rate as PTI_SR

# ── CONFIG ────────────────────────────────────────────────────────────────────
STEM_TYPES = ["melodic / piano", "bass", "vocal", "drums / percussive"]

# HPSS transient suppression defaults per stem
# 0.0 = off (pass-through), 1.0 = harmonic only
TRANSIENT_BLEND_MAP = {
    "melodic / piano":    0.6,   # moderate — preserve attack transients for onset accuracy
    "bass":               0.75,
    "vocal":              0.80,
    "drums / percussive": 0.0,   # off — transients ARE the signal
}

HPSS_MARGIN_MAP = {
    "melodic / piano":    2.0,
    "bass":               2.5,
    "vocal":              3.0,
    "drums / percussive": 1.0,
}

# AMT post-processor thresholds — exposed to UI
# Lower onset_threshold = more notes detected (higher false positive risk)
# Higher frame_threshold = shorter notes suppressed
DEFAULT_ONSET_THRESH  = 0.3
DEFAULT_OFFSET_THRESH = 0.3
DEFAULT_FRAME_THRESH  = 0.1

PTI_SAMPLE_RATE = PTI_SR   # 16000 Hz — model requirement
PREPROCESS_SR   = 22050    # librosa working rate before downsampling to PTI_SR
MAX_ANALYSIS_DURATION_SEC = 90

MUSIC_TOOL_CATALOG = [
    {
        "name": "StemToMIDI",
        "category": "midi_conversion",
        "description": "Analyze pitched stems, recommend AMT settings, and convert stems to MIDI.",
        "space_url": "https://huggingface.co/spaces/Ryanrealaf/Midi",
        "mcp_url": "https://ryanrealaf-midi.hf.space/gradio_api/mcp/sse",
        "tools": [
            "list_music_tool_hub",
            "recommend_midi_settings",
            "convert_stem_to_midi",
            "apply_stem_preset",
        ],
        "status": "enabled in this Space",
    },
    {
        "name": "Master Stem Splitter",
        "category": "six_stem_separation",
        "description": "Split a full mix into stems before selecting tracks for MIDI conversion.",
        "space_url": "https://huggingface.co/spaces/reallyrogueradio/Master-Stem-Splitter",
        "mcp_url": "https://reallyrogueradio-master-stem-splitter.hf.space/gradio_api/mcp/sse",
        "tools": [
            "split_master_stems",
            "retrieve_stems_zip",
            "deconstruct_drums",
            "retrieve_drums_zip",
            "analyze_flow_and_groove",
        ],
        "status": "companion Space; enable MCP there before connecting from ChatGPT",
    },
]

TOOL_HUB_GOALS = [
    "full song to MIDI",
    "6-stem separation",
    "MIDI conversion",
    "drum deconstruction",
    "flow and groove analysis",
]

# ─────────────────────────────────────────────────────────────────────────────

# ── MODEL SINGLETON ───────────────────────────────────────────────────────────
# Instantiated once at module load — avoids 165MB re-download per inference call
_transcriber = None

def get_transcriber(onset_thresh: float, offset_thresh: float, frame_thresh: float):
    '''
    Return a PianoTranscription instance with updated thresholds.
    Model weights are cached after first load; only threshold attributes are patched.
    '''
    global _transcriber
    if _transcriber is None:
        print("Loading PianoTranscription model (~165MB on first run)...")
        _transcriber = PianoTranscription(
            model_type="Note_pedal",
            device=torch.device("cpu"),
        )
        print("Model loaded.")

    # Patch thresholds without reloading weights
    _transcriber.onset_threshold       = onset_thresh
    _transcriber.offset_threshod       = offset_thresh   # sic — typo in upstream source
    _transcriber.frame_threshold       = frame_thresh

    return _transcriber


def list_music_tool_hub(goal: str = "full song to MIDI") -> tuple[list[dict[str, Any]], str]:
    '''
    Return the available music-production MCP tools and a recommended workflow.

    Args:
        goal (str): Desired workflow, such as full song to MIDI, 6-stem separation,
            MIDI conversion, drum deconstruction, or flow and groove analysis.

    Returns:
        tuple[list[dict[str, Any]], str]: Tool catalog entries and a concise workflow
        for the selected goal.
    '''
    normalized_goal = (goal or "full song to MIDI").strip().lower()

    if "full" in normalized_goal or "song" in normalized_goal:
        workflow = (
            "For a full song, split it first with Master Stem Splitter: call "
            "`split_master_stems`, wait for the job to finish, then call "
            "`retrieve_stems_zip`. Choose the melodic/bass/vocal stem that should "
            "become MIDI, then run StemToMIDI's `recommend_midi_settings` and "
            "`convert_stem_to_midi`."
        )
    elif "6" in normalized_goal or "stem" in normalized_goal or "split" in normalized_goal:
        workflow = (
            "Use Master Stem Splitter first: call `split_master_stems`, wait for the "
            "job to finish, then call `retrieve_stems_zip`. Feed the best melodic, "
            "bass, or vocal stem into StemToMIDI when you want MIDI."
        )
    elif "drum" in normalized_goal:
        workflow = (
            "Use Master Stem Splitter's drum tools first: call `deconstruct_drums`, "
            "then `retrieve_drums_zip`. For pitched MIDI, use StemToMIDI on non-drum "
            "stems because the current AMT model is optimized for pitched instruments."
        )
    elif "flow" in normalized_goal or "groove" in normalized_goal:
        workflow = (
            "Use Master Stem Splitter's `analyze_flow_and_groove` with vocal and drum "
            "audio. Use StemToMIDI afterward only if you also need pitched MIDI from a "
            "melodic, bass, or vocal stem."
        )
    elif "midi" in normalized_goal:
        workflow = (
            "Use StemToMIDI directly for an already-isolated stem: call "
            "`recommend_midi_settings`, review the suggested thresholds, then call "
            "`convert_stem_to_midi` with those values."
        )
    else:
        workflow = (
            "For a full song, split it first with Master Stem Splitter, retrieve the "
            "stem zip, choose the melodic/bass/vocal stem that should become MIDI, "
            "then run StemToMIDI's `recommend_midi_settings` and `convert_stem_to_midi`."
        )

    catalog = [dict(tool) for tool in MUSIC_TOOL_CATALOG]
    return catalog, workflow

# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    '''Load to mono float32 at PREPROCESS_SR.'''
    y, _ = librosa.load(path, sr=PREPROCESS_SR, mono=True, dtype=np.float32)
    return y

def normalize(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * ceiling
    return y

def suppress_transients(y: np.ndarray, blend: float, margin: float) -> np.ndarray:
    '''
    HPSS Wiener soft-mask.
    Separates harmonic (sustained pitch) from percussive (transient) content.
    blend=0 → unchanged. blend=1 → harmonic only.
    Soft mask avoids binary separation artifacts at note boundaries.
    '''
    if blend <= 0.0:
        return y

    D      = librosa.stft(y)
    H_mag, P_mag = librosa.decompose.hpss(np.abs(D), margin=margin)

    eps        = 1e-8
    soft_mask  = H_mag / (H_mag + P_mag + eps)
    y_harmonic = librosa.istft(D * soft_mask, length=len(y))

    min_len = min(len(y), len(y_harmonic))
    return (1.0 - blend) * y[:min_len] + blend * y_harmonic[:min_len]

def preprocess(path: str, blend: float, margin: float) -> np.ndarray:
    '''Full chain: load → HPSS suppress → normalize → resample to PTI SR.'''
    y = load_audio(path)
    y = suppress_transients(y, blend=blend, margin=margin)
    y = normalize(y)
    # Downsample to 16kHz for PianoTranscription model
    y = librosa.resample(y, orig_sr=PREPROCESS_SR, target_sr=PTI_SAMPLE_RATE)
    return y.astype(np.float32)

def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

def _safe_float(value: Any, fallback: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return fallback
    return float(value)

def _format_recommendation_report(
    stem_guess: str,
    confidence_notes: list[str],
    settings: dict[str, float],
    metrics: dict[str, float],
) -> str:
    setting_lines = [
        "Recommended MIDI settings:",
        f"- Stem type: {stem_guess}",
        f"- Transient blend: {settings['transient_blend']:.2f}",
        f"- HPSS margin: {settings['hpss_margin']:.1f}",
        f"- Onset threshold: {settings['onset_threshold']:.2f}",
        f"- Offset threshold: {settings['offset_threshold']:.2f}",
        f"- Frame threshold: {settings['frame_threshold']:.2f}",
    ]
    metric_lines = [
        "",
        "Audio analysis:",
        f"- Duration: {metrics['duration_sec']:.1f}s",
        f"- Peak level: {metrics['peak']:.2f}",
        f"- RMS level: {metrics['rms']:.3f}",
        f"- Harmonic share: {metrics['harmonic_share']:.2f}",
        f"- Percussive share: {metrics['percussive_share']:.2f}",
        f"- Low-frequency share: {metrics['low_frequency_share']:.2f}",
        f"- Onset density: {metrics['onset_density']:.2f} onsets/sec",
        f"- Estimated pitch coverage: {metrics['pitch_coverage']:.2f}",
        f"- Median spectral centroid: {metrics['spectral_centroid_hz']:.0f} Hz",
    ]
    reason_lines = ["", "Why these settings:", *[f"- {note}" for note in confidence_notes]]
    return "\n".join(setting_lines + metric_lines + reason_lines)

def recommend_midi_settings(
    audio_file: str | None,
    stem_hint: str = "auto",
) -> tuple[str, float, float, float, float, float, str]:
    '''
    Analyze an uploaded audio stem and recommend StemToMIDI settings.

    This MCP-friendly tool is intended for ChatGPT clients: provide an audio file,
    optionally include a stem hint, and use the returned settings with
    convert_stem_to_midi for a better first MIDI conversion pass.

    Args:
        audio_file (str | None): Local path or uploaded file reference for the stem audio.
        stem_hint (str): Optional hint: auto, melodic / piano, bass, vocal, or drums / percussive.

    Returns:
        tuple[str, float, float, float, float, float, str]: Recommended stem type,
        transient blend, HPSS margin, onset threshold, offset threshold, frame threshold,
        and an explanation report.
    '''
    if audio_file is None:
        report = "No audio file provided. Upload a stem audio file so I can analyze it."
        return (
            "melodic / piano",
            0.6,
            2.0,
            DEFAULT_ONSET_THRESH,
            DEFAULT_OFFSET_THRESH,
            DEFAULT_FRAME_THRESH,
            report,
        )

    try:
        y = load_audio(audio_file)
    except Exception as e:
        report = f"Audio analysis failed: {e}"
        return (
            "melodic / piano",
            0.6,
            2.0,
            DEFAULT_ONSET_THRESH,
            DEFAULT_OFFSET_THRESH,
            DEFAULT_FRAME_THRESH,
            report,
        )

    if y.size == 0:
        report = "The uploaded audio file appears to be empty."
        return (
            "melodic / piano",
            0.6,
            2.0,
            DEFAULT_ONSET_THRESH,
            DEFAULT_OFFSET_THRESH,
            DEFAULT_FRAME_THRESH,
            report,
        )

    analysis_samples = int(MAX_ANALYSIS_DURATION_SEC * PREPROCESS_SR)
    if len(y) > analysis_samples:
        y = y[:analysis_samples]

    peak = _safe_float(np.max(np.abs(y)))
    rms = _safe_float(np.sqrt(np.mean(np.square(y))))
    duration_sec = max(len(y) / PREPROCESS_SR, 1e-6)

    D = librosa.stft(y)
    magnitude = np.abs(D)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(magnitude)
    harmonic_energy = _safe_float(np.sum(harmonic_mag))
    percussive_energy = _safe_float(np.sum(percussive_mag))
    total_hpss_energy = harmonic_energy + percussive_energy + 1e-8
    harmonic_share = harmonic_energy / total_hpss_energy
    percussive_share = percussive_energy / total_hpss_energy

    freqs = librosa.fft_frequencies(sr=PREPROCESS_SR)
    low_frequency_share = _safe_float(
        np.sum(magnitude[freqs <= 180.0]) / (np.sum(magnitude) + 1e-8)
    )
    spectral_centroid = _safe_float(
        np.median(librosa.feature.spectral_centroid(S=magnitude, sr=PREPROCESS_SR))
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=PREPROCESS_SR)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=PREPROCESS_SR, units="time")
    onset_density = len(onsets) / duration_sec
    onset_strength = _safe_float(np.mean(onset_env), fallback=0.0)

    try:
        pitches, voiced_flags, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C7"),
            sr=PREPROCESS_SR,
        )
        pitch_coverage = _safe_float(np.mean(voiced_flags), fallback=0.0)
        voiced_pitches = pitches[voiced_flags]
        median_pitch_hz = (
            _safe_float(np.median(voiced_pitches), fallback=0.0)
            if voiced_pitches.size
            else 0.0
        )
    except Exception:
        pitch_coverage = 0.0
        median_pitch_hz = 0.0

    normalized_hint = (stem_hint or "auto").strip().lower()
    if normalized_hint in STEM_TYPES:
        stem_guess = normalized_hint
    elif "drum" in normalized_hint or "percuss" in normalized_hint:
        stem_guess = "drums / percussive"
    elif "bass" in normalized_hint:
        stem_guess = "bass"
    elif "vocal" in normalized_hint or "voice" in normalized_hint:
        stem_guess = "vocal"
    elif "piano" in normalized_hint or "melod" in normalized_hint:
        stem_guess = "melodic / piano"
    elif percussive_share > 0.58 and onset_density > 3.0 and pitch_coverage < 0.35:
        stem_guess = "drums / percussive"
    elif low_frequency_share > 0.42 and median_pitch_hz and median_pitch_hz < 180.0:
        stem_guess = "bass"
    elif pitch_coverage > 0.45 and harmonic_share > 0.56 and spectral_centroid > 1200.0:
        stem_guess = "vocal"
    else:
        stem_guess = "melodic / piano"

    transient_blend = TRANSIENT_BLEND_MAP[stem_guess]
    hpss_margin = HPSS_MARGIN_MAP[stem_guess]
    onset_threshold = DEFAULT_ONSET_THRESH
    offset_threshold = DEFAULT_OFFSET_THRESH
    frame_threshold = DEFAULT_FRAME_THRESH
    notes: list[str] = []

    if stem_guess == "drums / percussive":
        transient_blend = 0.0
        hpss_margin = 1.0
        onset_threshold = 0.25
        frame_threshold = 0.08
        notes.append(
            "The audio appears transient-heavy, so HPSS suppression is disabled "
            "to avoid removing the signal before conversion."
        )
        notes.append(
            "This piano AMT model is optimized for pitched notes; drums may still "
            "need a drum-specific MIDI workflow."
        )
    else:
        if percussive_share > 0.45 or onset_density > 3.5:
            transient_blend = _clamp(transient_blend + 0.15, 0.0, 0.9)
            hpss_margin = _clamp(hpss_margin + 0.5, 1.0, 6.0)
            onset_threshold = _clamp(onset_threshold + 0.05, 0.1, 0.9)
            notes.append(
                "Detected strong transients or drum bleed, so I increased harmonic "
                "blending and onset filtering."
            )
        elif harmonic_share > 0.65 and onset_density < 1.5:
            transient_blend = _clamp(transient_blend - 0.10, 0.0, 0.9)
            onset_threshold = _clamp(onset_threshold - 0.03, 0.1, 0.9)
            notes.append(
                "Detected a sustained harmonic signal, so I reduced transient "
                "suppression and lowered onset sensitivity for smoother notes."
            )

        if stem_guess == "bass" or low_frequency_share > 0.38:
            frame_threshold = _clamp(frame_threshold - 0.02, 0.05, 0.5)
            offset_threshold = _clamp(offset_threshold + 0.05, 0.1, 0.9)
            notes.append(
                "Low-frequency energy is prominent, so I lowered the frame threshold "
                "to retain quieter bass fundamentals."
            )

        if rms < 0.03 or peak < 0.20:
            onset_threshold = _clamp(onset_threshold - 0.05, 0.1, 0.9)
            frame_threshold = _clamp(frame_threshold - 0.02, 0.05, 0.5)
            notes.append(
                "The stem is quiet, so I lowered note-detection thresholds to catch "
                "weaker notes."
            )
        elif rms > 0.18 or peak > 0.98:
            onset_threshold = _clamp(onset_threshold + 0.05, 0.1, 0.9)
            notes.append(
                "The stem is loud or close to clipping, so I raised onset filtering "
                "to reduce false positives."
            )

    if not notes:
        notes.append(
            "The audio profile is close to the default target for this model, so the "
            "recommendation stays near the published defaults."
        )

    metrics = {
        "duration_sec": duration_sec,
        "peak": peak,
        "rms": rms,
        "harmonic_share": harmonic_share,
        "percussive_share": percussive_share,
        "low_frequency_share": low_frequency_share,
        "onset_density": onset_density,
        "onset_strength": onset_strength,
        "pitch_coverage": pitch_coverage,
        "spectral_centroid_hz": spectral_centroid,
    }
    settings = {
        "transient_blend": transient_blend,
        "hpss_margin": hpss_margin,
        "onset_threshold": onset_threshold,
        "offset_threshold": offset_threshold,
        "frame_threshold": frame_threshold,
    }
    report = _format_recommendation_report(stem_guess, notes, settings, metrics)
    return stem_guess, transient_blend, hpss_margin, onset_threshold, offset_threshold, frame_threshold, report


# ── MAIN CONVERSION ───────────────────────────────────────────────────────────

def convert_stem_to_midi(
    audio_file: str | None,
    stem_type: str,
    transient_blend: float,
    hpss_margin: float,
    onset_threshold: float,
    offset_threshold: float,
    frame_threshold: float,
) -> tuple[str | None, str]:
    '''
    Convert an uploaded stem audio file into a MIDI file.

    Args:
        audio_file (str | None): Local path or uploaded file reference for the stem audio.
        stem_type (str): Stem category used to describe the source audio.
        transient_blend (float): HPSS harmonic blend from 0.0 (off) to 1.0 (harmonic only).
        hpss_margin (float): HPSS separation margin; higher values isolate harmonic content more aggressively.
        onset_threshold (float): CRNN onset threshold; lower values detect more notes.
        offset_threshold (float): CRNN offset threshold for note endings.
        frame_threshold (float): CRNN frame threshold for filtering short or quiet notes.

    Returns:
        tuple[str | None, str]: MIDI file path when notes are detected, plus a human-readable status message.
    '''

    if audio_file is None:
        return None, "No audio file provided."

    # 1. Preprocess
    try:
        y = preprocess(audio_file, blend=transient_blend, margin=hpss_margin)
    except Exception as e:
        return None, f"Preprocessing failed: {e}"

    duration_sec = len(y) / PTI_SAMPLE_RATE

    # 2. Load model (cached after first call)
    try:
        transcriber = get_transcriber(
            onset_thresh=onset_threshold,
            offset_thresh=offset_threshold,
            frame_thresh=frame_threshold,
        )
    except Exception as e:
        return None, f"Model load failed: {e}"

    # 3. Transcribe — model writes MIDI directly to path
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    midi_path = out_tmp.name
    out_tmp.close()

    try:
        result = transcriber.transcribe(y, midi_path)
    except Exception as e:
        return None, f"Transcription failed: {e}"

    note_events  = result.get("est_note_events", [])
    pedal_events = result.get("est_pedal_events", [])

    if not note_events:
        return None, (
            "No notes detected. Try lowering Onset Threshold or Frame Threshold. "
            "For drums, consider a different AMT approach — this model is optimized "
            "for pitched instruments."
        )

    # 4. Build status
    pitches    = [e["midi_note"] for e in note_events]
    velocities = [e["velocity"] for e in note_events]
    status = (
        f"Done.\n"
        f"Notes: {len(note_events)} | "
        f"Pedal events: {len(pedal_events)} | "
        f"Duration: {duration_sec:.1f}s\n"
        f"Pitch range: MIDI {min(pitches)}–{max(pitches)} | "
        f"Velocity range: {min(velocities)}–{max(velocities)}\n"
        f"Stem: {stem_type} | "
        f"Transient blend: {transient_blend:.2f} | "
        f"Onset thresh: {onset_threshold}"
    )

    return midi_path, status


# ── GRADIO UI ─────────────────────────────────────────────────────────────────

def apply_preset(stem: str) -> tuple[float, float]:
    '''
    Return recommended transient-cleaning settings for a stem type.

    Args:
        stem (str): Stem category such as melodic / piano, bass, vocal, or drums / percussive.

    Returns:
        tuple[float, float]: Suppression blend and HPSS margin values for the selected stem.
    '''
    return (
        TRANSIENT_BLEND_MAP[stem],
        HPSS_MARGIN_MAP[stem],
    )


def build_ui():
    with gr.Blocks(
        theme=gr.themes.Monochrome(),
        css=".gradio-container { max-width: 960px !important; } footer { display: none !important; }",
    ) as demo:
        gr.Markdown(
            '''
            <div style="text-align: center;">
                <h1 style="font-size: 2.5em;">🎵 Stem-to-MIDI Converter 🎵</h1>
                <p style="font-size: 1.2em;">
                    Convert your audio stems into MIDI files with this powerful tool.
                    <br>
                    Powered by the Piano Transcription model by Kong et al. (2020).
                </p>
            </div>
            '''
        )

        with gr.Tabs():
            with gr.TabItem("MIDI Conversion"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            label="Upload Your Audio Stem",
                            type="filepath",
                            elem_id="audio_input",
                        )
                        with gr.Row():
                            analyze_btn = gr.Button("Analyze & Recommend Settings", variant="secondary")
                            convert_btn = gr.Button("Convert to MIDI", variant="primary")
                    with gr.Column(scale=1):
                        midi_output = gr.File(label="MIDI Output (.mid)")
                        status_output = gr.Textbox(
                            label="Status",
                            lines=6,
                            interactive=False,
                        )

            with gr.TabItem("Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Stem Configuration")
                        stem_type = gr.Dropdown(
                            label="Stem Type",
                            choices=STEM_TYPES,
                            value="melodic / piano",
                        )
                        apply_btn = gr.Button("Apply Stem Preset", variant="secondary")

                        gr.Markdown("### Transient Cleaning (HPSS)")
                        with gr.Accordion("Advanced Transient Cleaning Settings", open=False):
                            transient_blend = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                                label="Suppression Blend (0=off, 1=harmonic only)",
                            )
                            hpss_margin = gr.Slider(
                                minimum=1.0, maximum=6.0, value=2.0, step=0.1,
                                label="HPSS Margin (separation aggressiveness)",
                            )

                    with gr.Column():
                        gr.Markdown("### AMT Thresholds")
                        with gr.Accordion("Advanced AMT Threshold Settings", open=False):
                            onset_thresh = gr.Slider(
                                minimum=0.1, maximum=0.9, value=DEFAULT_ONSET_THRESH, step=0.01,
                                label="Onset Threshold",
                            )
                            offset_thresh = gr.Slider(
                                minimum=0.1, maximum=0.9, value=DEFAULT_OFFSET_THRESH, step=0.01,
                                label="Offset Threshold",
                            )
                            frame_thresh = gr.Slider(
                                minimum=0.05, maximum=0.5, value=DEFAULT_FRAME_THRESH, step=0.01,
                                label="Frame Threshold (suppresses short/quiet notes)",
                            )

                        recommendation_output = gr.Textbox(
                            label="Setting Recommendations",
                            lines=10,
                            interactive=False,
                        )

            with gr.TabItem("Music Tool Hub (MCP)"):
                gr.Markdown(
                    "Use this hub to choose the right connected music tool: split a full "
                    "song into stems first, then analyze and convert selected stems to MIDI."
                )
                hub_goal = gr.Dropdown(
                    label="What do you want to do?",
                    choices=TOOL_HUB_GOALS,
                    value="full song to MIDI",
                )
                hub_btn = gr.Button("Show Recommended Tool Workflow", variant="secondary")
                hub_catalog = gr.JSON(label="Available MCP Tools")
                hub_workflow = gr.Textbox(
                    label="Recommended Workflow",
                    lines=5,
                    interactive=False,
                )

        # Event handlers (wiring)
        hub_btn.click(
            fn=list_music_tool_hub,
            inputs=[hub_goal],
            outputs=[hub_catalog, hub_workflow],
            api_name="list_music_tool_hub",
        )

        apply_btn.click(
            fn=apply_preset,
            inputs=[stem_type],
            outputs=[transient_blend, hpss_margin],
            api_name="apply_stem_preset",
        )

        analyze_btn.click(
            fn=recommend_midi_settings,
            inputs=[audio_input, stem_type], # Changed from stem_hint
            outputs=[
                stem_type,
                transient_blend,
                hpss_margin,
                onset_thresh,
                offset_thresh,
                frame_thresh,
                recommendation_output,
            ],
            api_name="recommend_midi_settings",
        )

        convert_btn.click(
            fn=convert_stem_to_midi,
            inputs=[
                audio_input,
                stem_type,
                transient_blend,
                hpss_margin,
                onset_thresh,
                offset_thresh,
                frame_thresh,
            ],
            outputs=[midi_output, status_output],
            api_name="convert_stem_to_midi",
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(mcp_server=True)
