# ENTRY POINT: app.py — StemToMIDI (Polyphonic AMT)
# Engine: piano_transcription_inference (Kong et al. 2020)
#   - CRNN frame-level estimator: polyphonic, velocity-aware, pedal-aware
#   - F1=0.9677 on MAPS dataset
#   - Model: ~165MB, downloaded on first run to ~/piano_transcription_inference_data/
# Preprocessing: HPSS transient suppression via librosa
# Compatible: Gradio 6.x, Python 3.10, CPU inference

import os
import tempfile
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
# ─────────────────────────────────────────────────────────────────────────────

# ── MODEL SINGLETON ───────────────────────────────────────────────────────────
# Instantiated once at module load — avoids 165MB re-download per inference call
_transcriber = None

def get_transcriber(onset_thresh: float, offset_thresh: float, frame_thresh: float):
    """
    Return a PianoTranscription instance with updated thresholds.
    Model weights are cached after first load; only threshold attributes are patched.
    """
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


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load to mono float32 at PREPROCESS_SR."""
    y, _ = librosa.load(path, sr=PREPROCESS_SR, mono=True, dtype=np.float32)
    return y


def normalize(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * ceiling
    return y


def suppress_transients(y: np.ndarray, blend: float, margin: float) -> np.ndarray:
    """
    HPSS Wiener soft-mask.
    Separates harmonic (sustained pitch) from percussive (transient) content.
    blend=0 → unchanged. blend=1 → harmonic only.
    Soft mask avoids binary separation artifacts at note boundaries.
    """
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
    """Full chain: load → HPSS suppress → normalize → resample to PTI SR."""
    y = load_audio(path)
    y = suppress_transients(y, blend=blend, margin=margin)
    y = normalize(y)
    # Downsample to 16kHz for PianoTranscription model
    y = librosa.resample(y, orig_sr=PREPROCESS_SR, target_sr=PTI_SAMPLE_RATE)
    return y.astype(np.float32)


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
    """
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
    """

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
    """
    Return recommended transient-cleaning settings for a stem type.

    Args:
        stem (str): Stem category such as melodic / piano, bass, vocal, or drums / percussive.

    Returns:
        tuple[float, float]: Suppression blend and HPSS margin values for the selected stem.
    """
    return (
        TRANSIENT_BLEND_MAP[stem],
        HPSS_MARGIN_MAP[stem],
    )


def build_ui():
    with gr.Blocks(
        title="StemToMIDI",
        theme=gr.themes.Base(),
        css="footer { display: none !important; }",
    ) as demo:

        gr.Markdown("## StemToMIDI — Polyphonic AMT")
        gr.Markdown(
            "Frame-level polyphonic transcription using "
            "[Kong et al. 2020](https://arxiv.org/abs/2010.01815) CRNN model "
            "(F1=0.97 on MAPS). Handles chords, overlapping notes, velocity, and pedal. "
            "Optimized for pitched instruments. "
            "_Model (~165MB) downloads automatically on first run._"
        )

        with gr.Row():
            # ── LEFT: inputs
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Stem Audio",
                    type="filepath",
                )
                stem_type = gr.Dropdown(
                    label="Stem Type",
                    choices=STEM_TYPES,
                    value="melodic / piano",
                )

                with gr.Accordion("Transient Cleaning (HPSS)", open=True):
                    gr.Markdown(
                        "_Attenuates percussive content before transcription. "
                        "Reduces false positives from drum bleed on melodic stems. "
                        "Keep at 0 for drums/percussive stems._"
                    )
                    transient_blend = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                        label="Suppression Blend (0=off, 1=harmonic only)",
                    )
                    hpss_margin = gr.Slider(
                        minimum=1.0, maximum=6.0, value=2.0, step=0.1,
                        label="HPSS Margin (separation aggressiveness)",
                    )

                with gr.Accordion("AMT Thresholds", open=False):
                    gr.Markdown(
                        "_All three thresholds control the CRNN post-processor. "
                        "Lower = more notes detected, higher false positive risk. "
                        "Defaults (0.3 / 0.3 / 0.1) match the published model._"
                    )
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

                apply_btn   = gr.Button("Apply Stem Preset", variant="secondary")
                convert_btn = gr.Button("Convert to MIDI", variant="primary")

            # ── RIGHT: outputs
            with gr.Column(scale=1):
                midi_output   = gr.File(label="MIDI Output (.mid)")
                status_output = gr.Textbox(
                    label="Status",
                    lines=6,
                    interactive=False,
                )

        # ── Events
        apply_btn.click(
            fn=apply_preset,
            inputs=[stem_type],
            outputs=[transient_blend, hpss_margin],
            api_name="apply_stem_preset",
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
