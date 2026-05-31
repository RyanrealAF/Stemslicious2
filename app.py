# ENTRY POINT: app.py — StemToMIDI with transient cleaning pre-pass
# Transient suppression: HPSS (Harmonic-Percussive Source Separation) via librosa
# AMT engine: basic-pitch (Spotify ICASSP 2022 model)

import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import librosa
import librosa.decompose

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
STEM_TYPES = ["melodic", "bass", "vocal", "drums"]

# Per-stem AMT thresholds
ONSET_THRESH_MAP   = {"melodic": 0.5,  "bass": 0.4,  "vocal": 0.5,  "drums": 0.35}
FRAME_THRESH_MAP   = {"melodic": 0.3,  "bass": 0.25, "vocal": 0.3,  "drums": 0.2}
MIN_NOTE_LEN_MAP   = {"melodic": 0.058,"bass": 0.1,  "vocal": 0.058,"drums": 0.02}
MIN_FREQ_MAP       = {"melodic": 65.4, "bass": 32.7, "vocal": 130.8,"drums": None}
MAX_FREQ_MAP       = {"melodic": 2093.0,"bass":523.3,"vocal":1046.5, "drums": None}

# Per-stem transient cleaning defaults
# 0.0 = no cleaning (pass-through), 1.0 = harmonic-only (maximum suppression)
# drums intentionally low — transients ARE the content; light smoothing only
TRANSIENT_CLEAN_MAP = {
    "melodic": 0.85,
    "bass":    0.75,
    "vocal":   0.80,
    "drums":   0.15,
}

# HPSS kernel sizes — larger margin = more aggressive separation
# melodic/vocal benefit from wider harmonic kernel; drums need narrow to preserve hits
HPSS_MARGIN_MAP = {
    "melodic": 3.0,
    "bass":    2.5,
    "vocal":   3.0,
    "drums":   1.5,
}

SAMPLE_RATE = 22050
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio, collapse to mono, resample to 22050 Hz."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return y


def normalize(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Peak-normalize to ceiling. No-op if signal is silent."""
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * ceiling
    return y


def suppress_transients(
    y: np.ndarray,
    blend: float,
    hpss_margin: float,
) -> np.ndarray:
    """
    Isolate harmonic content via HPSS and blend it back with the original.

    Parameters
    ----------
    y           : mono audio signal at SAMPLE_RATE
    blend       : 0.0 = original unchanged, 1.0 = harmonic component only
    hpss_margin : HPSS separation aggressiveness (higher = cleaner but more artifact risk)

    Returns
    -------
    Blended signal: (1 - blend) * original + blend * harmonic
    """
    if blend <= 0.0:
        return y  # short-circuit — no processing needed

    # STFT → HPSS in magnitude spectrogram domain
    D = librosa.stft(y)
    D_harmonic, D_percussive = librosa.decompose.hpss(
        np.abs(D),
        margin=hpss_margin,
    )

    # Reconstruct harmonic signal via Wiener soft-masking
    # Mask = H / (H + P + epsilon) — soft, avoids hard binary artifacts
    eps = 1e-8
    harmonic_mask = D_harmonic / (D_harmonic + D_percussive + eps)
    D_masked = D * harmonic_mask          # complex STFT × soft mask
    y_harmonic = librosa.istft(D_masked, length=len(y))

    # Align lengths (STFT round-trip can add/trim samples)
    min_len = min(len(y), len(y_harmonic))
    y_out = (1.0 - blend) * y[:min_len] + blend * y_harmonic[:min_len]

    return y_out


def preprocess(
    audio_path: str,
    blend: float,
    hpss_margin: float,
) -> np.ndarray:
    """Full preprocessing chain: load → clean transients → normalize."""
    y = load_audio(audio_path)
    y = suppress_transients(y, blend=blend, hpss_margin=hpss_margin)
    y = normalize(y)
    return y


def write_temp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return tmp.name


def convert_stem_to_midi(
    audio_file,
    stem_type: str,
    transient_blend: float,
    hpss_margin: float,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: float,
    min_freq: float,
    max_freq: float,
    melodic_midi_only: bool,
) -> tuple:
    if audio_file is None:
        return None, "No audio file provided."

    # ── 1. Preprocess with transient suppression
    try:
        y = preprocess(audio_file, blend=transient_blend, hpss_margin=hpss_margin)
    except Exception as e:
        return None, f"Preprocessing failed: {e}"

    tmp_wav = write_temp_wav(y, SAMPLE_RATE)

    freq_min = min_freq if min_freq > 0 else None
    freq_max = max_freq if max_freq > 0 else None

    # ── 2. AMT inference
    try:
        model_output, midi_data, note_events = predict(
            tmp_wav,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length,
            minimum_frequency=freq_min,
            maximum_frequency=freq_max,
            melodia_trick=melodic_midi_only,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
    except Exception as e:
        os.unlink(tmp_wav)
        return None, f"basic-pitch inference failed: {e}"

    os.unlink(tmp_wav)

    if not note_events or len(note_events) == 0:
        return None, "No notes detected. Try lowering onset/frame thresholds or reducing transient blend."

    # ── 3. Write MIDI
    out_midi = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    midi_data.write(out_midi.name)

    duration_sec = librosa.get_duration(y=y, sr=SAMPLE_RATE)
    status = (
        f"Done. {len(note_events)} notes over {duration_sec:.1f}s. "
        f"Stem: {stem_type} | Transient suppression: {transient_blend:.2f} | "
        f"HPSS margin: {hpss_margin:.1f}"
    )

    return out_midi.name, status


# ── GRADIO UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="StemToMIDI",
        theme=gr.themes.Monochrome(),
        css="footer { display: none !important; }",
    ) as demo:

        gr.Markdown("## StemToMIDI")
        gr.Markdown(
            "Converts stemmed audio to MIDI. "
            "Runs HPSS transient suppression before AMT to improve pitch detection accuracy."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Stem Audio",
                    type="filepath",
                    sources=["upload"],
                )
                stem_type = gr.Dropdown(
                    label="Stem Type",
                    choices=STEM_TYPES,
                    value="melodic",
                )
                melodic_midi_only = gr.Checkbox(
                    label="Melodia trick (harmonic filtering — disable for drums/bass)",
                    value=True,
                )

                with gr.Accordion("Transient Cleaning", open=True):
                    gr.Markdown(
                        "_Suppresses percussive content before pitch detection. "
                        "Higher blend = cleaner pitch signal but removes attack character. "
                        "Keep low for drums._"
                    )
                    transient_blend = gr.Slider(
                        0.0, 1.0, value=0.85, step=0.01,
                        label="Suppression Blend (0 = off, 1 = harmonic only)"
                    )
                    hpss_margin = gr.Slider(
                        1.0, 6.0, value=3.0, step=0.1,
                        label="HPSS Margin (separation aggressiveness)"
                    )

                with gr.Accordion("AMT Thresholds", open=False):
                    onset_thresh = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.01,
                        label="Onset Threshold"
                    )
                    frame_thresh = gr.Slider(
                        0.1, 0.9, value=0.3, step=0.01,
                        label="Frame Threshold"
                    )
                    min_note_len = gr.Slider(
                        0.01, 1.0, value=0.058, step=0.01,
                        label="Min Note Length (seconds)"
                    )
                    min_freq_input = gr.Number(
                        label="Min Frequency Hz (0 = no limit)", value=65.4
                    )
                    max_freq_input = gr.Number(
                        label="Max Frequency Hz (0 = no limit)", value=2093.0
                    )

                apply_preset_btn = gr.Button("Apply Stem Preset", variant="secondary")
                convert_btn = gr.Button("Convert to MIDI", variant="primary")

            with gr.Column(scale=1):
                midi_output = gr.File(label="MIDI Output (.mid)")
                status_output = gr.Textbox(label="Status", lines=4, interactive=False)

        def apply_preset(stem):
            return (
                TRANSIENT_CLEAN_MAP[stem],
                HPSS_MARGIN_MAP[stem],
                ONSET_THRESH_MAP[stem],
                FRAME_THRESH_MAP[stem],
                MIN_NOTE_LEN_MAP[stem],
                MIN_FREQ_MAP[stem] or 0,
                MAX_FREQ_MAP[stem] or 0,
                stem not in ("drums", "bass"),
            )

        apply_preset_btn.click(
            fn=apply_preset,
            inputs=[stem_type],
            outputs=[
                transient_blend, hpss_margin,
                onset_thresh, frame_thresh, min_note_len,
                min_freq_input, max_freq_input,
                melodic_midi_only,
            ],
        )

        convert_btn.click(
            fn=convert_stem_to_midi,
            inputs=[
                audio_input, stem_type,
                transient_blend, hpss_margin,
                onset_thresh, frame_thresh, min_note_len,
                min_freq_input, max_freq_input,
                melodic_midi_only,
            ],
            outputs=[midi_output, status_output],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()