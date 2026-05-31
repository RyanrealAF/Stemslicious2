# ENTRY POINT: app.py — StemToMIDI
# AMT stack: crepe (neural pitch, monophonic) + aubio (onset + polyphonic tracking)
# Replaces basic-pitch which is incompatible with Python 3.11+ due to TF pin
# Transient suppression: HPSS via librosa (harmonic soft-mask)

import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import librosa
import librosa.decompose
import pretty_midi
import crepe
import aubio

# ── CONFIG ───────────────────────────────────────────────────────────────────
STEM_TYPES = ["melodic", "bass", "vocal", "drums"]

# Onset detection sensitivity per stem
ONSET_THRESH_MAP = {
    "melodic": 0.3,
    "bass":    0.25,
    "vocal":   0.3,
    "drums":   0.15,   # hair-trigger — drums are all onset
}

# crepe confidence floor — notes below this are discarded
PITCH_CONF_MAP = {
    "melodic": 0.5,
    "bass":    0.45,
    "vocal":   0.55,
    "drums":   None,   # drums don't use crepe pitch
}

MIN_NOTE_DUR_MAP = {   # seconds
    "melodic": 0.06,
    "bass":    0.10,
    "vocal":   0.06,
    "drums":   0.02,
}

MIN_FREQ_MAP = {
    "melodic": 65.4,
    "bass":    32.7,
    "vocal":   130.8,
    "drums":   None,
}

MAX_FREQ_MAP = {
    "melodic": 2093.0,
    "bass":    523.3,
    "vocal":   1046.5,
    "drums":   None,
}

# HPSS transient suppression defaults
TRANSIENT_BLEND_MAP = {
    "melodic": 0.85,
    "bass":    0.75,
    "vocal":   0.80,
    "drums":   0.15,
}

HPSS_MARGIN_MAP = {
    "melodic": 3.0,
    "bass":    2.5,
    "vocal":   3.0,
    "drums":   1.5,
}

SAMPLE_RATE    = 22050
CREPE_SR       = 16000   # crepe requires 16kHz input
MIDI_TEMPO     = 120
MIDI_PROGRAM   = 0       # acoustic grand piano
DRUM_MIDI_NOTE = 38      # snare — remappable in DAW
# ─────────────────────────────────────────────────────────────────────────────


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def normalize(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * ceiling
    return y


def suppress_transients(y: np.ndarray, blend: float, margin: float) -> np.ndarray:
    """
    HPSS soft-mask: attenuates percussive (vertical) spectrogram structures.
    blend=0 → pass-through; blend=1 → harmonic component only.
    Uses Wiener soft mask to avoid binary separation artifacts.
    """
    if blend <= 0.0:
        return y

    D = librosa.stft(y)
    H_mag, P_mag = librosa.decompose.hpss(np.abs(D), margin=margin)

    eps = 1e-8
    soft_mask = H_mag / (H_mag + P_mag + eps)
    y_harmonic = librosa.istft(D * soft_mask, length=len(y))

    min_len = min(len(y), len(y_harmonic))
    return (1.0 - blend) * y[:min_len] + blend * y_harmonic[:min_len]


def preprocess(path: str, blend: float, margin: float) -> np.ndarray:
    y = load_audio(path)
    y = suppress_transients(y, blend=blend, margin=margin)
    y = normalize(y)
    return y


# ── ONSET DETECTION ───────────────────────────────────────────────────────────

def detect_onsets(y: np.ndarray, threshold: float) -> np.ndarray:
    """
    aubio onset detector — returns onset times in seconds.
    Uses 'specflux' method: spectral flux, robust across stem types.
    """
    hop_size   = 512
    win_size   = 1024
    detector   = aubio.onset("specflux", win_size, hop_size, SAMPLE_RATE)
    detector.set_threshold(threshold)
    detector.set_silence(-60)

    y_float32 = y.astype(np.float32)
    onsets    = []
    pos       = 0

    while pos + hop_size <= len(y_float32):
        frame = y_float32[pos : pos + hop_size]
        if detector(frame):
            onsets.append(detector.get_last_s())
        pos += hop_size

    return np.array(onsets)


# ── PITCH EXTRACTION (melodic / bass / vocal) ─────────────────────────────────

def extract_pitch_crepe(
    y: np.ndarray,
    onsets: np.ndarray,
    conf_threshold: float,
    min_note_dur: float,
    min_freq: float | None,
    max_freq: float | None,
) -> list[tuple]:
    """
    Run crepe at 16kHz. For each onset window, take the median confident pitch.
    Returns list of (start_sec, end_sec, midi_note, velocity).
    """
    if len(onsets) == 0:
        return []

    # Resample to crepe's required rate
    y_16k = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=CREPE_SR)

    # Run full-signal crepe inference (step_size=10ms)
    times, freqs, confidences, _ = crepe.predict(
        y_16k,
        CREPE_SR,
        model_capacity="medium",
        step_size=10,
        viterbi=True,
        verbose=0,
    )

    notes = []
    total_dur = len(y) / SAMPLE_RATE

    for i, onset in enumerate(onsets):
        offset = onsets[i + 1] if i + 1 < len(onsets) else min(onset + 0.5, total_dur)

        # Slice crepe output to this note window
        mask = (times >= onset) & (times < offset)
        if not np.any(mask):
            continue

        w_freqs = freqs[mask]
        w_confs = confidences[mask]

        # Filter by confidence
        confident = w_confs >= conf_threshold
        if not np.any(confident):
            continue

        median_freq = np.median(w_freqs[confident])

        # Frequency bounds
        if min_freq and median_freq < min_freq:
            continue
        if max_freq and median_freq > max_freq:
            continue

        midi_note = int(round(librosa.hz_to_midi(median_freq)))
        midi_note = max(0, min(127, midi_note))

        dur = offset - onset
        if dur < min_note_dur:
            continue

        velocity = int(np.clip(np.mean(w_confs[confident]) * 100 + 27, 30, 127))
        notes.append((onset, offset, midi_note, velocity))

    return notes


# ── DRUM TRANSCRIPTION ────────────────────────────────────────────────────────

def extract_drum_onsets(
    y: np.ndarray,
    onsets: np.ndarray,
    min_note_dur: float,
) -> list[tuple]:
    """
    Drums: onset time → fixed MIDI note (GM snare placeholder).
    Velocity derived from RMS energy in a short window around each onset.
    All hits are 30ms duration — remapped in DAW.
    """
    notes   = []
    hop     = int(0.03 * SAMPLE_RATE)  # 30ms window for RMS

    for onset_sec in onsets:
        start_samp = int(onset_sec * SAMPLE_RATE)
        end_samp   = min(start_samp + hop, len(y))
        rms        = np.sqrt(np.mean(y[start_samp:end_samp] ** 2))
        velocity   = int(np.clip(rms * 800, 30, 127))
        end_sec    = onset_sec + max(0.03, min_note_dur)
        notes.append((onset_sec, end_sec, DRUM_MIDI_NOTE, velocity))

    return notes


# ── MIDI ASSEMBLY ─────────────────────────────────────────────────────────────

def notes_to_midi(
    notes: list[tuple],
    is_drums: bool,
) -> pretty_midi.PrettyMIDI:
    pm        = pretty_midi.PrettyMIDI(initial_tempo=MIDI_TEMPO)
    program   = 0 if not is_drums else 0
    instrument = pretty_midi.Instrument(
        program=program,
        is_drum=is_drums,
        name="Drums" if is_drums else "Stem",
    )

    for (start, end, pitch, velocity) in notes:
        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start),
            end=float(max(end, start + 0.02)),
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


# ── MAIN CONVERSION ───────────────────────────────────────────────────────────

def convert_stem_to_midi(
    audio_file,
    stem_type: str,
    transient_blend: float,
    hpss_margin: float,
    onset_threshold: float,
    pitch_conf: float,
    min_note_dur: float,
    min_freq: float,
    max_freq: float,
) -> tuple:
    if audio_file is None:
        return None, "No audio file provided."

    # 1. Preprocess
    try:
        y = preprocess(audio_file, blend=transient_blend, margin=hpss_margin)
    except Exception as e:
        return None, f"Preprocessing failed: {e}"

    freq_min = min_freq if min_freq > 0 else None
    freq_max = max_freq if max_freq > 0 else None
    is_drums = stem_type == "drums"

    # 2. Onset detection
    try:
        onsets = detect_onsets(y, threshold=onset_threshold)
    except Exception as e:
        return None, f"Onset detection failed: {e}"

    if len(onsets) == 0:
        return None, "No onsets detected. Try lowering the onset threshold."

    # 3. Pitch or drum extraction
    try:
        if is_drums:
            notes = extract_drum_onsets(y, onsets, min_note_dur)
        else:
            notes = extract_pitch_crepe(
                y, onsets, pitch_conf, min_note_dur, freq_min, freq_max
            )
    except Exception as e:
        return None, f"Pitch extraction failed: {e}"

    if not notes:
        return None, "No notes extracted. Try lowering confidence threshold or onset threshold."

    # 4. Build and write MIDI
    pm      = notes_to_midi(notes, is_drums=is_drums)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    pm.write(out_tmp.name)

    dur    = librosa.get_duration(y=y, sr=SAMPLE_RATE)
    status = (
        f"Done. {len(notes)} notes | {dur:.1f}s audio | "
        f"stem: {stem_type} | onsets detected: {len(onsets)} | "
        f"transient blend: {transient_blend:.2f}"
    )
    return out_tmp.name, status


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
            "Pipeline: HPSS transient suppression → aubio onset detection → "
            "crepe neural pitch → pretty_midi output."
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

                with gr.Accordion("Transient Cleaning", open=True):
                    transient_blend = gr.Slider(
                        0.0, 1.0, value=0.85, step=0.01,
                        label="Suppression Blend (0=off, 1=harmonic only)"
                    )
                    hpss_margin = gr.Slider(
                        1.0, 6.0, value=3.0, step=0.1,
                        label="HPSS Margin (separation aggressiveness)"
                    )

                with gr.Accordion("Detection Parameters", open=False):
                    onset_thresh = gr.Slider(
                        0.05, 0.9, value=0.3, step=0.01,
                        label="Onset Threshold (lower = more sensitive)"
                    )
                    pitch_conf = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.01,
                        label="Pitch Confidence Floor (crepe — ignored for drums)"
                    )
                    min_note_dur = gr.Slider(
                        0.01, 1.0, value=0.06, step=0.01,
                        label="Min Note Duration (seconds)"
                    )
                    min_freq_in = gr.Number(label="Min Frequency Hz (0=none)", value=65.4)
                    max_freq_in = gr.Number(label="Max Frequency Hz (0=none)", value=2093.0)

                apply_btn  = gr.Button("Apply Stem Preset", variant="secondary")
                convert_btn = gr.Button("Convert to MIDI", variant="primary")

            with gr.Column(scale=1):
                midi_output   = gr.File(label="MIDI Output (.mid)")
                status_output = gr.Textbox(label="Status", lines=4, interactive=False)

        def apply_preset(stem):
            return (
                TRANSIENT_BLEND_MAP[stem],
                HPSS_MARGIN_MAP[stem],
                ONSET_THRESH_MAP[stem],
                PITCH_CONF_MAP[stem] or 0.5,
                MIN_NOTE_DUR_MAP[stem],
                MIN_FREQ_MAP[stem] or 0,
                MAX_FREQ_MAP[stem] or 0,
            )

        apply_btn.click(
            fn=apply_preset,
            inputs=[stem_type],
            outputs=[
                transient_blend, hpss_margin,
                onset_thresh, pitch_conf, min_note_dur,
                min_freq_in, max_freq_in,
            ],
        )

        convert_btn.click(
            fn=convert_stem_to_midi,
            inputs=[
                audio_input, stem_type,
                transient_blend, hpss_margin,
                onset_thresh, pitch_conf, min_note_dur,
                min_freq_in, max_freq_in,
            ],
            outputs=[midi_output, status_output],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()