---
title: StemToMIDI
emoji: 🎵
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# StemToMIDI

Converts stemmed audio files (drums, bass, melodic, vocal) to MIDI using Spotify's `basic-pitch` neural AMT model.

## Usage
1. Upload a stem (WAV, MP3, FLAC)
2. Select the stem type
3. Click **Apply Stem Preset** to auto-load tuned thresholds
4. Click **Convert to MIDI**
5. Download the `.mid` file

## Notes
- Best results with clean, isolated stems (from Demucs or similar)
- Drums output maps hits to pitched MIDI — import into a DAW and remap to GM drum map
- Melodia trick filters out non-melodic content — disable it for drums/bass
- Lower thresholds = more notes detected (higher false positive risk)