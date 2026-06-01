---
title: MusicToolHub
emoji: 🎵
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.15.2
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
tags:
- mcp
---


## Music tool hub workflow

This Space is the MIDI side of a larger ChatGPT music-production hub. Use `list_music_tool_hub` first when you are not sure which tool should run next. The intended full-song workflow is:

1. **6-stem separation:** use the companion Master Stem Splitter connector to split a full mix and retrieve a stem zip.
2. **Stem selection:** choose the melodic, bass, piano, or vocal stem that should become MIDI.
3. **Settings recommendation:** call `recommend_midi_settings` on the chosen stem to get HPSS and AMT thresholds.
4. **MIDI conversion:** call `convert_stem_to_midi` with the recommended settings.

## MCP server

This Space also runs as a Gradio MCP tool hub. When deployed to Hugging Face Spaces, MCP-compatible clients can call the MIDI conversion and hub-planning tools at:

```
https://ryanrealaf-midi.hf.space/gradio_api/mcp/sse
```

The public Gradio UI remains available at the Space URL, while the MCP endpoint exposes `list_music_tool_hub`, `recommend_midi_settings`, `convert_stem_to_midi`, and `apply_stem_preset` as tools. ChatGPT can call `recommend_midi_settings` with an uploaded stem to inspect the audio and suggest transient cleaning and AMT threshold values before running conversion.

### Connect this Space to ChatGPT

Use this information when adding the Space as a custom MCP connector/app in ChatGPT:

- **Connector name:** StemToMIDI
- **MCP server URL:** `https://ryanrealaf-midi.hf.space/gradio_api/mcp/sse`
- **Authentication:** None, unless you later make the Space private or add auth
- **Useful tools:**
  - `list_music_tool_hub` — choose the right workflow across stem separation, MIDI conversion, drums, and analysis
  - `recommend_midi_settings` — upload an audio stem and get suggested settings
  - `convert_stem_to_midi` — run the MIDI conversion with selected settings
  - `apply_stem_preset` — apply the built-in preset for a known stem type

In ChatGPT, enable developer mode or custom MCP apps/connectors, create a new custom app/connector, paste the MCP server URL above, and save/connect it. Then start a chat and ask something like:

> Use StemToMIDI as my music tool hub. First call `list_music_tool_hub` for a full song to MIDI workflow. If I provide a full mix, use Master Stem Splitter first; if I provide an isolated stem, call `recommend_midi_settings`, explain the recommended settings, then ask whether to run `convert_stem_to_midi`.

If your ChatGPT workspace requires admin approval for custom MCP apps, ask an admin to enable or publish the connector for your workspace.

### Companion Master Stem Splitter Space

If you also want ChatGPT to split a full mix before converting stems to MIDI, add the Master Stem Splitter Space as a separate MCP connector after enabling MCP in that Space. Use:

- **Connector name:** Master Stem Splitter
- **MCP server URL:** `https://reallyrogueradio-master-stem-splitter.hf.space/gradio_api/mcp/sse`

See `MASTER_STEM_SPLITTER_MCP.md` for the required Space edits and a combined splitter-to-MIDI prompt.

## Automatic Hugging Face deployment

This repository includes a GitHub Actions workflow that syncs each pushed commit on `main`, `master`, or `work` to the Hugging Face Space.

To enable it, add these repository settings in GitHub:

- **Secret:** `HF_TOKEN` — a Hugging Face access token with write access to the Space.
- **Variable:** `HF_SPACE` — optional; defaults to `Ryanrealaf/Midi` when unset.

The workflow can also be started manually from the **Actions** tab with `workflow_dispatch`.
