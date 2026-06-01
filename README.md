---
title: StemToMIDI
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

## MCP server

This Space also runs as a Gradio MCP server. When deployed to Hugging Face Spaces, MCP-compatible clients can call the MIDI conversion tool at:

```
https://ryanrealaf-midi.hf.space/gradio_api/mcp/sse
```

The public Gradio UI remains available at the Space URL, while the MCP endpoint exposes `convert_stem_to_midi` and `apply_stem_preset` as tools.
