# Master Stem Splitter MCP connector notes

Target Space: https://huggingface.co/spaces/reallyrogueradio/Master-Stem-Splitter

## Role in the music tool hub

Master Stem Splitter is the first step for full songs: split the mix into stems, retrieve the stem zip, then send the melodic, bass, piano, or vocal stem to the StemToMIDI connector for settings analysis and MIDI conversion.

## ChatGPT connector information

Use these values when adding the Master Stem Splitter Space as a custom MCP connector in ChatGPT:

- **Connector name:** Master Stem Splitter
- **MCP server URL:** `https://reallyrogueradio-master-stem-splitter.hf.space/gradio_api/mcp/sse`
- **Authentication:** None, unless the Space is made private or custom auth is added later.

## Required Space changes

The linked Space must be updated before ChatGPT can use the MCP URL above. In that Space:

1. Install Gradio's MCP extra in `requirements.txt`:

   ```txt
   gradio[mcp]
   ```

2. Launch the Gradio app with MCP enabled in `app.py`:

   ```python
   if __name__ == "__main__":
       demo.launch(mcp_server=True)
   ```

3. Give each ChatGPT-facing event a stable `api_name`, for example:

   ```python
   submit_btn.click(
       fn=start_processing,
       inputs=[audio_in, name_in],
       outputs=status_out,
       api_name="split_master_stems",
   )

   check_btn.click(
       fn=retrieve_stems,
       inputs=search_in,
       outputs=[file_out, status_download],
       api_name="retrieve_stems_zip",
   )

   drum_submit_btn.click(
       fn=start_drum_processing,
       inputs=[drum_in, drum_name_in],
       outputs=drum_status_out,
       api_name="deconstruct_drums",
   )

   drum_check_btn.click(
       fn=retrieve_drums,
       inputs=drum_search_in,
       outputs=[drum_file_out, drum_status_download],
       api_name="retrieve_drums_zip",
   )

   analyze_btn.click(
       fn=analyze_flow_and_groove,
       inputs=[vocal_analysis_in, drum_analysis_in],
       outputs=[plot_out, text_report_out],
       api_name="analyze_flow_and_groove",
   )
   ```

4. Add the MCP tag to the Space README frontmatter:

   ```yaml
   tags:
   - mcp
   ```

## Suggested ChatGPT prompt

After the connector is added, upload audio in ChatGPT and ask:

> Use Master Stem Splitter to split this mix into stems. When the stem zip is ready, retrieve it. Then use StemToMIDI to analyze the melodic or bass stem with `recommend_midi_settings` and explain the best MIDI conversion settings before converting.

## Companion StemToMIDI connector

This repo's MIDI connector remains:

- **Connector name:** StemToMIDI
- **MCP server URL:** `https://ryanrealaf-midi.hf.space/gradio_api/mcp/sse`
