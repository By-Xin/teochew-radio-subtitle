# Milestone 1: Radio To Subtitle Prototype

## Goal

Build a minimal, inspectable prototype for:

```text
radio stream -> short audio sample -> Teochew ASR subtitle -> optional Mandarin subtitle
```

## Current Status

Done:

- TingFM page parsing
- TingFM stream API resolution
- MP3 stream recording
- FFmpeg conversion to 16 kHz mono WAV
- Local Hugging Face Whisper inference
- JSON and SRT subtitle output
- Optional OpenAI-compatible translation hook
- Local audio/video file input
- First-pass energy-based VAD and silence filtering

Observed limitation:

- The initial test station, `汕头电台音乐之声`, may contain music, Mandarin, or English segments. It is not a reliable Teochew ASR benchmark by itself.
- The current VAD is energy-based. It removes silence reasonably, but it does not reliably distinguish speech from music.

## Next Tasks

1. Add stronger speech/music filtering, likely with Silero VAD or another speech activity model.
2. Compare `panlr/whisper-finetune-teochew` with `panlr/Qwen3_ASR_teochew`.
3. Build a small realtime loop with 5-10 second chunks.
4. Add a minimal local web UI for input, status, and subtitle preview.
