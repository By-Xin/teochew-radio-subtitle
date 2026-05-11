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

Observed limitation:

- The initial test station, `汕头电台音乐之声`, may contain music, Mandarin, or English segments. It is not a reliable Teochew ASR benchmark by itself.

## Next Tasks

1. Add local audio file input.
2. Add VAD-based speech segmentation.
3. Add no-speech and music filtering.
4. Compare `panlr/whisper-finetune-teochew` with `panlr/Qwen3_ASR_teochew`.
5. Build a small realtime loop with 5-10 second chunks.
