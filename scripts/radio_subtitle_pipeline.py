from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_RADIO_URL = "https://tingfm.com/radio/16543?lang=zh_CN"
DEFAULT_ASR_MODEL = "panlr/whisper-finetune-teochew"


@dataclass
class StreamInfo:
    title: str
    streams: list[dict[str, Any]]


@dataclass
class WhisperRuntime:
    processor: Any
    model: Any
    device: str
    dtype: Any


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def request_text(url: str, headers: dict[str, str] | None = None) -> str:
    response = requests.get(
        url,
        headers=headers or {},
        timeout=30,
        allow_redirects=True,
        verify=True,
    )
    response.raise_for_status()
    return response.text


def extract_tingfm_runtime(page_html: str) -> dict[str, Any]:
    match = re.search(r"<script>let wndt = (\{.*?\});</script>", page_html, re.S)
    if not match:
        raise RuntimeError("Could not find TingFM runtime config (wndt) in page.")
    return json.loads(match.group(1))


def extract_post_id(url: str, page_html: str) -> int:
    url_match = re.search(r"/radio/(\d+)", url)
    if url_match:
        return int(url_match.group(1))

    page_match = re.search(r'"post_id"\s*:\s*(\d+)', page_html)
    if page_match:
        return int(page_match.group(1))

    raise RuntimeError("Could not infer TingFM post_id.")


def resolve_tingfm_streams(radio_url: str) -> StreamInfo:
    page_html = request_text(radio_url)
    runtime = extract_tingfm_runtime(page_html)
    post_id = extract_post_id(radio_url, page_html)
    token_key = runtime["token_key"]
    stream_token = runtime[token_key]
    api_root = runtime.get("api", "https://api.tingfm.com/wp-json/")
    api_url = f"{api_root.rstrip('/')}/query/wndt_streams"
    response = requests.get(
        api_url,
        params={"post_id": post_id, "in_web": "true"},
        headers={"Stream-Token": stream_token},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") <= 0:
        raise RuntimeError(f"TingFM stream API failed: {payload.get('msg')}")
    data = payload["data"]
    return StreamInfo(title=data["title"], streams=data["streams"])


def choose_stream(streams: list[dict[str, Any]], prefer: str = "mp3") -> dict[str, Any]:
    for stream in streams:
        if stream.get("type") == prefer:
            return stream
    if not streams:
        raise RuntimeError("No stream URL returned.")
    return streams[0]


def record_stream(stream_url: str, output_path: Path, seconds: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Recording {seconds}s stream to {output_path}")
    with requests.get(stream_url, stream=True, timeout=30) as response:
        response.raise_for_status()
        start = time.monotonic()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    handle.write(chunk)
                if time.monotonic() - start >= seconds:
                    break


def ffmpeg_executable() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is not on PATH and imageio-ffmpeg is not installed. "
            "Install requirements-radio-subtitles.txt first."
        ) from exc


def to_wav(input_path: Path, output_path: Path, max_seconds: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(input_path),
    ]
    if max_seconds:
        command.extend(["-t", str(max_seconds)])
    command.extend(["-ac", "1", "-ar", "16000", str(output_path)])
    log(f"Converting audio to 16 kHz mono WAV: {output_path}")
    subprocess.run(command, check=True)


def load_audio_mono(wav_path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    return audio, sample_rate


def detect_speech_segments(
    wav_path: Path,
    top_db: int = 30,
    min_speech_seconds: float = 0.7,
    merge_gap_seconds: float = 0.4,
    padding_seconds: float = 0.15,
) -> list[dict[str, float]]:
    import librosa

    audio, sample_rate = load_audio_mono(wav_path)
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return []

    total_seconds = len(audio) / sample_rate
    padded: list[dict[str, float]] = []
    for start_sample, end_sample in intervals:
        start = max(0.0, start_sample / sample_rate - padding_seconds)
        end = min(total_seconds, end_sample / sample_rate + padding_seconds)
        if end - start >= min_speech_seconds:
            padded.append({"start": start, "end": end})

    if not padded:
        return []

    merged: list[dict[str, float]] = [padded[0]]
    for segment in padded[1:]:
        previous = merged[-1]
        if segment["start"] - previous["end"] <= merge_gap_seconds:
            previous["end"] = max(previous["end"], segment["end"])
        else:
            merged.append(segment)

    return [segment for segment in merged if segment["end"] - segment["start"] >= min_speech_seconds]


def iter_windows(
    total_seconds: float,
    chunk_seconds: float,
    speech_segments: list[dict[str, float]] | None = None,
) -> list[tuple[float, float]]:
    source_segments = speech_segments or [{"start": 0.0, "end": total_seconds}]
    windows: list[tuple[float, float]] = []
    for segment in source_segments:
        cursor = segment["start"]
        end = min(segment["end"], total_seconds)
        while cursor < end:
            window_end = min(cursor + chunk_seconds, end)
            if window_end - cursor >= 0.5:
                windows.append((cursor, window_end))
            cursor = window_end
    return windows


def load_whisper_runtime(model_id: str) -> WhisperRuntime:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    log(f"Loading ASR model {model_id} on {'cuda' if use_cuda else 'cpu'}")
    processor = WhisperProcessor.from_pretrained(model_id)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="chinese",
        task="transcribe",
    )
    model.to(device)
    model.eval()
    return WhisperRuntime(processor=processor, model=model, device=device, dtype=dtype)


def generate_whisper_text(
    runtime: WhisperRuntime,
    audio: Any,
    sample_rate: int,
) -> str:
    features = runtime.processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_features
    features = features.to(device=runtime.device, dtype=runtime.dtype)
    predicted_ids = runtime.model.generate(features, max_new_tokens=225)
    return runtime.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def transcribe_whisper(
    wav_path: Path,
    model_id: str,
    return_timestamps: bool = True,
    speech_segments: list[dict[str, float]] | None = None,
    chunk_seconds: float = 25.0,
) -> dict[str, Any]:
    import torch

    audio, sample_rate = load_audio_mono(wav_path)
    runtime = load_whisper_runtime(model_id)
    log(f"Transcribing {wav_path}")

    total_seconds = len(audio) / sample_rate
    windows = iter_windows(total_seconds, chunk_seconds, speech_segments=speech_segments)
    segments: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start, end in windows:
            chunk = audio[int(start * sample_rate) : int(end * sample_rate)]
            if len(chunk) < sample_rate * 0.5:
                continue
            text = generate_whisper_text(runtime, chunk, sample_rate)
            if text:
                segments.append({"timestamp": (start, end), "text": text})

    return {
        "text": "".join(segment["text"] for segment in segments),
        "chunks": segments if return_timestamps else [],
    }


def seconds_to_srt_time(value: float) -> str:
    millis = int(round(value * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def result_to_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = result.get("chunks") or []
    segments: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        timestamp = chunk.get("timestamp") or (None, None)
        start = timestamp[0] if timestamp[0] is not None else idx * 30.0
        end = timestamp[1] if timestamp[1] is not None else start + 30.0
        text = (chunk.get("text") or "").strip()
        if text:
            segments.append({"start": float(start), "end": float(end), "text": text})
    if not segments and result.get("text"):
        segments.append({"start": 0.0, "end": 0.0, "text": result["text"].strip()})
    return segments


def write_srt(segments: list[dict[str, Any]], output_path: Path, text_key: str = "text") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = seconds_to_srt_time(segment["start"])
        end = seconds_to_srt_time(max(segment["end"], segment["start"] + 0.5))
        text = segment.get(text_key) or segment.get("text") or ""
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def translate_openai_compatible(
    segments: list[dict[str, Any]],
    api_base: str,
    model: str,
    api_key: str,
) -> list[dict[str, Any]]:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    translated: list[dict[str, Any]] = []
    system = (
        "你是潮汕话字幕翻译器。把输入的潮汕话正字、谐音字或夹杂普通话文本"
        "翻译成自然、简洁的现代标准汉语。只输出译文，不解释。"
    )
    for segment in segments:
        source = segment["text"]
        if not source:
            translated.append({**segment, "translation": ""})
            continue
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": source},
                ],
                "temperature": 0.1,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        target = payload["choices"][0]["message"]["content"].strip()
        translated.append({**segment, "translation": target})
    return translated


def add_common_asr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=25.0)
    parser.add_argument("--vad", action="store_true", help="Enable simple energy-based speech segmentation.")
    parser.add_argument("--vad-top-db", type=int, default=30)
    parser.add_argument("--min-speech-seconds", type=float, default=0.7)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.4)
    parser.add_argument("--speech-padding-seconds", type=float, default=0.15)
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--translate-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--translate-model", default="gpt-4.1-mini")
    parser.add_argument("--translate-key-env", default="OPENAI_API_KEY")


def write_asr_outputs(wav_audio: Path, out_dir: Path, args: argparse.Namespace) -> None:
    speech_segments = None
    if args.vad:
        speech_segments = detect_speech_segments(
            wav_audio,
            top_db=args.vad_top_db,
            min_speech_seconds=args.min_speech_seconds,
            merge_gap_seconds=args.merge_gap_seconds,
            padding_seconds=args.speech_padding_seconds,
        )
        (out_dir / "vad_segments.json").write_text(
            json.dumps(speech_segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not speech_segments:
            log("VAD found no speech-like segments; no ASR output will be generated.")
            write_srt([], out_dir / "teochew.srt")
            (out_dir / "segments.json").write_text("[]", encoding="utf-8")
            return
        log(f"VAD kept {len(speech_segments)} speech-like segment(s).")

    if args.no_asr:
        log("Skipping ASR because --no-asr was set.")
        return

    result = transcribe_whisper(
        wav_audio,
        args.asr_model,
        return_timestamps=not args.no_timestamps,
        speech_segments=speech_segments,
        chunk_seconds=args.chunk_seconds,
    )
    (out_dir / "asr_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    segments = result_to_segments(result)
    (out_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(segments, out_dir / "teochew.srt")

    if args.translate:
        api_key = os.environ.get(args.translate_key_env, "")
        if not api_key:
            raise RuntimeError(f"Set {args.translate_key_env} before using --translate.")
        translated = translate_openai_compatible(
            segments,
            api_base=args.translate_api_base,
            model=args.translate_model,
            api_key=api_key,
        )
        (out_dir / "segments.translated.json").write_text(
            json.dumps(translated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_srt(translated, out_dir / "mandarin.srt", text_key="translation")


def cmd_resolve(args: argparse.Namespace) -> None:
    stream_info = resolve_tingfm_streams(args.radio_url)
    print(json.dumps({"title": stream_info.title, "streams": stream_info.streams}, ensure_ascii=False, indent=2))


def cmd_record(args: argparse.Namespace) -> None:
    stream_info = resolve_tingfm_streams(args.radio_url)
    stream = choose_stream(stream_info.streams, args.prefer_stream)
    record_stream(stream["url"], Path(args.output), args.seconds)


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_info = resolve_tingfm_streams(args.radio_url)
    stream = choose_stream(stream_info.streams, args.prefer_stream)
    metadata = {"title": stream_info.title, "stream": stream}
    (out_dir / "stream.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_audio = out_dir / "radio_sample.mp3"
    wav_audio = out_dir / "radio_sample.wav"
    record_stream(stream["url"], raw_audio, args.seconds)
    to_wav(raw_audio, wav_audio, max_seconds=args.seconds)

    write_asr_outputs(wav_audio, out_dir, args)


def cmd_file(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_path}")

    metadata = {"input": str(input_path.resolve())}
    (out_dir / "input.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    wav_audio = out_dir / "input.wav"
    to_wav(input_path, wav_audio, max_seconds=args.max_seconds)
    write_asr_outputs(wav_audio, out_dir, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TingFM radio to Teochew subtitle prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Resolve TingFM stream URLs.")
    resolve.add_argument("--radio-url", default=DEFAULT_RADIO_URL)
    resolve.set_defaults(func=cmd_resolve)

    record = subparsers.add_parser("record", help="Record a short MP3 sample from TingFM.")
    record.add_argument("--radio-url", default=DEFAULT_RADIO_URL)
    record.add_argument("--prefer-stream", default="mp3", choices=["mp3", "m3u8"])
    record.add_argument("--seconds", type=int, default=30)
    record.add_argument("--output", default="outputs/radio_sample.mp3")
    record.set_defaults(func=cmd_record)

    run = subparsers.add_parser("run", help="Record, convert, transcribe, and optionally translate.")
    run.add_argument("--radio-url", default=DEFAULT_RADIO_URL)
    run.add_argument("--prefer-stream", default="mp3", choices=["mp3", "m3u8"])
    run.add_argument("--seconds", type=int, default=30)
    run.add_argument("--output-dir", default="outputs/radio_16543")
    add_common_asr_args(run)
    run.set_defaults(func=cmd_run)

    file_parser = subparsers.add_parser("file", help="Transcribe a local audio/video file.")
    file_parser.add_argument("--input", required=True)
    file_parser.add_argument("--output-dir", default="outputs/local_file")
    file_parser.add_argument("--max-seconds", type=int, default=None)
    add_common_asr_args(file_parser)
    file_parser.set_defaults(func=cmd_file)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
