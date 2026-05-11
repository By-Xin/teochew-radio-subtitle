from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from radio_subtitle_pipeline import (  # noqa: E402
    DEFAULT_ASR_MODEL,
    generate_whisper_text,
    load_audio_mono,
    load_whisper_runtime,
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str, remove_spaces: bool = True) -> str:
    text = text.strip()
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    return text


def edit_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_char != hyp_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> tuple[int, int]:
    normalized_ref = normalize_text(reference)
    normalized_hyp = normalize_text(hypothesis)
    return edit_distance(normalized_ref, normalized_hyp), max(1, len(normalized_ref))


def resolve_audio_path(record: dict[str, Any], dataset_root: Path | None) -> Path:
    if record.get("audio_path"):
        return Path(record["audio_path"])
    if dataset_root and record.get("audio"):
        return dataset_root / record["audio"].lstrip("./")
    raise ValueError(f"Record has no audio path: {record.get('id')}")


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    records = load_jsonl(manifest_path)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records found in {manifest_path}")

    log(f"Loading ASR model: {args.asr_model}")
    runtime = load_whisper_runtime(args.asr_model)
    predictions_path = output_dir / "predictions.jsonl"
    total_edits = 0
    total_chars = 0
    rows: list[dict[str, Any]] = []

    with torch.inference_mode(), predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            audio_path = resolve_audio_path(record, dataset_root)
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio for {record.get('id')}: {audio_path}")
            audio, sample_rate = load_audio_mono(audio_path)
            prediction = generate_whisper_text(runtime, audio, sample_rate)
            edits, chars = cer(record["text"], prediction)
            total_edits += edits
            total_chars += chars
            row = {
                "index": index,
                "id": record.get("id"),
                "audio_path": str(audio_path),
                "reference": record["text"],
                "prediction": prediction,
                "edits": edits,
                "chars": chars,
                "cer": edits / chars,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[{index}/{len(records)}] {row['id']} CER={row['cer']:.3f}")

    metrics = {
        "manifest": str(manifest_path),
        "model": args.asr_model,
        "num_records": len(rows),
        "total_edits": total_edits,
        "total_chars": total_chars,
        "cer": total_edits / total_chars if total_chars else None,
        "predictions": str(predictions_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ASR model on a JSONL manifest.")
    parser.add_argument("--manifest", default="data/teochew_wild/prepared/splits/val.jsonl")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_val")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
