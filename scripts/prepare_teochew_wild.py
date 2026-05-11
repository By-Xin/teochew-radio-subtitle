from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


DEFAULT_REPO_ID = "panlr/teochew_wild"
ANNOTATION_FILE = "annotation.txt"
RAW_ANNOTATION_FILE = "raw_annotation.txt"
SPEAKER_INFO_FILE = "speaker_info.csv"
AUDIO_ARCHIVE = "teochew_wild.zip"
QWEN_LABEL_FILES = {
    "train": "label_for_qwen_asr/train.jsonl",
    "val": "label_for_qwen_asr/val.jsonl",
    "test": "label_for_qwen_asr/test.jsonl",
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_token(token_env: str) -> str | None:
    return os.environ.get(token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_hf_file(repo_id: str, filename: str, target_dir: Path, token: str | None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=target_dir,
        )
        return Path(path)
    except GatedRepoError as exc:
        raise RuntimeError(
            "teochew_wild is a gated Hugging Face dataset. "
            "Log in, open https://huggingface.co/datasets/panlr/teochew_wild, "
            "accept the dataset conditions, then set HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {exc}") from exc


def parse_annotation_line(line: str, line_number: int) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid annotation line {line_number}: expected 4 pipe-separated fields.")
    audio_relpath, speaker, text, pinyin = [part.strip() for part in parts]
    return {
        "id": Path(audio_relpath).stem,
        "audio": audio_relpath,
        "speaker": speaker,
        "text": text,
        "pinyin": pinyin,
        "source": "teochew_wild",
    }


def load_annotations(annotation_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = parse_annotation_line(line, line_number)
            if record:
                rows.append(record)
    return rows


def load_speaker_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        speakers: dict[str, dict[str, str]] = {}
        for row in reader:
            speaker = row.get("speaker") or row.get("Speaker") or row.get("id") or row.get("ID")
            if speaker:
                speakers[speaker] = row
        return speakers


def enrich_with_audio_paths(
    records: list[dict[str, Any]],
    audio_root: Path | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if audio_root:
            item["audio_path"] = str((audio_root / item["audio"]).resolve())
        enriched.append(item)
    return enriched


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]], speaker_info: dict[str, dict[str, str]]) -> dict[str, Any]:
    speakers = sorted({record["speaker"] for record in records})
    char_count = sum(len(record["text"]) for record in records)
    pinyin_token_count = sum(len(record["pinyin"].split()) for record in records)
    return {
        "num_records": len(records),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "speaker_info_rows": len(speaker_info),
        "char_count": char_count,
        "pinyin_token_count": pinyin_token_count,
    }


def extract_audio_archive(archive_path: Path, audio_root: Path, force: bool = False) -> None:
    marker = audio_root / ".extract_complete"
    if marker.exists() and not force:
        log(f"Audio archive already extracted at {audio_root}")
        return
    audio_root.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {archive_path} to {audio_root}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(audio_root)
    marker.write_text("ok\n", encoding="utf-8")


def strip_qwen_prefix(text: str) -> str:
    return re.sub(r"^language\s+Teochew<asr_text>", "", text).strip()


def normalize_qwen_label_file(path: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            audio = payload.get("audio")
            text = payload.get("text", "")
            if not audio:
                raise ValueError(f"Missing audio in {path}:{line_number}")
            records.append(
                {
                    "id": Path(audio).stem,
                    "audio": audio,
                    "text": strip_qwen_prefix(text),
                    "qwen_text": text,
                    "split": split,
                    "source": "teochew_wild_qwen_label",
                }
            )
    return records


def prepare_qwen_labels(repo_id: str, hf_dir: Path, out_dir: Path, token: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    qwen_out_dir = out_dir / "qwen_labels"
    for split, filename in QWEN_LABEL_FILES.items():
        label_path = download_hf_file(repo_id, filename, hf_dir, token)
        records = normalize_qwen_label_file(label_path, split)
        write_jsonl(records, qwen_out_dir / f"{split}.jsonl")
        counts[split] = len(records)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare panlr/teochew_wild manifests.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--data-dir", default="data/teochew_wild")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument("--download-audio", action="store_true", help="Download and extract teochew_wild.zip.")
    parser.add_argument("--skip-qwen-labels", action="store_true", help="Do not download label_for_qwen_asr JSONL files.")
    parser.add_argument("--force-extract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    hf_dir = data_dir / "hf"
    out_dir = data_dir / "prepared"
    audio_root = data_dir / "audio" if args.download_audio else None
    token = resolve_token(args.token_env)

    log(f"Preparing {args.repo_id} into {data_dir}")
    annotation_path = download_hf_file(args.repo_id, ANNOTATION_FILE, hf_dir, token)
    raw_annotation_path = download_hf_file(args.repo_id, RAW_ANNOTATION_FILE, hf_dir, token)
    speaker_info_path = download_hf_file(args.repo_id, SPEAKER_INFO_FILE, hf_dir, token)
    log(f"Downloaded annotations: {annotation_path}")
    log(f"Downloaded raw annotations: {raw_annotation_path}")
    log(f"Downloaded speaker info: {speaker_info_path}")

    if args.download_audio:
        archive_path = download_hf_file(args.repo_id, AUDIO_ARCHIVE, hf_dir, token)
        extract_audio_archive(archive_path, audio_root, force=args.force_extract)

    speaker_info = load_speaker_info(speaker_info_path)
    records = enrich_with_audio_paths(load_annotations(annotation_path), audio_root)
    manifest_path = out_dir / args.manifest_name
    write_jsonl(records, manifest_path)

    summary = summarize(records, speaker_info)
    if not args.skip_qwen_labels:
        summary["qwen_label_counts"] = prepare_qwen_labels(args.repo_id, hf_dir, out_dir, token)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Wrote manifest: {manifest_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
