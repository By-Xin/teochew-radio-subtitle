from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {row["id"]: row for row in load_jsonl(path)}


def bucket_length(chars: int) -> str:
    if chars <= 10:
        return "001-010"
    if chars <= 20:
        return "011-020"
    if chars <= 40:
        return "021-040"
    if chars <= 80:
        return "041-080"
    return "081+"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(row["edits"] for row in rows)
    chars = sum(row["chars"] for row in rows)
    return {
        "count": len(rows),
        "edits": edits,
        "chars": chars,
        "cer": edits / chars if chars else None,
    }


def levenshtein_ops(reference: str, hypothesis: str) -> list[tuple[str, str, str]]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("ins", 0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1][j] + 1, ("del", i - 1, j)),
                (dp[i][j - 1] + 1, ("ins", i, j - 1)),
                (dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]), ("sub", i - 1, j - 1)),
            ]
            cost, op = min(choices, key=lambda item: item[0])
            dp[i][j] = cost
            back[i][j] = op

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op is None:
            break
        kind, ref_idx, hyp_idx = op
        if kind == "sub":
            if reference[ref_idx] != hypothesis[hyp_idx]:
                ops.append(("sub", reference[ref_idx], hypothesis[hyp_idx]))
            i -= 1
            j -= 1
        elif kind == "del":
            ops.append(("del", reference[ref_idx], ""))
            i -= 1
        elif kind == "ins":
            ops.append(("ins", "", hypothesis[hyp_idx]))
            j -= 1
    ops.reverse()
    return ops


def analyze(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    predictions = load_jsonl(predictions_path)
    for row in predictions:
        meta = manifest.get(row.get("id"), {})
        row["speaker"] = meta.get("speaker", "unknown")
        row["length_bucket"] = bucket_length(row["chars"])

    total = summarize_group(predictions)
    by_speaker = {
        speaker: summarize_group(rows)
        for speaker, rows in sorted(group_by(predictions, "speaker").items())
    }
    by_length = {
        bucket: summarize_group(rows)
        for bucket, rows in sorted(group_by(predictions, "length_bucket").items())
    }
    worst = sorted(predictions, key=lambda row: (row["cer"], row["edits"]), reverse=True)[: args.top_k]

    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    for row in predictions:
        for kind, ref, hyp in levenshtein_ops(row["reference"], row["prediction"]):
            if kind == "sub":
                substitutions[(ref, hyp)] += 1
            elif kind == "del":
                deletions[ref] += 1
            elif kind == "ins":
                insertions[hyp] += 1

    report = {
        "predictions": str(predictions_path),
        "total": total,
        "by_speaker": by_speaker,
        "by_length": by_length,
        "worst": worst,
        "top_substitutions": [
            {"reference": ref, "prediction": hyp, "count": count}
            for (ref, hyp), count in substitutions.most_common(args.top_k)
        ],
        "top_deletions": [
            {"reference": ref, "count": count}
            for ref, count in deletions.most_common(args.top_k)
        ],
        "top_insertions": [
            {"prediction": hyp, "count": count}
            for hyp, count in insertions.most_common(args.top_k)
        ],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "worst.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in worst),
        encoding="utf-8",
    )
    print(json.dumps({"total": total, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze ASR prediction JSONL output.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
