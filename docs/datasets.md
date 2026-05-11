# Datasets

## Teochew-Wild

Primary dataset:

```text
panlr/teochew_wild
```

Dataset card:

<https://huggingface.co/datasets/panlr/teochew_wild>

This is the first dataset we use for repeatable Teochew ASR evaluation. It contains 12,500 clips, about 18.87 hours, 20 speakers, and orthographic plus Teochew Pinyin annotations.

The dataset is gated on Hugging Face. Before running the preparation script:

1. Log in to Hugging Face in a browser.
2. Open <https://huggingface.co/datasets/panlr/teochew_wild>.
3. Accept the dataset access conditions.
4. Set a token in the shell:

```powershell
$env:HF_TOKEN = "hf_..."
```

Prepare annotation manifests only:

```powershell
python scripts/prepare_teochew_wild.py --data-dir data/teochew_wild
```

Prepare annotations and download/extract the 2.6 GB audio archive:

```powershell
python scripts/prepare_teochew_wild.py `
  --data-dir data/teochew_wild `
  --download-audio
```

Generated files:

- `data/teochew_wild/hf/annotation.txt`
- `data/teochew_wild/hf/raw_annotation.txt`
- `data/teochew_wild/hf/speaker_info.csv`
- `data/teochew_wild/prepared/manifest.jsonl`
- `data/teochew_wild/prepared/summary.json`
- `data/teochew_wild/prepared/qwen_labels/train.jsonl`
- `data/teochew_wild/prepared/qwen_labels/val.jsonl`
- `data/teochew_wild/prepared/qwen_labels/test.jsonl`

The manifest format is one JSON object per line:

```json
{
  "id": "S001F001C001",
  "audio": "S001/S001F001/S001F001C001.wav",
  "audio_path": "D:/.../data/teochew_wild/audio/S001/S001F001/S001F001C001.wav",
  "speaker": "S001",
  "text": "伊祇个人",
  "pinyin": "i1 zi2 gai5 nang5",
  "source": "teochew_wild"
}
```

`audio_path` is only included when `--download-audio` is used.

## License Note

The Hugging Face dataset page has shown mixed license metadata and notes. The page metadata lists `cc-by-4.0`, while the dataset card notes non-commercial usage under `CC BY-NC-4.0` and states that audio copyright belongs to the original source owners.

Treat this dataset as research/evaluation data unless the license is clarified for a commercial use case.
