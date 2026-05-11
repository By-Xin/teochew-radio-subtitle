# Evaluation

## Whisper Teochew Baseline

Model:

```text
panlr/whisper-finetune-teochew
```

Evaluation manifest:

```text
data/teochew_wild/prepared/splits/val.jsonl
```

Smoke test command:

```powershell
python scripts/evaluate_asr_manifest.py `
  --manifest data/teochew_wild/prepared/splits/val.jsonl `
  --limit 20 `
  --output-dir outputs/eval_val_20
```

Current smoke result:

```json
{
  "num_records": 20,
  "total_edits": 12,
  "total_chars": 389,
  "cer": 0.030848329048843187
}
```

This is only a smoke baseline, not a full validation result. Run the full `val` and `test` manifests before comparing model changes.

## Important Decode Setting

Whisper must be forced into Chinese transcription mode. Without this, some clips can be decoded as English translations, which makes CER meaningless.

The shared inference runtime sets:

```python
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="chinese",
    task="transcribe",
)
```
