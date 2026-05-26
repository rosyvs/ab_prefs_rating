# ASR A/B preference rating

Minimal standalone package for comparing two ASR transcripts on ground-truth audio clips. No dependency on the main `asr_eval` repo.

## Requirements

- Python 3.10+
- **ffmpeg** on PATH (audio clip extraction)
- Your data: GT transcript JSONL dir, source audio dir, ASR result JSON dirs

## Install

```bash
git clone https://github.com/rosyvs/ab_prefs_rating.git
cd ab_prefs_rating
pip install -e .
# or: pip install -r requirements.txt && pip install -e .
```

## Local workflow (lead)

1. Copy and edit session config (paths to your GT/audio):

```bash
cp configs/ab_prefs_demo.session.example.json configs/ab_prefs_demo.session.json
cp configs/ab_prefs_demo.providers.example.json configs/ab_prefs_demo.providers.json
# edit gt_dir, audio_dir, provider dirs
```

2. Build shared item queue + unit cache:

```bash
python -m ab_prefs_demo.create_session --session-config configs/ab_prefs_demo.session.json
# add --rebuild-cache to refresh cached units
```

3. Open `ab_prefs_demo/rate.ipynb`, set `RATER_ID`, run all cells.

Ratings → `results/ab_prefs/preferences_{RATER_ID}.json`

Summarize (optional):

```bash
python -m ab_prefs_demo.summarize_preferences results/ab_prefs/preferences_Alice.json
```

## Colab / GCS team workflow

See [ab_prefs_demo/COLAB.md](ab_prefs_demo/COLAB.md). One-click notebook: `ab_prefs_demo/rate_colab.ipynb`.

Data layout on bucket `dd_tfx_full_transcripts`:

```
gs://dd_tfx_full_transcripts/transcripts/   # {recording_id}.jsonl
gs://dd_tfx_full_transcripts/audio/         # {recording_id}.mp3
gs://dd_tfx_full_transcripts/asr/dd210/     # ASR JSON trees (lead rsyncs)
```

## Config files

| File | Purpose |
|------|---------|
| `configs/ab_prefs_demo.session.example.json` | Template — copy to `session.json` |
| `configs/ab_prefs_demo.providers.example.json` | Template — copy to `providers.json` |
| `configs/ab_prefs_demo.providers.colab.json` | Same; paths patched at Colab bootstrap |
| `configs/ab_prefs_demo.manifest.json` | Fixed item queue (regenerate with `create_session`) |

More detail: [ab_prefs_demo/USAGE.md](ab_prefs_demo/USAGE.md)
