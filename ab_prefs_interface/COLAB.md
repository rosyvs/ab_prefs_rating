# A/B preference rating on Colab / Workbench

Team members rate in the browser; GT audio + transcripts load from GCS. Code + manifest come from the **`ab_prefs_rating`** repo (`pip install -e .`).

## GCS layout (bucket `dd_tfx_full_transcripts`)

Lead uploads once:

```
gs://dd_tfx_full_transcripts/
  transcripts/          # {recording_id}.jsonl  (ground truth)
  audio/                # {recording_id}.mp3
  asr/dd210/            # ASR JSONs (same tree as results/dd210 locally)
    aai/universal-3-pro-v2/
    azure/fast-transcription/
    deepgram/nova-3/
    ...
```

Sync ASR results from your machine:

```bash
gsutil -m rsync -r results/dd210 gs://dd_tfx_full_transcripts/asr/dd210
gsutil -m rsync -r /path/to/transcripts gs://dd_tfx_full_transcripts/transcripts
gsutil -m rsync -r /path/to/audio gs://dd_tfx_full_transcripts/audio
```

Commit `configs/ab_prefs.manifest.json` in git so every rater sees the same items.

## Access control

**No anonymous access** — do not grant `allUsers` (public read).

Authenticated access is fine: e.g. `allAuthenticatedUsers` with `objectViewer`, a Google Group, or named accounts — whatever your team already uses for this bucket.

| Role | Purpose |
|------|---------|
| `roles/storage.objectViewer` | Read transcripts, audio, ASR JSONs via gcsfuse |
| `roles/storage.objectCreator` on prefix `ab_prefs/` | Upload ratings JSON (optional) |

Colab: sign in with Google → IAM check (`gsutil ls gs://…/transcripts/`) → gcsfuse mount. Must be signed in; accounts without bucket IAM fail at the check. Audio is read through the mount, not copied to disk.

## Rater: Colab

1. Open in Colab: [rate_colab.ipynb](https://colab.research.google.com/github/rosyvs/ab_prefs_rating/blob/main/ab_prefs_interface/rate_colab.ipynb) (or GitHub → notebook → **Open in Colab**).
2. Set **RATER_ID** (unique per person).
3. **Runtime → Run all**. First run: clone repo → `pip install -e .` → Google auth → gcsfuse mount (~1–2 min).
4. Use **Choose A / B / Tie / Skip**.
5. Download `preferences_{RATER_ID}.json` from Files, or copy to bucket:

```python
!gsutil cp results/ab_prefs/preferences_Alice.json gs://dd_tfx_full_transcripts/ab_prefs/preferences_Alice.json
```

## Rater: Vertex Workbench

If the bucket is already visible (e.g. `/home/jupyter/gcs/dd_tfx_full_transcripts`), set in the notebook:

```python
MOUNT_POINT = "/home/jupyter/gcs/dd_tfx_full_transcripts"
```

Then run the same cells (mount step becomes a no-op).

## Lead: regenerate manifest (local or Workbench)

```bash
cd /content/ab_prefs_rating  # or local clone
python -m ab_prefs_interface.create_session \
  --session-config configs/ab_prefs.session.runtime.json \
  --rebuild-cache
git add configs/ab_prefs.manifest.json && git commit ...
```

Or edit `compare_providers` / `session_items` in `colab_setup.write_colab_session_config` defaults before bootstrap.

## Dependencies (installed by notebook)

- `pip install -e .` — `ipywidgets`, `jiwer`, `pandas`, `tqdm` (see `pyproject.toml`)
- `ffmpeg` — GT-span audio clips
- `gcsfuse` — Colab only

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'jiwer'` | Re-run the main setup cell (needs `pip install -e .` before imports); pull latest notebook from GitHub |
| `No authenticated access` / PermissionError | Sign in with a Google account that has bucket access (objectViewer) |
| `apt-get install gcsfuse` exit 100 | Stale clone — Runtime → Restart session, delete `/content/ab_prefs_rating`, re-run; notebook now git pulls + installs .deb fallback |
| gcsfuse mount failed | Colab FUSE limitation — use Vertex Workbench with bucket pre-mounted instead |
| `transcripts/` not found after mount | Check bucket layout; confirm IAM |
| Missing ASR json for recording | `gsutil rsync` asr/dd210; demo uses 5 recordings from manifest |
| Widget blank | Colab: re-run setup cell (`enable_custom_widget_manager`) |
| Different items than colleagues | Pull latest git; same `manifest.json` |
