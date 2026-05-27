# A/B ASR preference rating — usage

Human preference testing: listen to a GT audio clip, compare two transcripts side-by-side, pick A / B / tie / skip. Results saved to JSON.

**Standalone repo:** `ab_prefs_rating/` in this monorepo is the exportable package (`pip install -e .`, minimal deps). Push it to `github.com/rosyvs/ab_prefs_rating` for Colab raters. Refresh from here: `scripts/sync_ab_prefs_rating.sh`.

**Raters:** open `ab_prefs_interface/rate.ipynb` (local) or `ab_prefs_interface/rate_colab.ipynb` (Colab / Workbench), set `RATER_ID`, Run All.

**Colab / GCS:** see `ab_prefs_interface/COLAB.md`.

**Lead:** edit configs, regenerate manifest when study setup changes.

---

## Config files (`configs/`)

| File | Purpose | Who edits |
|------|---------|-----------|
| `ab_prefs.providers.json` | ASR name → folder of `{recording_id}.json` transcripts | Lead (when adding/swapping models) |
| `ab_prefs.session.json` | Paths, merge rules, sampling, recording pool size, UI flags | Lead |
| `ab_prefs.manifest.json` | Fixed list of 30 A/B trials (recording, span, pair) | Lead regenerates; team commits |

**Generated outputs** (`results/ab_prefs/`):

| Path | Purpose |
|------|---------|
| `preferences_{RATER_ID}.json` | One rater's choices |
| `unit_cache/` | Pickled comparison units (speed) |
| `audio_clips/` | Short mp3 clips per GT span |

### providers.json

All eight DD210 ASR runs are listed (see `results/dd210_asr_comparison.md` for labels). Ground truth is **not** in this file — it comes from `gt_dir` and is optionally added to pairs via `include_ground_truth` in `session.json`.

### session.json

Key fields:

- `gt_dir`, `audio_dir` — ground truth JSONL + `{id}.mp3` audio
- `config_json` — points at `providers.json`
- `session_manifest` — points at `manifest.json`
- `unique_recordings` — cap on distinct recording IDs in pool/manifest (e.g. `5` pilot; omit/`null` for full DD210)
- `recording_seed` — RNG seed when sampling the recording pool
- `min_gt_words`, `min_audio_seconds` — merge consecutive GT lines until both met
- `exclude_gt_markers` — default `true`: drop GT lines with `should_scrub`, `XXX`, `SCRUB`/`<SCRUB>`, `<crosstalk>` when **creating** a new manifest (not when loading an existing one)
- `asr_eval_root` — path to `asr_eval` repo; sampling WER uses `analyze_multiple_asr` + provider JSON under `results/dd210/` (via `providers.json`)
- `session_items`, `seed`, `strategy` — used when **creating** manifest
- `include_ground_truth` — `true`: GT included in pair pool (with empty `compare_providers`); `false`: ASR-vs-ASR only
- `compare_providers` — `""` = all names from `providers.json` (+ GT if `include_ground_truth`); or comma list to restrict
- `show_providers` — `false` = blind A/B labels; `true` = debug (shows model names)
- `rebuild_cache` — set `true` once after code/GT changes, then back to `false`

---

## Lead workflow

### 1. Edit study settings

Update `configs/ab_prefs.session.json` (and `providers.json` if models changed).

### 2. Regenerate manifest + unit cache

```bash
cd /home/rosy_teachfx_com/asr_eval
python -m ab_prefs_interface.create_session \
  --session-config /home/rosy_teachfx_com/asr_eval/configs/ab_prefs.session.json \
  --rebuild-cache
```

Writes `configs/ab_prefs.manifest.json` (30 items, balanced ASR exposure).

Use `--rebuild-cache` when matching/display rules change or cache is stale. To wipe subset cache manually:

```bash
rm -rf /home/rosy_teachfx_com/asr_eval/results/ab_prefs/unit_cache/subset
```

### 3. Commit for team

Share via git (or bucket): `configs/*.json`, not rater preference files.

---

## Rater workflow

### 1. Open notebook

`ab_prefs_interface/rate.ipynb` in Cursor / Jupyter / Colab / Workbench.

### 2. Set your id

```python
RATER_ID = "Alice"
```

### 3. Run All

Widget appears: audio player, two transcript columns, **Choose A / B / Tie / Skip**.

Choices append to `results/ab_prefs/preferences_Alice.json`.

**Note:** "Computed WER features for…" on startup runs **DD210 lexical WER** (`analyze_multiple_asr` on ASR JSON from `results/dd210/`) for item sampling — not human ratings.

### 4. Summarize (optional)

Command is printed at launch and when you finish all items:

```bash
python -m ab_prefs_interface.summarize_preferences \
  --input-json /home/rosy_teachfx_com/asr_eval/results/ab_prefs/preferences_Alice.json \
  --ground-truth-name ground_truth
```

Win rates shown as ratios (e.g. `0.500`), not raw counts.

---

## Sampling strategies (lead, when creating manifest)

| Strategy | Picks items with… |
|----------|-------------------|
| `random` | Random order (default) |
| `max_discrepancy` | Highest WER spread across ASRs vs GT |
| `max_wer` | Highest average WER vs GT |
| `most_deletions` | Most deletions vs GT |

All raters see the same items via `manifest.json`, regardless of strategy name stored in metadata.

---

## Display notes

- **Word-level ASR** (`response.words`, `utterances[].words`, or `segments[].words` — e.g. AAI, Azure fast/LLM, Deepgram nova-3): word-by-word highlight during playback.
- **Segment-level ASR** (utterance/segment bounds only, no word arrays — e.g. Azure mai, Gemini): phrase rows; utterances need ≥1s overlap with GT span to appear; grey + note if utterance extends before/after the clip.
- Audio is a **short clip** of the GT span (not the full recording).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: ab_prefs_interface` | Notebook cell bootstraps repo path; restart kernel and re-run |
| Play button dead | Fixed: clips embedded as base64; re-run launch cell |
| Wrong/old utterances showing | Restart kernel, re-run launch cell (or `--rebuild-cache` after GT/code changes) |
| Summarize shows old data | Point `--input-json` at your `preferences_{RATER_ID}.json` |
| Colleagues see different items | Everyone must use same committed `manifest.json` |

---

## Input data

See `ab_prefs_interface/INPUT_SCHEMA.md` for GT JSONL, provider JSON, and audio layout.

## Module map

| Module | Role |
|--------|------|
| `launch.py` | `launch_rating(rater_id)` from session config |
| `create_session.py` | Lead: build manifest |
| `interface_notebook.py` | A/B widget |
| `matching.py` | GT span ↔ provider alignment |
| `sampling.py` | Queue building + balanced ASR exposure |
| `summarize_preferences.py` | Aggregate win rates |
| `storage_json.py` | Append/read preference JSON |
