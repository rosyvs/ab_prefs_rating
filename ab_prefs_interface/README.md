# A/B ASR preference rating

Python notebook workflow for human preference testing between ASR transcripts on matched GT audio spans.

**→ Full usage guide: [USAGE.md](USAGE.md)**

Quick start for raters: open `rate.ipynb`, set `RATER_ID`, Run All.

Quick start for lead:

```bash
cd ab_prefs_rating
python -m ab_prefs_interface.create_session \
  --session-config configs/ab_prefs.session.json \
  --rebuild-cache
```

Configs live in `configs/ab_prefs.{providers,session,manifest}.json`.
