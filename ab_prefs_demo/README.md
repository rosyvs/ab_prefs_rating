# A/B ASR preference demo

Python notebook workflow for human preference testing between ASR transcripts on matched GT audio spans.

**→ Full usage guide: [USAGE.md](USAGE.md)**

Quick start for raters: open `rate.ipynb`, set `RATER_ID`, Run All.

Quick start for lead:

```bash
cd /home/rosy_teachfx_com/asr_eval
python -m ab_prefs_demo.create_session \
  --session-config /home/rosy_teachfx_com/asr_eval/configs/ab_prefs_demo.session.json \
  --rebuild-cache
```

Configs live in `configs/ab_prefs_demo.{providers,session,manifest}.json`.
