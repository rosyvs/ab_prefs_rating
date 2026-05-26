import os, sys
from pathlib import Path

repo = Path.cwd()
for _ in range(6):
    if (repo / "configs" / "ab_prefs_demo.session.json").exists():
        break
    repo = repo.parent
else:
    repo = Path("/home/rosy_teachfx_com/asr_eval")
os.chdir(repo)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from ab_prefs_demo.launch import launch_rating

RATER_ID = "Rosy1"
interface = launch_rating(RATER_ID)
interface
