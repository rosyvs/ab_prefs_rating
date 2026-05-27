"""Launch the A/B rating widget from a shared session config + rater id."""
from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

from ab_prefs_interface.interface_notebook import NotebookPreferenceInterface
from ab_prefs_interface.run_rating import run_notebook_rating
from ab_prefs_interface.summarize_preferences import summarize_cli_command

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSION_CONFIG = REPO_ROOT / "configs" / "ab_prefs.session.json"


def resolve_session_paths(session: dict, config_path: Path) -> dict:
    """Resolve notebook_root and repo-relative paths from session JSON."""
    config_path = config_path.expanduser().resolve()
    repo = Path(session["notebook_root"]).expanduser()
    if not repo.is_absolute():
        repo = (config_path.parent.parent / repo).resolve()
    out = dict(session)
    out["notebook_root"] = str(repo)
    for key in ("config_json", "session_manifest", "clip_dir", "cache_dir", "output_dir"):
        val = out.get(key)
        if not val:
            continue
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = (repo / p).resolve()
        out[key] = str(p)
    return out


def load_session_config(path: Path | str) -> dict:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return resolve_session_paths(payload, config_path)


def setup_repo_path(notebook_root: Path) -> None:
    repo = notebook_root.expanduser().resolve()
    os.chdir(repo)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def namespace_for_rater(session: dict, rater_id: str) -> Namespace:
    if not rater_id.strip():
        raise ValueError("rater_id is required (pass to launch_rating or set AB_PREFS_RATER_ID)")
    notebook_root = Path(session["notebook_root"]).expanduser().resolve()
    output_dir = Path(session.get("output_dir", notebook_root / "results/ab_prefs")).expanduser().resolve()
    session_manifest = session.get("session_manifest")
    export_session_manifest = session.get("export_session_manifest")
    per_pair = session.get("per_pair_sample_size")
    return Namespace(
        gt_dir=Path(session["gt_dir"]),
        audio_dir=Path(session["audio_dir"]),
        provider=[],
        config_json=Path(session["config_json"]),
        compare_providers=session.get("compare_providers", ""),
        ground_truth_name=str(session.get("ground_truth_name", "ground_truth")),
        include_ground_truth=bool(session.get("include_ground_truth", True)),
        strategy=str(session.get("strategy", "random")),
        session_items=int(session.get("session_items", 30)),
        per_pair_sample_size=per_pair if per_pair is None else int(per_pair),
        seed=int(session.get("seed", 7)),
        output_json=output_dir / f"preferences_{rater_id.strip()}.json",
        verbose=bool(session.get("verbose", True)),
        cache_dir=Path(session["cache_dir"]),
        rebuild_cache=bool(session.get("rebuild_cache", False)),
        unique_recordings=int(session["unique_recordings"]) if session.get("unique_recordings") is not None else None,
        recording_seed=int(session.get("recording_seed", 7)),
        min_gt_words=int(session.get("min_gt_words", 0)),
        min_audio_seconds=float(session.get("min_audio_seconds", 3.0)),
        show_note=bool(session.get("show_note", False)),
        show_providers=bool(session.get("show_providers", False)),
        clip_dir=Path(session["clip_dir"]),
        notebook_root=notebook_root,
        session_manifest=Path(session_manifest) if session_manifest else None,
        export_session_manifest=Path(export_session_manifest) if export_session_manifest else None,
        rater_id=rater_id.strip(),
    )


def launch_rating(
    rater_id: str | None = None,
    session_config_path: Path | str | None = None,
) -> NotebookPreferenceInterface:
    """Load shared session config, open rating widget. Colleagues only set rater_id."""
    rater = (rater_id or os.environ.get("AB_PREFS_RATER_ID", "")).strip()
    config_path = Path(session_config_path or os.environ.get("AB_PREFS_SESSION_CONFIG", DEFAULT_SESSION_CONFIG))
    session = load_session_config(config_path)
    setup_repo_path(Path(session["notebook_root"]))
    args = namespace_for_rater(session, rater)
    print(f"Rater: {rater} → {args.output_json}")
    print(f"Session config: {config_path.resolve()}")
    if args.session_manifest:
        print(f"Items from: {args.session_manifest.resolve()}")
    print(f"\nSummarize after rating (separate, optional):\n{summarize_cli_command(args.output_json, args.ground_truth_name)}")
    return run_notebook_rating(args)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch A/B rating widget for one rater")
    parser.add_argument("--rater-id", type=str, default="", help="Unique rater label (or set AB_PREFS_RATER_ID)")
    parser.add_argument(
        "--session-config",
        type=Path,
        default=DEFAULT_SESSION_CONFIG,
        help="Shared session JSON (paths, manifest, sampling settings)",
    )
    args = parser.parse_args()
    launch_rating(rater_id=args.rater_id or None, session_config_path=args.session_config)
