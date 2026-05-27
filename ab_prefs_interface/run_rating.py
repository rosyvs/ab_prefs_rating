from __future__ import annotations

import argparse
import json
from pathlib import Path

from ab_prefs_interface.interface_notebook import NotebookPreferenceInterface
from ab_prefs_interface.session_config import compare_provider_names, session_recording_pool_size
from ab_prefs_interface.matching import build_comparison_units
from ab_prefs_interface.sampling import SAMPLING_STRATEGIES, build_session_queue, filter_queue_with_transcripts
from ab_prefs_interface.scoring import score_units
from ab_prefs_interface.session_manifest import (
    build_manifest_payload,
    load_session_manifest,
    manifest_recording_ids,
    queue_from_manifest,
    save_session_manifest,
)
from ab_prefs_interface.storage_json import initialize_store


def parse_provider_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Provider argument must look like name=/path/to/jsons, got {value}")
    name, path_string = value.split("=", 1)
    name_clean = name.strip()
    if not name_clean:
        raise ValueError(f"Provider name is empty in {value}")
    return name_clean, Path(path_string).expanduser().resolve()


def load_provider_dirs(args: argparse.Namespace) -> dict[str, Path]:
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        providers = payload.get("providers")
        if not isinstance(providers, dict) or not providers:
            raise ValueError("config_json must contain a non-empty 'providers' object")
        return {str(name): Path(path).expanduser().resolve() for name, path in providers.items()}
    if not args.provider:
        raise ValueError("Provide at least one --provider name=/path OR pass --config-json")
    provider_dirs: dict[str, Path] = {}
    for item in args.provider:
        provider_name, provider_path = parse_provider_arg(item)
        provider_dirs[provider_name] = provider_path
    return provider_dirs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B ASR preference rating session")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with GT {recording_id}.jsonl files")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory with audio {recording_id}.mp3 files")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider input as name=/path/to/provider_json_dir (repeatable)",
    )
    parser.add_argument("--config-json", type=Path, help="Optional JSON config with providers map")
    parser.add_argument(
        "--compare-providers",
        type=str,
        default="",
        help="Comma-separated provider names to include in A/B queue. Empty means all providers plus ground_truth.",
    )
    parser.add_argument(
        "--ground-truth-name",
        type=str,
        default="ground_truth",
        help="Provider label used for GT in pair generation",
    )
    parser.add_argument(
        "--include-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include ground_truth in A/B pair pool when compare_providers is empty",
    )
    parser.add_argument("--strategy", choices=SAMPLING_STRATEGIES, default="random")
    parser.add_argument(
        "--session-items",
        type=int,
        default=30,
        help="Total A/B items in the review session (split across provider pairs)",
    )
    parser.add_argument(
        "--per-pair-sample-size",
        type=int,
        default=None,
        help="Override: fixed items per provider pair instead of splitting session-items",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ab_prefs/preferences.json"),
        help="Output JSON path for saved preferences",
    )
    parser.add_argument("--verbose", action="store_true", help="tqdm progress + stage printouts")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/ab_prefs/unit_cache"),
        help="Per-recording pickle cache for comparison units (pass empty string to disable)",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore existing unit cache and rebuild")
    parser.add_argument(
        "--unique-recordings",
        type=int,
        default=0,
        help="Random recording pool size when building units (uses unit_cache/subset/)",
    )
    parser.add_argument("--recording-seed", type=int, default=7, help="RNG seed for recording pool sampling")
    parser.add_argument(
        "--min-gt-words",
        type=int,
        default=0,
        help="Merge consecutive GT lines until at least this many whitespace-separated words (0=off)",
    )
    parser.add_argument("--show-note", action="store_true", help="Show note textbox by default (else use Add note checkbox)")
    parser.add_argument(
        "--show-providers",
        action="store_true",
        help="Show provider names and recording/segment ids in the UI (blind A/B by default)",
    )
    parser.add_argument(
        "--min-audio-seconds",
        type=float,
        default=3.0,
        help="Merge consecutive GT lines until span is at least this long in seconds (0=off)",
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=Path("results/ab_prefs/audio_clips"),
        help="Cached GT-span mp3 clips served to the notebook UI",
    )
    parser.add_argument(
        "--notebook-root",
        type=Path,
        default=Path("."),
        help="Repo/content root for files/... audio URLs (set to repo path in Colab/Workbench)",
    )
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=None,
        help="Load fixed session queue from JSON (same items for all raters)",
    )
    parser.add_argument(
        "--export-session-manifest",
        type=Path,
        default=None,
        help="Write session queue manifest after sampling (for team distribution)",
    )
    parser.add_argument(
        "--rater-id",
        type=str,
        default="",
        help="Optional label stored in session_id (e.g. alice); each rater uses own --output-json",
    )
    return parser


def run_notebook_rating(args: argparse.Namespace) -> NotebookPreferenceInterface:
    verbose = bool(getattr(args, "verbose", False))
    provider_dirs = load_provider_dirs(args)
    compare_providers = compare_provider_names(
        provider_dirs,
        ground_truth_name=args.ground_truth_name,
        compare_providers=args.compare_providers,
        include_ground_truth=bool(getattr(args, "include_ground_truth", True)),
    )
    asr_names = [n for n in compare_providers if n != args.ground_truth_name]
    provider_dirs = {n: provider_dirs[n] for n in asr_names}
    manifest_path = getattr(args, "session_manifest", None)
    manifest = None
    recording_ids = None
    if manifest_path:
        manifest = load_session_manifest(manifest_path)
        recording_ids = manifest_recording_ids(manifest)
        if verbose:
            print(f"Manifest: {len(manifest['items'])} items, {len(recording_ids)} recordings")
    raw_unique = getattr(args, "unique_recordings", None)
    unique_n = int(raw_unique) if raw_unique is not None and int(raw_unique) > 0 else None
    pool_n = unique_n
    if verbose:
        print("Loading comparison units...")
    cache_dir = args.cache_dir.expanduser().resolve() if str(args.cache_dir).strip() else None
    units = build_comparison_units(
        gt_dir=args.gt_dir.expanduser().resolve(),
        provider_dirs=provider_dirs,
        audio_dir=args.audio_dir.expanduser().resolve(),
        ground_truth_name=args.ground_truth_name,
        verbose=verbose,
        cache_dir=cache_dir,
        rebuild_cache=bool(getattr(args, "rebuild_cache", False)),
        recording_pool_size=pool_n if pool_n and not recording_ids else None,
        recording_seed=int(getattr(args, "recording_seed", 7)),
        recording_ids=recording_ids,
        min_gt_words=int(getattr(args, "min_gt_words", 0) or 0),
        min_audio_seconds=float(getattr(args, "min_audio_seconds", 3.0) or 0.0),
    )
    if verbose:
        print("Scoring units vs ground truth...")
    score_units(
        units=units,
        provider_names=sorted(provider_dirs.keys()),
        ground_truth_name=args.ground_truth_name,
        verbose=verbose,
    )
    if verbose:
        gt_note = "with ground_truth" if args.ground_truth_name in compare_providers else "ASR only"
        print(f"Compare pool ({gt_note}): {compare_providers}")
    if len(compare_providers) < 2:
        raise ValueError("Need at least two providers in compare_providers")
    per_pair = getattr(args, "per_pair_sample_size", None)
    session_items = int(getattr(args, "session_items", 30))
    min_gt_words = int(getattr(args, "min_gt_words", 0) or 0)
    min_audio_seconds = float(getattr(args, "min_audio_seconds", 3.0) or 0.0)
    unique_target = unique_n
    if manifest:
        if verbose:
            print(f"Loading session queue from {manifest_path}...")
        queue = queue_from_manifest(manifest, units)
        before = len(queue)
        queue = filter_queue_with_transcripts(queue)
        expected = int(manifest.get("session_items", session_items))
        if before > len(queue) and verbose:
            print(f"Dropped {before - len(queue)} manifest items with no overlapping transcript for both providers")
        if len(queue) < expected:
            raise ValueError(
                f"Manifest has only {len(queue)} eligible items but session_items={expected}. "
                "Regenerate: python -m ab_prefs_interface.create_session --session-config ... --rebuild-cache"
            )
    else:
        if verbose:
            print("Building session queue...")
        queue = build_session_queue(
            units=units,
            provider_names=compare_providers,
            strategy=args.strategy,
            seed=args.seed,
            session_items=session_items,
            per_pair_sample_size=per_pair,
            ground_truth_name=args.ground_truth_name,
            verbose=verbose,
            unique_recordings=unique_target,
        )
    if not queue:
        raise ValueError("Sampling queue is empty; check provider files and compare provider list.")
    export_manifest = getattr(args, "export_session_manifest", None)
    if export_manifest and not manifest_path:
        save_session_manifest(
            export_manifest,
            build_manifest_payload(
                queue,
                strategy=args.strategy,
                seed=args.seed,
                session_items=session_items,
                ground_truth_name=args.ground_truth_name,
                compare_providers=compare_providers,
                include_ground_truth=bool(getattr(args, "include_ground_truth", True)),
                min_gt_words=min_gt_words,
                min_audio_seconds=min_audio_seconds,
                unique_recordings=unique_target,
                recording_seed=int(getattr(args, "recording_seed", 7)),
            ),
        )
        if verbose:
            print(f"Wrote session manifest → {export_manifest.resolve()}")
    output_path = args.output_json.expanduser().resolve()
    initialize_store(output_path)
    rater_id = str(getattr(args, "rater_id", "") or "").strip()
    session_id = rater_id if rater_id else None
    notebook_root = args.notebook_root.expanduser().resolve()
    clip_dir = args.clip_dir.expanduser().resolve()
    if verbose:
        print(f"Launching UI ({len(queue)} items) → {output_path}")
        print(f"Audio clips cached under {clip_dir}")
    interface = NotebookPreferenceInterface(
        queue=queue,
        output_json_path=output_path,
        strategy=args.strategy,
        session_id=session_id,
        show_note=bool(getattr(args, "show_note", False)),
        show_providers=bool(getattr(args, "show_providers", False)),
        clip_dir=clip_dir,
        notebook_root=notebook_root,
        verbose=verbose,
        ground_truth_name=args.ground_truth_name,
    )
    interface.show()
    return interface


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_notebook_rating(args)


if __name__ == "__main__":
    main()
