"""Lead: build session_manifest.json from configs/ab_prefs.session.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from ab_prefs_interface.launch import DEFAULT_SESSION_CONFIG, load_session_config, namespace_for_rater, setup_repo_path
from ab_prefs_interface.session_config import compare_provider_names, session_recording_pool_size, session_unique_recordings_target
from ab_prefs_interface.matching import build_comparison_units
from ab_prefs_interface.run_rating import load_provider_dirs
from ab_prefs_interface.sampling import build_session_queue, pair_combinations
from ab_prefs_interface.scoring import score_units
from ab_prefs_interface.session_manifest import build_manifest_payload, copy_manifest_from_source, save_session_manifest


def create_session_manifest(
    session_config_path: Path,
    manifest_path: Path | None = None,
    *,
    rebuild_cache: bool = False,
    from_manifest: Path | None = None,
    exclude_scrub: bool = False,
) -> Path:
    session = load_session_config(session_config_path)
    setup_repo_path(Path(session["notebook_root"]))
    out = manifest_path or Path(session["session_manifest"])
    if from_manifest is not None:
        payload = copy_manifest_from_source(
            from_manifest,
            gt_dir=Path(session["gt_dir"]),
            exclude_scrub=exclude_scrub,
        )
        save_session_manifest(out, payload)
        n_unique = len({item["recording_id"] for item in payload["items"]})
        dropped = len(payload.get("source_manifest_filter", {}).get("dropped_span_keys", []))
        print(
            f"Wrote {len(payload['items'])} items ({n_unique} unique recordings)"
            + (f", dropped {dropped} scrub spans" if dropped else "")
            + f" → {out.resolve()}"
        )
        return out.resolve()
    args = namespace_for_rater(session, rater_id="_lead_export_")
    if rebuild_cache:
        args.rebuild_cache = True
    provider_dirs = load_provider_dirs(args)
    pool_n = session_recording_pool_size(session)
    unique_n = session_unique_recordings_target(session)
    units = build_comparison_units(
        gt_dir=args.gt_dir.expanduser().resolve(),
        provider_dirs=provider_dirs,
        audio_dir=args.audio_dir.expanduser().resolve(),
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose,
        cache_dir=args.cache_dir.expanduser().resolve(),
        rebuild_cache=bool(args.rebuild_cache),
        recording_pool_size=pool_n,
        recording_seed=int(args.recording_seed),
        min_gt_words=int(args.min_gt_words or 0),
        min_audio_seconds=float(args.min_audio_seconds or 0.0),
        exclude_gt_markers=bool(getattr(args, "exclude_gt_markers", True)),
    )
    asr_eval_root = getattr(args, "asr_eval_root", None)
    if not asr_eval_root:
        raise ValueError("session config must set asr_eval_root (path to asr_eval repo for DD210 WER sampling)")
    score_units(
        units=units,
        provider_names=sorted(provider_dirs.keys()),
        gt_dir=args.gt_dir.expanduser().resolve(),
        provider_dirs=provider_dirs,
        asr_eval_root=Path(asr_eval_root).expanduser().resolve(),
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose,
    )
    compare_providers = compare_provider_names(
        provider_dirs,
        ground_truth_name=args.ground_truth_name,
        compare_providers=args.compare_providers,
        include_ground_truth=bool(getattr(args, "include_ground_truth", True)),
    )
    per_pair = args.per_pair_sample_size
    n_pairs = len(pair_combinations(compare_providers))
    expected = per_pair * n_pairs if per_pair is not None else int(args.session_items)
    queue = build_session_queue(
        units=units,
        provider_names=compare_providers,
        strategy=args.strategy,
        seed=args.seed,
        session_items=int(args.session_items),
        per_pair_sample_size=per_pair,
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose,
        unique_recordings=unique_n,
    )
    if len(queue) != expected:
        raise ValueError(f"Session queue has {len(queue)} items, expected {expected}.")
    out = manifest_path or Path(session["session_manifest"])
    save_session_manifest(
        out,
        build_manifest_payload(
            queue,
            strategy=args.strategy,
            seed=args.seed,
            session_items=int(args.session_items),
            ground_truth_name=args.ground_truth_name,
            compare_providers=compare_providers,
            include_ground_truth=bool(getattr(args, "include_ground_truth", True)),
            min_gt_words=int(args.min_gt_words or 0),
            min_audio_seconds=float(args.min_audio_seconds or 0.0),
            exclude_gt_markers=bool(getattr(args, "exclude_gt_markers", True)),
            unique_recordings=unique_n,
            recording_seed=int(args.recording_seed),
        ),
    )
    n_unique = len({u.recording_id for u, _, _ in queue})
    print(f"Wrote {len(queue)} items ({n_unique} unique recordings) → {out.resolve()}")
    return out.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead: sample fixed session queue and write session_manifest.json")
    parser.add_argument("--session-config", type=Path, default=DEFAULT_SESSION_CONFIG)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Override manifest output path (default: session_manifest field in session config)",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore cached unit pickles and rebuild")
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Copy items from an existing manifest (e.g. configs/ab_prefs.manifest.json) instead of resampling",
    )
    parser.add_argument(
        "--exclude-scrub",
        action="store_true",
        help="With --from-manifest: drop items whose GT span is SCRUB or should_scrub",
    )
    args = parser.parse_args()
    create_session_manifest(
        args.session_config,
        args.manifest_path,
        rebuild_cache=args.rebuild_cache,
        from_manifest=args.from_manifest,
        exclude_scrub=args.exclude_scrub,
    )


if __name__ == "__main__":
    main()
