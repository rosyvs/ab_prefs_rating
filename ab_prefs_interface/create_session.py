"""Lead: build session_manifest.json from configs/ab_prefs.session.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from ab_prefs_interface.launch import DEFAULT_SESSION_CONFIG, load_session_config, namespace_for_rater, setup_repo_path
from ab_prefs_interface.session_config import compare_provider_names
from ab_prefs_interface.matching import build_comparison_units
from ab_prefs_interface.run_demo import load_provider_dirs
from ab_prefs_interface.sampling import build_session_queue, pair_combinations
from ab_prefs_interface.scoring import score_units
from ab_prefs_interface.session_manifest import build_manifest_payload, save_session_manifest


def create_session_manifest(session_config_path: Path, manifest_path: Path | None = None) -> Path:
    session = load_session_config(session_config_path)
    setup_repo_path(Path(session["notebook_root"]))
    args = namespace_for_rater(session, rater_id="_lead_export_")
    provider_dirs = load_provider_dirs(args)
    demo_n = int(args.demo_recordings or 0)
    units = build_comparison_units(
        gt_dir=args.gt_dir.expanduser().resolve(),
        provider_dirs=provider_dirs,
        audio_dir=args.audio_dir.expanduser().resolve(),
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose,
        cache_dir=args.cache_dir.expanduser().resolve(),
        rebuild_cache=bool(args.rebuild_cache),
        demo_recordings=demo_n if demo_n > 0 else None,
        demo_seed=int(args.demo_seed),
        min_gt_words=int(args.min_gt_words or 0),
        min_audio_seconds=float(args.min_audio_seconds or 0.0),
    )
    score_units(
        units=units,
        provider_names=sorted(provider_dirs.keys()),
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
            demo_recordings=demo_n if demo_n > 0 else None,
            demo_seed=int(args.demo_seed),
        ),
    )
    print(f"Wrote {len(queue)} items → {out.resolve()}")
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
    args = parser.parse_args()
    create_session_manifest(args.session_config, args.manifest_path)


if __name__ == "__main__":
    main()
