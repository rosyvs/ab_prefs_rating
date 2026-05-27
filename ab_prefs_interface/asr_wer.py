"""DD210 lexical WER for ab_prefs sampling — analyze_multiple_asr policy, ASR JSON from results/."""
from __future__ import annotations

import sys
from pathlib import Path

from ab_prefs_interface.matching import load_provider_payload


def import_analyze_multiple_asr(asr_eval_root: Path | str):
    root = Path(asr_eval_root).expanduser().resolve()
    if not (root / "analyze_multiple_asr.py").is_file():
        raise FileNotFoundError(f"analyze_multiple_asr.py not found under asr_eval_root={root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import analyze_multiple_asr as am

    return am


class RecordingWerCache:
    """Per (recording_id, provider): segment-level lex WER rows from full-recording jiwer alignment."""

    def __init__(self, asr_eval_root: Path | str, gt_segments_by_recording: dict[str, list[dict]], provider_dirs: dict[str, Path]):
        self.asr_eval_root = Path(asr_eval_root).expanduser().resolve()
        self.gt_segments_by_recording = gt_segments_by_recording
        self.provider_dirs = {k: Path(v).expanduser().resolve() for k, v in provider_dirs.items()}
        self.am = import_analyze_multiple_asr(self.asr_eval_root)
        self._rows: dict[tuple[str, str], list[dict]] = {}

    def segment_rows(self, recording_id: str, provider_name: str) -> list[dict]:
        key = (recording_id, provider_name)
        if key in self._rows:
            return self._rows[key]
        segments = self.gt_segments_by_recording[recording_id]
        provider_dir = self.provider_dirs[provider_name]
        wrapped = load_provider_payload(provider_dir, recording_id)
        rows = self.am.align_recording_lex_by_gt_segment(segments, wrapped)
        self._rows[key] = rows
        return rows

    def clip_wer(self, recording_id: str, provider_name: str, start_seconds: float, end_seconds: float) -> dict:
        rows = self.segment_rows(recording_id, provider_name)
        return self.am.clip_lex_wer_from_segment_rows(rows, start_seconds, end_seconds)
