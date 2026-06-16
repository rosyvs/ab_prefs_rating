from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class WordToken:
    text: str
    start_seconds: float
    end_seconds: float
    speaker: str | None = None


@dataclass(slots=True)
class ProviderCandidate:
    provider_name: str
    text: str
    words: list[WordToken]
    source_type: str  # "words" | "segments" | "ground_truth"
    segment_rows: list[dict] = field(default_factory=list)  # start_seconds, end_seconds, text, speaker?
    wer_vs_gt: float | None = None
    deletions_vs_gt: int | None = None
    substitutions_vs_gt: int | None = None
    insertions_vs_gt: int | None = None

    def has_transcript(self) -> bool:
        if self.text.strip():
            return True
        if self.words:
            return True
        if self.segment_rows:
            return True
        return False


@dataclass(slots=True)
class ComparisonUnit:
    recording_id: str
    segment_index: int  # first raw GT segment index in this unit
    start_seconds: float
    end_seconds: float
    ground_truth_text: str
    audio_path: Path
    segment_index_end: int | None = None  # last raw GT segment when multiple were merged
    provider_candidates: dict[str, ProviderCandidate] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

    @property
    def span_key(self) -> str:
        end_ix = self.segment_index if self.segment_index_end is None else self.segment_index_end
        return f"{self.recording_id}:{self.segment_index}-{end_ix}:{self.start_seconds:.3f}-{self.end_seconds:.3f}"


@dataclass(slots=True)
class PreferenceRecord:
    session_id: str
    timestamp_utc: str
    strategy: str
    recording_id: str
    segment_index: int
    start_seconds: float
    end_seconds: float
    provider_a: str
    provider_b: str
    choice: str
    note: str
    ground_truth_text: str
    transcript_a: str
    transcript_b: str
    rating_mode: str = "overall"  # "overall" | "multi_dimension"
    choice_text: str = ""
    choice_timing: str = ""
    choice_diarization: str = ""
    choice_punctuation: str = ""
