from __future__ import annotations

import bisect
import json
import random
from pathlib import Path

from tqdm import tqdm

from ab_prefs_demo.data_model import ComparisonUnit, ProviderCandidate, WordToken
from ab_prefs_demo.unit_cache import (
    build_cache_key,
    demo_cache_root,
    load_recording_units,
    recording_cache_path,
    save_recording_units,
)


def words_aai_from_stored_deepgram_raw(raw: dict) -> list[dict]:
    """Build response.words (text + start/end ms) from saved Deepgram raw dict."""
    out: list[dict] = []
    if not isinstance(raw, dict):
        return out
    res = raw.get("results")
    if not isinstance(res, dict):
        return out
    chans = res.get("channels")
    if not isinstance(chans, list) or not chans:
        return out
    alts = chans[0].get("alternatives") if isinstance(chans[0], dict) else None
    if not isinstance(alts, list) or not alts:
        return out
    words = alts[0].get("words") if isinstance(alts[0], dict) else None
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, dict):
            continue
        pw = (w.get("punctuated_word") or w.get("word") or "").strip()
        if not pw:
            continue
        st, en = w.get("start"), w.get("end")
        if st is None or en is None:
            continue
        out.append({"text": pw, "start": float(st) * 1000.0, "end": float(en) * 1000.0, "speaker": w.get("speaker")})
    return out


def gt_word_count(orthographic_text: str) -> int:
    return len((orthographic_text or "").split())


def chunk_span_seconds(chunk: list[dict]) -> float:
    return float(chunk[-1]["end_seconds"]) - float(chunk[0]["start_seconds"])


def merge_gt_segments(
    segments: list[dict],
    *,
    min_gt_words: int = 0,
    min_audio_seconds: float = 0.0,
) -> list[dict]:
    """Merge consecutive GT lines until word-count and/or audio span thresholds are met."""
    if min_gt_words <= 1 and min_audio_seconds <= 0:
        return [
            {
                **segment,
                "segment_index_start": index,
                "segment_index_end": index,
            }
            for index, segment in enumerate(segments)
        ]
    merged: list[dict] = []
    index = 0
    while index < len(segments):
        chunk = [segments[index]]
        text_parts = [str(segments[index].get("orthographic_text") or "")]
        end_index = index
        while end_index + 1 < len(segments):
            merged_text = " ".join(text_parts)
            word_count = gt_word_count(merged_text)
            duration = chunk_span_seconds(chunk)
            words_ok = min_gt_words <= 1 or word_count >= min_gt_words
            duration_ok = min_audio_seconds <= 0 or duration >= min_audio_seconds
            if words_ok and duration_ok:
                break
            end_index += 1
            chunk.append(segments[end_index])
            text_parts.append(str(segments[end_index].get("orthographic_text") or ""))
        merged.append(
            {
                "start_seconds": float(chunk[0]["start_seconds"]),
                "end_seconds": float(chunk[-1]["end_seconds"]),
                "orthographic_text": " ".join(part for part in text_parts if part).strip(),
                "segment_index_start": index,
                "segment_index_end": end_index,
            }
        )
        index = end_index + 1
    return merged


def select_demo_recording_ids(transcripts: dict[str, list[dict]], n: int, seed: int) -> list[str]:
    recording_ids = sorted(transcripts.keys())
    if n >= len(recording_ids):
        return recording_ids
    return sorted(random.Random(seed).sample(recording_ids, n))


def parse_jsonl_transcript(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_ground_truth_transcripts(gt_dir: Path) -> dict[str, list[dict]]:
    files = sorted(gt_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No GT jsonl files under {gt_dir}")
    return {path.stem: parse_jsonl_transcript(path) for path in files}


def load_provider_payload(provider_dir: Path, recording_id: str) -> dict:
    payload_path = provider_dir / f"{recording_id}.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing provider JSON: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def response_dict(payload: dict) -> dict:
    candidate = payload.get("response", payload)
    if not isinstance(candidate, dict):
        raise ValueError("Provider payload must contain a dict response")
    return candidate


def normalize_word_entry(word: dict, default_speaker: str | None = None) -> tuple[str, float, float, str | None] | None:
    text = (word.get("text") or word.get("word") or word.get("punctuated_word") or "").strip()
    start_value = word.get("start")
    end_value = word.get("end")
    if not text or start_value is None or end_value is None:
        return None
    speaker = word.get("speaker")
    if speaker is None:
        speaker = default_speaker
    return text, float(start_value), float(end_value), None if speaker is None else str(speaker)


def detect_millisecond_scale(raw_values: list[float]) -> bool:
    if not raw_values:
        return False
    return max(raw_values) > 1_000.0


def word_tokens_from_second_entries(entries: list[dict]) -> list[WordToken]:
    """Word dicts with start/end already in seconds."""
    out: list[WordToken] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = (entry.get("text") or entry.get("word") or entry.get("punctuated_word") or "").strip()
        start_value, end_value = entry.get("start"), entry.get("end")
        if not text or start_value is None or end_value is None:
            continue
        speaker = entry.get("speaker")
        out.append(
            WordToken(
                text=text,
                start_seconds=float(start_value),
                end_seconds=float(end_value),
                speaker=None if speaker is None else str(speaker),
            )
        )
    out.sort(key=lambda token: (token.start_seconds, token.end_seconds))
    return out


def extract_azure_words_seconds(response: dict) -> list[WordToken]:
    """Azure fast/LLM: word times under segments[].words are seconds (even on long recordings)."""
    entries: list[dict] = []
    for segment in response.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        default_speaker = segment.get("speaker")
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            entry = dict(word)
            if entry.get("speaker") is None and default_speaker is not None:
                entry["speaker"] = default_speaker
            entries.append(entry)
    return word_tokens_from_second_entries(entries)


def extract_aai_style_words_seconds(response: dict) -> list[WordToken]:
    """AssemblyAI-style top-level or utterance words with start/end in ms."""
    entries: list[dict] = []
    top_words = response.get("words")
    if isinstance(top_words, list) and top_words:
        entries.extend(w for w in top_words if isinstance(w, dict))
    else:
        for utterance in response.get("utterances") or []:
            if not isinstance(utterance, dict):
                continue
            parent_speaker = utterance.get("speaker")
            for word in utterance.get("words") or []:
                if not isinstance(word, dict):
                    continue
                entry = dict(word)
                if entry.get("speaker") is None and parent_speaker is not None:
                    entry["speaker"] = parent_speaker
                entries.append(entry)
    return word_tokens_from_ms_entries(entries)


def word_tokens_from_ms_entries(entries: list[dict]) -> list[WordToken]:
    """AAI-shaped word dicts with start/end in ms → WordToken list in seconds."""
    out: list[WordToken] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = (entry.get("text") or entry.get("word") or entry.get("punctuated_word") or "").strip()
        start_value, end_value = entry.get("start"), entry.get("end")
        if not text or start_value is None or end_value is None:
            continue
        speaker = entry.get("speaker")
        out.append(
            WordToken(
                text=text,
                start_seconds=float(start_value) / 1000.0,
                end_seconds=float(end_value) / 1000.0,
                speaker=None if speaker is None else str(speaker),
            )
        )
    out.sort(key=lambda token: (token.start_seconds, token.end_seconds))
    return out


def extract_deepgram_words_seconds(response: dict) -> list[WordToken]:
    """Deepgram: ``response.words`` (ms, AAI-shaped) or rebuild from ``response.raw`` — same as analyze_multiple_asr."""
    words_top = response.get("words")
    if isinstance(words_top, list) and words_top:
        tokens = word_tokens_from_ms_entries(words_top)
        if tokens:
            return tokens
    raw = response.get("raw")
    if isinstance(raw, dict):
        return word_tokens_from_ms_entries(words_aai_from_stored_deepgram_raw(raw))
    return []


def extract_words_seconds(payload: dict) -> list[WordToken]:
    response = response_dict(payload)
    provider = response.get("provider")
    if provider == "deepgram":
        deepgram_words = extract_deepgram_words_seconds(response)
        if deepgram_words:
            return deepgram_words
    if provider == "azure_speech":
        azure_words = extract_azure_words_seconds(response)
        if azure_words:
            return azure_words
    aai_words = extract_aai_style_words_seconds(response)
    if aai_words:
        return aai_words
    # rare: word arrays nested under segments (non-Azure); treat timestamps as seconds
    segment_word_entries: list[dict] = []
    for segment in response.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        default_speaker = segment.get("speaker")
        for word in segment.get("words") or []:
            if not isinstance(word, dict):
                continue
            entry = dict(word)
            if entry.get("speaker") is None and default_speaker is not None:
                entry["speaker"] = default_speaker
            segment_word_entries.append(entry)
    return word_tokens_from_second_entries(segment_word_entries)


def extract_segments_seconds(payload: dict) -> list[dict]:
    """Segment/utterance spans in seconds (utterances use ms when magnitudes are large)."""
    return extract_timed_spans_seconds(payload)


def extract_timed_spans_seconds(payload: dict) -> list[dict]:
    response = response_dict(payload)
    out: list[dict] = []
    segments = response.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start_value = segment.get("start")
            end_value = segment.get("end")
            if start_value is None or end_value is None:
                continue
            out.append(
                {
                    "start_seconds": float(start_value),
                    "end_seconds": float(end_value),
                    "text": str(segment.get("text") or "").strip(),
                    "speaker": segment.get("speaker"),
                }
            )
    utterances = response.get("utterances") or []
    utterance_times: list[float] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        for key in ("start", "end"):
            value = utterance.get(key)
            if value is not None:
                utterance_times.append(float(value))
    utterance_scale = 0.001 if detect_millisecond_scale(utterance_times) else 1.0
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        start_value = utterance.get("start")
        end_value = utterance.get("end")
        text = str(utterance.get("text") or "").strip()
        if start_value is None or end_value is None or not text:
            continue
        out.append(
            {
                "start_seconds": float(start_value) * utterance_scale,
                "end_seconds": float(end_value) * utterance_scale,
                "text": text,
                "speaker": utterance.get("speaker"),
            }
        )
    out.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
    return out


def spans_overlapping_gt(
    spans: list[dict],
    span_start: float,
    span_end: float,
    *,
    min_overlap_seconds: float = 1.0,
) -> list[dict]:
    """Overlapping phrase/segment rows in chronological order."""
    rows: list[dict] = []
    for span in spans:
        overlap = overlap_seconds(span["start_seconds"], span["end_seconds"], span_start, span_end)
        if overlap < min_overlap_seconds:
            continue
        row = dict(span)
        row["partial_overlap"] = span["start_seconds"] < span_start or span["end_seconds"] > span_end
        rows.append(row)
    rows.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
    return rows


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def words_overlapping_span(words: list[WordToken], span_start: float, span_end: float) -> list[WordToken]:
    """Words with time overlap vs [span_start, span_end]; words must be sorted by start_seconds."""
    if not words:
        return []
    starts = [word.start_seconds for word in words]
    # word.start < span_end  →  index of first word with start >= span_end
    end_index = bisect.bisect_left(starts, span_end)
    return [word for word in words[:end_index] if word.end_seconds > span_start]


def timed_words_from_text(text: str, start_seconds: float, end_seconds: float, speaker: str | None = None) -> list[WordToken]:
    tokens = [tok for tok in text.split() if tok.strip()]
    if not tokens:
        return []
    duration = max(0.001, end_seconds - start_seconds)
    step = duration / len(tokens)
    words: list[WordToken] = []
    for index, token in enumerate(tokens):
        token_start = start_seconds + index * step
        token_end = start_seconds + (index + 1) * step
        words.append(WordToken(text=token, start_seconds=token_start, end_seconds=token_end, speaker=speaker))
    return words


def provider_candidate_for_span(
    provider_name: str,
    span_start: float,
    span_end: float,
    *,
    payload: dict | None = None,
    recording_words: list[WordToken] | None = None,
    recording_segments: list[dict] | None = None,
) -> ProviderCandidate:
    if recording_words is None:
        if payload is None:
            raise ValueError("provider_candidate_for_span needs payload or recording_words")
        words = extract_words_seconds(payload)
    else:
        words = recording_words
    overlapped_words = words_overlapping_span(words, span_start, span_end)
    if overlapped_words:
        text = " ".join(word.text for word in overlapped_words)
        return ProviderCandidate(
            provider_name=provider_name,
            text=text,
            words=overlapped_words,
            source_type="words",
        )

    # Word-level provider but no words timed inside this GT clip → show nothing (don't fall back to segment text)
    if words:
        return ProviderCandidate(
            provider_name=provider_name,
            text="",
            words=[],
            source_type="words",
        )

    if recording_segments is None:
        if payload is None:
            raise ValueError("provider_candidate_for_span needs payload or recording_segments")
        timed_spans = extract_timed_spans_seconds(payload)
    else:
        timed_spans = recording_segments
    overlapped_spans = spans_overlapping_gt(timed_spans, span_start, span_end)
    text = " ".join(span["text"] for span in overlapped_spans if span["text"])
    return ProviderCandidate(
        provider_name=provider_name,
        text=text,
        words=[],
        segment_rows=overlapped_spans,
        source_type="segments",
    )


def ground_truth_candidate(ground_truth_name: str, text: str, span_start: float, span_end: float) -> ProviderCandidate:
    return ProviderCandidate(
        provider_name=ground_truth_name,
        text=text,
        words=timed_words_from_text(text=text, start_seconds=span_start, end_seconds=span_end),
        source_type="ground_truth",
    )


def build_units_for_recording(
    recording_id: str,
    segments: list[dict],
    audio_path: Path,
    provider_dirs: dict[str, Path],
    ground_truth_name: str,
    verbose: bool,
    min_gt_words: int = 0,
    min_audio_seconds: float = 0.0,
) -> list[ComparisonUnit]:
    """One recording: load ~N provider JSON files, align each GT span to provider text/words."""
    if min_gt_words > 1 or min_audio_seconds > 0:
        segments = merge_gt_segments(
            segments, min_gt_words=min_gt_words, min_audio_seconds=min_audio_seconds
        )
    payloads_by_provider = {
        provider_name: load_provider_payload(provider_dir, recording_id)
        for provider_name, provider_dir in provider_dirs.items()
    }
    words_by_provider = {
        provider_name: extract_words_seconds(payload)
        for provider_name, payload in payloads_by_provider.items()
    }
    segments_by_provider = {
        provider_name: extract_timed_spans_seconds(payload)
        for provider_name, payload in payloads_by_provider.items()
    }
    units: list[ComparisonUnit] = []
    seg_iter = tqdm(
        enumerate(segments),
        total=len(segments),
        desc=f"  segments {recording_id}",
        leave=False,
        disable=not verbose,
    )
    for _, segment in seg_iter:
        span_start = float(segment["start_seconds"])
        span_end = float(segment["end_seconds"])
        gt_text = str(segment.get("orthographic_text") or "")
        segment_index_start = int(segment["segment_index_start"])
        segment_index_end = int(segment["segment_index_end"])
        unit = ComparisonUnit(
            recording_id=recording_id,
            segment_index=segment_index_start,
            start_seconds=span_start,
            end_seconds=span_end,
            ground_truth_text=gt_text,
            audio_path=audio_path,
            segment_index_end=segment_index_end if segment_index_end != segment_index_start else None,
        )
        if segment_index_end != segment_index_start:
            unit.features["n_gt_segments_merged"] = float(segment_index_end - segment_index_start + 1)
            unit.features["gt_word_count"] = float(gt_word_count(gt_text))
        unit.provider_candidates[ground_truth_name] = ground_truth_candidate(
            ground_truth_name=ground_truth_name,
            text=gt_text,
            span_start=span_start,
            span_end=span_end,
        )
        for provider_name in payloads_by_provider:
            unit.provider_candidates[provider_name] = provider_candidate_for_span(
                provider_name=provider_name,
                span_start=span_start,
                span_end=span_end,
                recording_words=words_by_provider[provider_name],
                recording_segments=segments_by_provider[provider_name],
            )
        units.append(unit)
    return units


def build_comparison_units(
    gt_dir: Path,
    provider_dirs: dict[str, Path],
    audio_dir: Path,
    ground_truth_name: str = "ground_truth",
    verbose: bool = False,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    demo_recordings: int | None = None,
    demo_seed: int = 7,
    min_gt_words: int = 0,
    min_audio_seconds: float = 3.0,
) -> list[ComparisonUnit]:
    gt_dir = gt_dir.expanduser().resolve()
    audio_dir = audio_dir.expanduser().resolve()
    provider_dirs = {name: path.expanduser().resolve() for name, path in provider_dirs.items()}
    transcripts = load_ground_truth_transcripts(gt_dir)
    all_recording_count = len(transcripts)
    if demo_recordings is not None and demo_recordings > 0:
        picked = select_demo_recording_ids(transcripts, demo_recordings, demo_seed)
        transcripts = {recording_id: transcripts[recording_id] for recording_id in picked}
        if verbose:
            print(
                f"Demo mode: {len(picked)} / {all_recording_count} recordings "
                f"(seed={demo_seed}): {picked}"
            )
    if verbose:
        print(f"GT recordings: {len(transcripts)} · providers: {len(provider_dirs)}")
        if min_gt_words > 1 or min_audio_seconds > 0:
            print(
                f"merge rules: min_gt_words={min_gt_words}, min_audio_seconds={min_audio_seconds}"
            )
        est_segments = sum(len(segs) for segs in transcripts.values())
        print(f"~{est_segments} raw GT lines → fewer units after merge (load JSON + word overlap per unit×provider)")
    effective_cache_dir = None
    if cache_dir:
        effective_cache_dir = demo_cache_root(cache_dir) if demo_recordings else cache_dir.expanduser().resolve()
    cache_key = (
        build_cache_key(
            gt_dir,
            provider_dirs,
            audio_dir,
            ground_truth_name,
            demo_recordings=demo_recordings if demo_recordings else None,
            demo_seed=demo_seed if demo_recordings else None,
            min_gt_words=min_gt_words,
            min_audio_seconds=min_audio_seconds,
        )
        if effective_cache_dir
        else None
    )
    if verbose and cache_key:
        label = "demo cache" if demo_recordings else "unit cache"
        print(f"{label} key {cache_key} under {effective_cache_dir}")
    units: list[ComparisonUnit] = []
    rec_iter = tqdm(transcripts.items(), desc="build units (recordings)", disable=not verbose)
    for recording_id, segments in rec_iter:
        rec_iter.set_postfix(recording_id=recording_id, n_seg=len(segments))
        audio_path = audio_dir / f"{recording_id}.mp3"
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        cache_path = (
            recording_cache_path(effective_cache_dir, cache_key, recording_id)
            if effective_cache_dir and cache_key
            else None
        )
        if cache_path and cache_path.is_file() and not rebuild_cache:
            units.extend(load_recording_units(cache_path))
            continue
        recording_units = build_units_for_recording(
            recording_id=recording_id,
            segments=segments,
            audio_path=audio_path,
            provider_dirs=provider_dirs,
            ground_truth_name=ground_truth_name,
            verbose=verbose,
            min_gt_words=min_gt_words,
            min_audio_seconds=min_audio_seconds,
        )
        if cache_path:
            save_recording_units(cache_path, recording_units)
        units.extend(recording_units)
    if verbose:
        print(f"Built {len(units)} comparison units")
    return units
