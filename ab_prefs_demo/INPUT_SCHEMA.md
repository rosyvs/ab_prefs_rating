# A/B Preference Demo Input Schema

This module builds comparison units from three sources: ground-truth JSONL transcripts, provider JSON files, and local audio files.

## Required Inputs

### 1) Ground-truth transcript directory
- Directory with files named `{recording_id}.jsonl`
- Each line is one JSON object with at least:
  - `start_seconds` (float)
  - `end_seconds` (float)
  - `orthographic_text` (str)
- Optional fields retained in metadata:
  - `tier`, `should_scrub`, and any other keys

### 2) Provider outputs
- Configured as a mapping: `provider_name -> provider_json_directory`
- Each provider directory contains `{recording_id}.json`
- Supported JSON shapes:
  - Wrapped AssemblyAI-like:
    - `{ "response": { "words": [...], "utterances": [...] } }`
  - Raw AssemblyAI-like:
    - `{ "words": [...], "utterances": [...] }`
  - Segment-style provider outputs:
    - `{ "segments": [ { "start": ..., "end": ..., "text": ..., "words": [...] } ] }`

Word entries are expected to include:
- `text` (or `word` / `punctuated_word`)
- `start`, `end`
- Optional: `speaker`

Start/end may be in milliseconds or seconds. The loader infers milliseconds if magnitudes exceed typical audio lengths.

### 3) Audio directory
- Directory containing `{recording_id}.mp3`
- The demo embeds the full file and seeks to GT segment start/end for each item.

## Built Comparison Unit

One unit corresponds to one GT segment for one recording:
- `recording_id`
- `segment_index`
- `start_seconds`, `end_seconds`
- `ground_truth_text`
- `audio_path`
- `provider_candidates` map with:
  - transcript text for this span
  - per-word timings when available (for word highlighting)
  - sampling metrics vs GT (`wer`, `deletions`, etc.)

## Sampling Features

For each unit, features across configured providers:
- `avg_wer`
- `wer_spread` (max-min WER)
- `avg_deletions`

Used by selection strategies:
- `random`
- `max_discrepancy` (`wer_spread`)
- `max_wer` (`avg_wer`)
- `most_deletions` (`avg_deletions`)
