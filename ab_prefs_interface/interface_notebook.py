from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import ipywidgets as widgets
from IPython.display import display

from ab_prefs_interface.audio_clips import ensure_queue_clips
from ab_prefs_interface.data_model import ComparisonUnit, PreferenceRecord, ProviderCandidate
from ab_prefs_interface.dimension_ui import (
    DIMENSION_CHOICES,
    DIMENSION_LABELS,
    DIMENSIONS,
    all_dimensions_selected,
    dimension_button_label,
    empty_dimension_picks,
)
from ab_prefs_interface.storage_json import append_record

rating_style = """
<style>
.ab-wrap{margin:10px 0 18px 0;}
.ab-cols{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;width:100%;}
.ab-col{flex:1 1 360px;min-width:320px;}
.ab-panel{max-height:280px;overflow-y:auto;font-size:14px;line-height:1.5;border:1px solid #d1d5db;border-radius:6px;padding:8px;}
.ab-seg.current{background:#fde68a;box-shadow:0 0 0 1px #f59e0b inset;}
.ab-seg.partial{color:#6b7280;background:#f3f4f6;}
.ab-seg.partial.current{background:#e5e7eb;color:#374151;box-shadow:0 0 0 1px #9ca3af inset;}
.ab-partial-note{font-size:12px;color:#6b7280;margin:0 0 8px 0;font-style:italic;}
.ab-spk{font-weight:600;opacity:.8;}
.ab-meta{font-size:12px;color:#4b5563;margin:6px 0 10px 0;}
</style>
"""

# Highlight + scroll only when the active segment index changes (not every ontimeupdate tick).
ontimeupdate_dual = (
    "var root=this.closest('.ab-wrap'),t=this.currentTime,stop=parseFloat(this.dataset.stopAt);"
    "if(!isNaN(stop)&&t>=stop){this.pause();this.currentTime=0;return;}"
    "if(!root)return;"
    "root.querySelectorAll('.ab-panel').forEach(function(col){"
    "var segs=col.querySelectorAll('.ab-seg'),bestIdx=-1,bestA=-1;"
    "for(var i=0;i<segs.length;i++){var s=segs[i],a=parseFloat(s.dataset.start),b=parseFloat(s.dataset.end);"
    "if(isNaN(a)||isNaN(b))continue;"
    "var cur=s.classList.contains('excl-end')?(t>=a&&t<b):(t>=a&&t<=b);"
    "if(cur&&a>=bestA){bestA=a;bestIdx=i;}}"
    "if(String(col.dataset.abLastIdx)===String(bestIdx))return;"
    "col.dataset.abLastIdx=bestIdx;"
    "for(var j=0;j<segs.length;j++)segs[j].classList.toggle('current',j===bestIdx);"
    "if(bestIdx>=0){var best=segs[bestIdx],R=col.getBoundingClientRect(),B=best.getBoundingClientRect();"
    "col.scrollTop+=((B.top+B.height/2)-(R.top+R.height/2));}"
    "});"
)


def segment_rows_html(
    segment_rows: list[dict],
    *,
    time_offset: float = 0.0,
    clip_duration: float | None = None,
) -> str:
    """Phrase/segment rows; grey + note when utterance only partly inside GT clip."""
    if not segment_rows:
        return '<div class="ab-seg">(no overlapping transcript)</div>'
    click_js = (
        "var a=this.closest('.ab-wrap').querySelector('audio');"
        "if(a){a.currentTime=Math.max(0,parseFloat(this.dataset.start));a.play();}"
    )
    parts: list[str] = []
    has_partial = False
    rows = sorted(segment_rows, key=lambda row: (row["start_seconds"], row["end_seconds"]))
    for row in rows:
        body = html.escape((row.get("text") or "").strip())
        speaker = row.get("speaker")
        if speaker is not None:
            body = f"<strong>{html.escape(str(speaker))}:</strong> " + body
        start = float(row["start_seconds"]) - time_offset
        end = float(row["end_seconds"]) - time_offset
        partial = bool(row.get("partial_overlap"))
        if clip_duration is not None:
            partial = partial or start < 0.0 or end > clip_duration
            start = max(0.0, start)
            end = min(end, clip_duration)
        if partial:
            has_partial = True
        cls = "ab-seg partial" if partial else "ab-seg"
        parts.append(
            f'<div class="{cls}" data-start="{start}" data-end="{end}" '
            f'onclick="{click_js}">{body}</div>'
        )
    if has_partial:
        parts.insert(
            0,
            '<div class="ab-partial-note">This ASR uses segment-level timestamps (not word-level). '
            'Grey utterances partly overlap this GT clip and may extend before/after the audio.</div>',
        )
    return "\n".join(parts)


def words_html(candidate: ProviderCandidate, *, time_offset: float = 0.0, clip_duration: float | None = None) -> str:
    if candidate.words:
        parts: list[str] = []
        current_speaker: str | None = None
        for word in candidate.words:
            if word.speaker and word.speaker != current_speaker:
                parts.append("<br>" if current_speaker is not None else "")
                parts.append(f'<span class="ab-spk">{html.escape(word.speaker)}:</span> ')
                current_speaker = word.speaker
            start = word.start_seconds - time_offset
            end = word.end_seconds - time_offset
            parts.append(
                f'<span class="ab-seg excl-end" data-start="{start}" data-end="{end}">'
                f"{html.escape(word.text)}</span> "
            )
        return "".join(parts).strip()
    if candidate.segment_rows:
        return segment_rows_html(candidate.segment_rows, time_offset=time_offset, clip_duration=clip_duration)
    if candidate.text:
        return f'<div class="ab-seg">{html.escape(candidate.text)}</div>'
    return '<div class="ab-seg">(no overlapping transcript)</div>'


def comparison_block(
    unit: ComparisonUnit,
    provider_a: str,
    provider_b: str,
    *,
    audio_url: str,
    clip_duration: float,
    time_offset: float,
    show_providers: bool = False,
    item_label: str | None = None,
) -> str:
    html_a = words_html(unit.provider_candidates[provider_a], time_offset=time_offset, clip_duration=clip_duration)
    html_b = words_html(unit.provider_candidates[provider_b], time_offset=time_offset, clip_duration=clip_duration)
    start = unit.start_seconds
    end = unit.end_seconds
    if show_providers:
        if unit.segment_index_end is not None:
            seg_label = f"GT segments {unit.segment_index}–{unit.segment_index_end} (merged)"
        else:
            seg_label = f"GT segment {unit.segment_index}"
        meta = (
            f"Recording {html.escape(unit.recording_id)} · {seg_label} · "
            f"Span {start:.2f}s to {end:.2f}s"
        )
        label_a = f"A: {html.escape(provider_a)}"
        label_b = f"B: {html.escape(provider_b)}"
    else:
        meta = f"Span {start:.2f}s to {end:.2f}s"
        label_a = "A"
        label_b = "B"
    ontime = html.escape(ontimeupdate_dual, quote=True)
    audio_src = html.escape(audio_url, quote=True)
    title_html = f"<h3 style=\"margin:0 0 10px 0;\">{html.escape(item_label)}</h3>" if item_label else ""
    return f"""
{title_html}
<div class="ab-wrap">
  <div class="ab-meta">{meta}</div>
  <audio controls style="width:100%;max-width:980px;margin-bottom:10px;"
    data-stop-at="{clip_duration:.3f}"
    ontimeupdate="{ontime}">
    <source src="{audio_src}" type="audio/mpeg">
  </audio>
  <div class="ab-cols">
    <div class="ab-col">
      <div><strong>{label_a}</strong></div>
      <div class="ab-panel">{html_a}</div>
    </div>
    <div class="ab-col">
      <div><strong>{label_b}</strong></div>
      <div class="ab-panel">{html_b}</div>
    </div>
  </div>
</div>
"""


class NotebookPreferenceInterface:
    def __init__(
        self,
        queue: list[tuple[ComparisonUnit, str, str]],
        output_json_path: Path,
        strategy: str,
        session_id: str | None = None,
        show_note: bool = True,
        show_providers: bool = False,
        clip_dir: Path | None = None,
        notebook_root: Path | None = None,
        verbose: bool = False,
        ground_truth_name: str = "ground_truth",
        rating_mode: str = "overall",
    ) -> None:
        if not queue:
            raise ValueError("Queue is empty; nothing to review.")
        self.queue = queue
        self.output_json_path = output_json_path
        self.strategy = strategy
        self.session_id = session_id or uuid4().hex[:12]
        self.show_providers = show_providers
        self.ground_truth_name = ground_truth_name
        self.rating_mode = rating_mode
        self.dimension_picks = empty_dimension_picks()
        self.current_index = 0
        clip_root = clip_dir or Path("results/ab_prefs/audio_clips")
        nb_root = notebook_root or Path.cwd()
        self.clip_dir = clip_root
        self.notebook_root = nb_root
        self.verbose = verbose
        self.audio_urls: dict[str, str] = {}
        self.show_note = show_note
        self.note_widget = widgets.Textarea(
            value="",
            placeholder="Optional note about why A/B/tie",
            description="Note:",
            layout=widgets.Layout(width="900px", height="70px", display="none"),
        )
        self.note_toggle = widgets.Checkbox(value=False, description="Add note")
        self.note_toggle.observe(self.on_note_toggle, names="value")
        self.style_html = widgets.HTML(value=rating_style)
        self.item_html = widgets.HTML(value="")
        self.status_html = widgets.HTML(value="")
        if rating_mode == "multi_dimension":
            self.dimension_buttons: dict[str, dict[str, widgets.Button]] = {}
            dimension_rows: list[widgets.Widget] = []
            for dim in DIMENSIONS:
                row_buttons: list[widgets.Widget] = [widgets.HTML(value=f"<strong>{DIMENSION_LABELS[dim]}</strong>")]
                self.dimension_buttons[dim] = {}
                for choice in DIMENSION_CHOICES:
                    btn = widgets.Button(description=dimension_button_label(choice))
                    btn.on_click(lambda _, d=dim, c=choice: self.set_dimension_pick(d, c))
                    row_buttons.append(btn)
                    self.dimension_buttons[dim][choice] = btn
                dimension_rows.append(widgets.HBox(row_buttons))
            self.button_next = widgets.Button(description="Next item", disabled=True, button_style="primary")
            self.button_next.on_click(lambda _: self.submit_dimension_record())
            choice_buttons: list[widgets.Widget] = dimension_rows + [self.button_next]
        else:
            self.button_a = widgets.Button(description="Choose A", button_style="success")
            self.button_b = widgets.Button(description="Choose B", button_style="success")
            self.button_tie = widgets.Button(description="Tie")
            self.button_a.on_click(lambda _: self.submit_choice("A"))
            self.button_b.on_click(lambda _: self.submit_choice("B"))
            self.button_tie.on_click(lambda _: self.submit_choice("tie"))
            choice_buttons = [self.button_a, self.button_b, self.button_tie]
        if show_note:
            choice_buttons.append(self.note_toggle)
        self.button_row = widgets.VBox(choice_buttons) if rating_mode == "multi_dimension" else widgets.HBox(choice_buttons)
        root_children = [self.style_html, self.item_html, self.button_row]
        if show_note:
            root_children.append(self.note_widget)
        root_children.append(self.status_html)
        self.root = widgets.VBox(root_children)
        self.shown = False
        self.set_placeholder(f"Preparing audio clips ({len(queue)} items)…")
        self.button_row.layout.display = "none"

    def set_placeholder(self, message: str) -> None:
        self.item_html.value = f'<p style="margin:8px 0;color:#4b5563;">{html.escape(message)}</p>'

    def load_clips(self) -> None:
        if self.audio_urls:
            return
        self.set_placeholder(f"Preparing audio clips ({len(self.queue)} items)…")
        self.audio_urls = ensure_queue_clips(
            self.queue,
            self.clip_dir,
            self.notebook_root,
            verbose=self.verbose,
        )
        self.button_row.layout.display = None
        if self.rating_mode == "multi_dimension":
            self.reset_dimension_picks()
        self.render_current()
        self.show(force=True)

    def current_item(self) -> tuple[ComparisonUnit, str, str]:
        return self.queue[self.current_index]

    def on_note_toggle(self, change: dict) -> None:
        if change.get("name") != "value":
            return
        if change["new"]:
            self.note_widget.layout.display = None
        else:
            self.note_widget.value = ""
            self.note_widget.layout.display = "none"

    def show(self, *, force: bool = False) -> None:
        if force or not self.shown:
            display(self.root)
            self.shown = True
        if self.audio_urls:
            self.render_current()

    def item_html_value(self) -> str:
        unit, provider_a, provider_b = self.current_item()
        audio_url = self.audio_urls[unit.span_key]
        clip_duration = unit.end_seconds - unit.start_seconds
        return comparison_block(
            unit=unit,
            provider_a=provider_a,
            provider_b=provider_b,
            audio_url=audio_url,
            clip_duration=clip_duration,
            time_offset=unit.start_seconds,
            show_providers=self.show_providers,
            item_label=f"Preference item {self.current_index + 1}/{len(self.queue)}",
        )

    def render_current(self) -> None:
        if not self.audio_urls:
            return
        self.item_html.value = self.item_html_value()

    def reset_dimension_picks(self) -> None:
        self.dimension_picks = empty_dimension_picks()
        if self.rating_mode != "multi_dimension":
            return
        for dim in DIMENSIONS:
            for btn in self.dimension_buttons[dim].values():
                btn.button_style = ""
        self.button_next.disabled = True

    def set_dimension_pick(self, dimension: str, choice: str) -> None:
        self.dimension_picks[dimension] = choice
        for c, btn in self.dimension_buttons[dimension].items():
            btn.button_style = "success" if c == choice else ""
        self.button_next.disabled = not all_dimensions_selected(self.dimension_picks)

    def show_complete(self) -> None:
        from ab_prefs_interface.summarize_preferences import summarize_completion_html

        self.item_html.value = summarize_completion_html(
            self.output_json_path, self.ground_truth_name, self.rating_mode
        )
        self.button_row.layout.display = "none"

    def submit_dimension_record(self) -> None:
        if not all_dimensions_selected(self.dimension_picks):
            return
        unit, provider_a, provider_b = self.current_item()
        note = self.note_widget.value.strip() if self.show_note else ""
        record = PreferenceRecord(
            session_id=self.session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            strategy=self.strategy,
            recording_id=unit.recording_id,
            segment_index=unit.segment_index,
            start_seconds=unit.start_seconds,
            end_seconds=unit.end_seconds,
            provider_a=provider_a,
            provider_b=provider_b,
            choice="",
            note=note,
            ground_truth_text=unit.ground_truth_text,
            transcript_a=unit.provider_candidates[provider_a].text,
            transcript_b=unit.provider_candidates[provider_b].text,
            rating_mode="multi_dimension",
            choice_text=str(self.dimension_picks["text"]),
            choice_timing=str(self.dimension_picks["timing"]),
            choice_diarization=str(self.dimension_picks["diarization"]),
        )
        append_record(self.output_json_path, record)
        self.note_widget.value = ""
        self.current_index += 1
        saved = (
            f"Saved: text={self.dimension_picks['text']}, timing={self.dimension_picks['timing']}, "
            f"diarization={self.dimension_picks['diarization']}"
        )
        self.status_html.value = f'<p style="margin:8px 0;color:#4b5563;">{html.escape(saved)}</p>'
        if self.current_index >= len(self.queue):
            self.show_complete()
            return
        self.reset_dimension_picks()
        self.render_current()

    def submit_choice(self, choice: str) -> None:
        unit, provider_a, provider_b = self.current_item()
        note = self.note_widget.value.strip() if self.show_note else ""
        record = PreferenceRecord(
            session_id=self.session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            strategy=self.strategy,
            recording_id=unit.recording_id,
            segment_index=unit.segment_index,
            start_seconds=unit.start_seconds,
            end_seconds=unit.end_seconds,
            provider_a=provider_a,
            provider_b=provider_b,
            choice=choice,
            note=note,
            ground_truth_text=unit.ground_truth_text,
            transcript_a=unit.provider_candidates[provider_a].text,
            transcript_b=unit.provider_candidates[provider_b].text,
        )
        append_record(self.output_json_path, record)
        self.note_widget.value = ""
        self.current_index += 1
        if self.show_providers:
            saved = f"Saved: {choice} for {unit.span_key} ({provider_a} vs {provider_b})"
        else:
            saved = f"Saved: {choice}"
        self.status_html.value = f'<p style="margin:8px 0;color:#4b5563;">{html.escape(saved)}</p>'
        if self.current_index >= len(self.queue):
            self.show_complete()
            return
        if self.rating_mode == "multi_dimension":
            self.reset_dimension_picks()
        self.render_current()
