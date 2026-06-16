"""Colab rating UI via IPython HTML + kernel callbacks (ipywidgets breaks when pre-imported)."""
from __future__ import annotations

import html
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from IPython.display import HTML, Javascript, clear_output, display

from ab_prefs_interface.audio_clips import ensure_queue_clips
from ab_prefs_interface.data_model import ComparisonUnit, PreferenceRecord
from ab_prefs_interface.dimension_ui import (
    DIMENSION_KEYS,
    DIMENSIONS,
    all_dimensions_selected,
    dimension_rows_html,
    empty_dimension_picks,
    parse_dimension_choice,
    parse_dimension_submit,
)
from ab_prefs_interface.interface_notebook import comparison_block, rating_style
from ab_prefs_interface.storage_json import append_record

CALLBACK_NAME = "ab_prefs_choice"


def _dim_key_map_js(active_dimensions: tuple[str, ...]) -> str:
    """Build a JS object literal mapping key → [dim, choice] for active dimensions only."""
    pairs = []
    for dim, choices in DIMENSION_KEYS.items():
        if dim not in active_dimensions:
            continue
        for choice, key in choices.items():
            pairs.append(f'"{key}": ["{dim}", "{choice}"]')
    return "{" + ", ".join(pairs) + "}"


class ColabHtmlPreferenceInterface:
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
        gcs_bucket: str | None = None,
        rating_dimensions: list[str] | None = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        if not queue:
            raise ValueError("Queue is empty; nothing to review.")
        self.queue = queue
        self.output_json_path = output_json_path
        self.strategy = strategy
        self.session_id = session_id or uuid4().hex[:12]
        self.show_note = show_note
        self.show_providers = show_providers
        self.ground_truth_name = ground_truth_name
        self.rating_mode = rating_mode
        self.active_dimensions: tuple[str, ...] = tuple(rating_dimensions) if rating_dimensions else DIMENSIONS
        self.dimension_picks = empty_dimension_picks(self.active_dimensions)
        self.debug = debug
        self.current_index = 0
        self.clip_dir = clip_dir or Path("results/ab_prefs/audio_clips")
        self.notebook_root = notebook_root or Path.cwd()
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket or ""
        self.audio_urls: dict[str, str] = {}
        self.status_message = ""
        self.callback_registered = False

    def current_item(self) -> tuple[ComparisonUnit, str, str]:
        return self.queue[self.current_index]

    def register_callback(self) -> None:
        if self.callback_registered:
            return
        from google.colab import output  # type: ignore

        output.register_callback(CALLBACK_NAME, self.on_choice)
        self.callback_registered = True

    def show(self, *, force: bool = False) -> None:
        self.register_callback()
        self.render(placeholder=not self.audio_urls)

    def load_clips(self) -> None:
        if self.audio_urls:
            return
        self.render(placeholder=True)
        self.audio_urls = ensure_queue_clips(
            self.queue,
            self.clip_dir,
            self.notebook_root,
            verbose=self.verbose,
        )
        if self.rating_mode == "multi_dimension":
            self.dimension_picks = empty_dimension_picks(self.active_dimensions)
        self.render()

    def choice_button(self, label: str, choice: str) -> str:
        return (
            f'<button type="button" data-ab-choice="{html.escape(choice)}" '
            f'style="margin:4px 8px 4px 0;padding:6px 14px;">{html.escape(label)}</button>'
        )

    def wire_page_js(self) -> None:
        if self.rating_mode == "multi_dimension":
            self.wire_multi_dimension_js()
            return
        display(Javascript(f"""
(function() {{
  if (!google.colab || !google.colab.kernel) {{
    console.error("google.colab.kernel not available");
    return;
  }}
  var invoke = google.colab.kernel.invokeFunction;
  var toggle = document.getElementById("ab-note-toggle");
  var field = document.getElementById("ab-note-field");
  if (toggle && field) {{
    toggle.checked = false;
    field.style.display = "none";
    toggle.onchange = function() {{
      field.style.display = toggle.checked ? "block" : "none";
      if (!toggle.checked) field.value = "";
    }};
  }}
  document.querySelectorAll("[data-ab-choice]").forEach(function(btn) {{
    btn.onclick = async function() {{
      var note = field ? field.value.trim() : "";
      await invoke("{CALLBACK_NAME}", [btn.getAttribute("data-ab-choice"), note], {{}});
    }};
  }});
  // Keep iframe focused so keyboard events land here after clicking audio play.
  document.body.tabIndex = -1;
  document.body.focus();
  var audio = document.querySelector("audio");
  if (audio) {{
    audio.addEventListener("play", function() {{ document.body.focus(); }});
  }}
  if (window._abKeyHandler) document.removeEventListener('keydown', window._abKeyHandler, true);
  window._abKeyHandler = function(e) {{
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (e.key === ' ') {{
      var aud = document.querySelector('audio');
      if (aud) {{ aud.paused ? aud.play() : aud.pause(); }}
      e.preventDefault();
    }} else if (e.key === 'n') {{
      var tog = document.getElementById("ab-note-toggle");
      if (tog) {{ tog.checked = !tog.checked; tog.dispatchEvent(new Event('change')); }}
      e.preventDefault();
    }}
  }};
  document.addEventListener('keydown', window._abKeyHandler, true);
}})();
"""))

    def wire_multi_dimension_js(self) -> None:
        # dimension A/B/Tie clicks stay client-side; only Next hits the kernel (avoids full-page flash)
        key_map_js = _dim_key_map_js(self.active_dimensions)
        active_dims_js = "[" + ", ".join(f'"{d}"' for d in self.active_dimensions) + "]"
        init_picks_js = "{" + ", ".join(f'"{d}": null' for d in self.active_dimensions) + "}"
        display(Javascript(f"""
(function() {{
  if (!google.colab || !google.colab.kernel) {{
    console.error("google.colab.kernel not available");
    return;
  }}
  var invoke = google.colab.kernel.invokeFunction;
  var toggle = document.getElementById("ab-note-toggle");
  var field = document.getElementById("ab-note-field");
  if (toggle && field) {{
    toggle.onchange = function() {{
      field.style.display = toggle.checked ? "block" : "none";
      if (!toggle.checked) field.value = "";
    }};
  }}
  var activeDims = {active_dims_js};
  window.abDimPicks = {init_picks_js};
  function abUpdateDimUI() {{
    var allSet = true;
    activeDims.forEach(function(dim) {{
      document.querySelectorAll('[data-ab-dim="' + dim + '"]').forEach(function(btn) {{
        var choice = btn.getAttribute("data-ab-choice-val");
        var sel = window.abDimPicks[dim] === choice;
        btn.style.fontWeight = sel ? "600" : "";
        btn.style.background = sel ? "#dcfce7" : "";
      }});
      if (!window.abDimPicks[dim]) allSet = false;
    }});
    var nextBtn = document.getElementById("ab-next-btn");
    if (nextBtn) nextBtn.disabled = !allSet;
  }}
  document.querySelectorAll("[data-ab-dim]").forEach(function(btn) {{
    btn.onclick = function() {{
      window.abDimPicks[btn.getAttribute("data-ab-dim")] = btn.getAttribute("data-ab-choice-val");
      abUpdateDimUI();
    }};
  }});
  var nextBtn = document.getElementById("ab-next-btn");
  if (nextBtn) {{
    nextBtn.onclick = async function() {{
      if (nextBtn.disabled) return;
      var note = field ? field.value.trim() : "";
      var payload = "submit:" + activeDims.map(function(d) {{
        return d + ":" + window.abDimPicks[d];
      }}).join(",");
      await invoke("{CALLBACK_NAME}", [payload, note], {{}});
    }};
  }}
  abUpdateDimUI();

  // Focus: make document.body focusable so keyboard events land here even after
  // clicking the audio play button (which would otherwise leave the iframe without focus).
  document.body.tabIndex = -1;
  document.body.focus();
  var audio = document.querySelector("audio");
  if (audio) {{
    audio.addEventListener("play", function() {{
      document.body.focus();
    }});
  }}

  // Keyboard shortcuts (capture phase so they fire even when audio element has focus).
  var abKeyMap = {key_map_js};
  if (window._abKeyHandler) document.removeEventListener('keydown', window._abKeyHandler, true);
  window._abKeyHandler = function(e) {{
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    // Space — toggle audio play/pause
    if (e.key === ' ') {{
      var aud = document.querySelector('audio');
      if (aud) {{ aud.paused ? aud.play() : aud.pause(); }}
      e.preventDefault();
      return;
    }}
    // Enter — submit when all dimensions selected
    if (e.key === 'Enter') {{
      var nb = document.getElementById("ab-next-btn");
      if (nb && !nb.disabled) {{ nb.click(); e.preventDefault(); }}
      return;
    }}
    // n — toggle Add note
    if (e.key === 'n') {{
      var tog = document.getElementById("ab-note-toggle");
      if (tog) {{ tog.checked = !tog.checked; tog.dispatchEvent(new Event('change')); }}
      e.preventDefault();
      return;
    }}
    var mapping = abKeyMap[e.key];
    if (!mapping) return;
    e.preventDefault();
    var btn = document.querySelector('[data-ab-dim="' + mapping[0] + '"][data-ab-choice-val="' + mapping[1] + '"]');
    if (btn) btn.click();
  }};
  document.addEventListener('keydown', window._abKeyHandler, true);
}})();
"""))

    def overall_buttons_html(self) -> str:
        return (
            self.choice_button("Choose A", "A")
            + self.choice_button("Choose B", "B")
            + self.choice_button("Tie", "tie")
        )

    def show_complete(self) -> str:
        from ab_prefs_interface.summarize_preferences import summarize_completion_html

        return rating_style + summarize_completion_html(
            self.output_json_path, self.ground_truth_name, self.rating_mode
        )

    def page_html(self) -> str:
        if self.current_index >= len(self.queue):
            return self.show_complete()
        unit, provider_a, provider_b = self.current_item()
        audio_url = self.audio_urls[unit.span_key]
        clip_duration = unit.end_seconds - unit.start_seconds
        body = comparison_block(
            unit=unit,
            provider_a=provider_a,
            provider_b=provider_b,
            audio_url=audio_url,
            clip_duration=clip_duration,
            time_offset=unit.start_seconds,
            show_providers=self.show_providers,
            debug=self.debug,
            item_label=f"Preference item {self.current_index + 1}/{len(self.queue)}",
        )
        if self.rating_mode == "multi_dimension":
            buttons = dimension_rows_html(self.dimension_picks, self.active_dimensions)
        else:
            buttons = self.overall_buttons_html()
        if self.show_note:
            buttons += (
                '<label style="margin-left:12px;"><input type="checkbox" id="ab-note-toggle"> Add note</label>'
            )
        status = (
            f'<p style="margin:8px 0;color:#4b5563;">{html.escape(self.status_message)}</p>'
            if self.status_message
            else ""
        )
        note_field = ""
        if self.show_note:
            note_field = (
                '<textarea id="ab-note-field" placeholder="Optional note" '
                'style="display:none;width:100%;max-width:900px;height:70px;margin-top:6px;"></textarea>'
            )
        return f'{rating_style}{body}{status}<div style="margin-top:10px;">{buttons}</div>{note_field}'

    def render(self, *, placeholder: bool = False) -> None:
        clear_output(wait=True)
        if placeholder or not self.audio_urls:
            display(HTML(
                f"{rating_style}<p style=\"color:#4b5563;\">"
                f"Preparing audio clips ({len(self.queue)} items)…</p>"
            ))
            return
        display(HTML(self.page_html()))
        if self.current_index < len(self.queue):
            self.wire_page_js()

    def _sync_to_gcs(self) -> None:
        """Push the output JSON to GCS in the background after every save (best-effort)."""
        if not self.gcs_bucket:
            return
        gcs_path = (
            f"gs://{self.gcs_bucket}/"
            f"{self.output_json_path.parent.name}/"
            f"{self.output_json_path.name}"
        )
        try:
            subprocess.Popen(
                ["gsutil", "cp", str(self.output_json_path), gcs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass  # gsutil unavailable — silent, never interrupt a rating

    def submit_dimension_record(self, note: str) -> None:
        if not all_dimensions_selected(self.dimension_picks, self.active_dimensions):
            return
        unit, provider_a, provider_b = self.current_item()
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
            choice_text=str(self.dimension_picks.get("text", "")),
            choice_timing=str(self.dimension_picks.get("timing", "")),
            choice_diarization=str(self.dimension_picks.get("diarization", "")),
            choice_punctuation=str(self.dimension_picks.get("punctuation", "")),
        )
        append_record(self.output_json_path, record)
        self._sync_to_gcs()
        self.current_index += 1
        saved_parts = [f"{dim}={self.dimension_picks[dim]}" for dim in self.active_dimensions]
        self.status_message = "Saved: " + ", ".join(saved_parts)
        self.dimension_picks = empty_dimension_picks(self.active_dimensions)
        self.render()

    def on_choice(self, choice: str, note: str = "") -> None:
        if self.current_index >= len(self.queue):
            return
        note = (note or "").strip()
        if self.rating_mode == "multi_dimension":
            picks = parse_dimension_submit(choice, self.active_dimensions)
            if picks is not None:
                self.dimension_picks = picks
                self.submit_dimension_record(note)
                return
            if choice == "next":
                self.submit_dimension_record(note)
                return
            parsed = parse_dimension_choice(choice)
            if parsed is None:
                return
            dimension, dim_choice = parsed
            self.dimension_picks[dimension] = dim_choice
            return
        unit, provider_a, provider_b = self.current_item()
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
        self._sync_to_gcs()
        self.current_index += 1
        if self.show_providers:
            self.status_message = f"Saved: {choice} for {unit.span_key} ({provider_a} vs {provider_b})"
        else:
            self.status_message = f"Saved: {choice}"
        self.render()
