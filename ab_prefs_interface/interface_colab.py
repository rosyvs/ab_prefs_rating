"""Colab rating UI via IPython HTML + kernel callbacks (ipywidgets breaks when pre-imported)."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from IPython.display import HTML, Javascript, clear_output, display

from ab_prefs_interface.audio_clips import ensure_queue_clips
from ab_prefs_interface.data_model import ComparisonUnit, PreferenceRecord
from ab_prefs_interface.interface_notebook import comparison_block, rating_style
from ab_prefs_interface.storage_json import append_record

CALLBACK_NAME = "ab_prefs_choice"


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
        self.current_index = 0
        self.clip_dir = clip_dir or Path("results/ab_prefs/audio_clips")
        self.notebook_root = notebook_root or Path.cwd()
        self.verbose = verbose
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
        self.render()

    def choice_button(self, label: str, choice: str) -> str:
        # no onclick — Colab sanitizes it from display(HTML); wired in wire_choice_buttons()
        return (
            f'<button type="button" data-ab-choice="{html.escape(choice)}" '
            f'style="margin:4px 8px 4px 0;padding:6px 14px;">{html.escape(label)}</button>'
        )

    def wire_page_js(self) -> None:
        show_note = "true" if self.show_note else "false"
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
    toggle.checked = {show_note};
    field.style.display = toggle.checked ? "block" : "none";
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
}})();
"""))

    def page_html(self) -> str:
        if self.current_index >= len(self.queue):
            from ab_prefs_interface.summarize_preferences import summarize_cli_command

            summarize_cmd = summarize_cli_command(self.output_json_path, self.ground_truth_name)
            print(f"\nSummarize after rating (separate, optional):\n{summarize_cmd}")
            return (
                f"{rating_style}<h3>Review complete</h3>"
                f"<p>Saved responses to <code>{html.escape(str(self.output_json_path))}</code></p>"
                f"<p><strong>Summarize after rating (separate, optional):</strong></p>"
                f"<pre>{html.escape(summarize_cmd)}</pre>"
            )
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
            item_label=f"Preference item {self.current_index + 1}/{len(self.queue)}",
        )
        buttons = (
            self.choice_button("Choose A", "A")
            + self.choice_button("Choose B", "B")
            + self.choice_button("Tie", "tie")
            + self.choice_button("Skip", "skip")
            + '<label style="margin-left:12px;"><input type="checkbox" id="ab-note-toggle"> Add note</label>'
        )
        status = (
            f'<p style="margin:8px 0;color:#4b5563;">{html.escape(self.status_message)}</p>'
            if self.status_message
            else ""
        )
        note_field = (
            '<textarea id="ab-note-field" placeholder="Optional note about why A/B/tie/skip" '
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

    def on_choice(self, choice: str, note: str = "") -> None:
        if self.current_index >= len(self.queue):
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
            note=(note or "").strip(),
            ground_truth_text=unit.ground_truth_text,
            transcript_a=unit.provider_candidates[provider_a].text,
            transcript_b=unit.provider_candidates[provider_b].text,
        )
        append_record(self.output_json_path, record)
        self.current_index += 1
        if self.show_providers:
            self.status_message = f"Saved: {choice} for {unit.span_key} ({provider_a} vs {provider_b})"
        else:
            self.status_message = f"Saved: {choice}"
        self.render()
