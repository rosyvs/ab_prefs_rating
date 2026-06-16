"""Shared helpers for multi-dimension A/B rating UI (notebook + Colab)."""
from __future__ import annotations

import html

DIMENSIONS = ("text", "timing", "diarization", "punctuation")
DIMENSION_LABELS = {"text": "Text", "timing": "Timing", "diarization": "Diarization", "punctuation": "Punctuation"}
# Keyboard shortcuts: row → choice → key
# punctuation shares a/s/d with diarization (never active at the same time);
# if both are ever active simultaneously, punctuation falls back to z/x/c.
DIMENSION_KEYS: dict[str, dict[str, str]] = {
    "text":         {"A": "1", "B": "2", "tie": "3"},
    "timing":       {"A": "q", "B": "w", "tie": "e"},
    "diarization":  {"A": "a", "B": "s", "tie": "d"},
    "punctuation":  {"A": "a", "B": "s", "tie": "d"},
}
# Numpad shortcuts (e.code): text=789, timing=456, diar/punc=123
NUMPAD_KEYS: dict[str, dict[str, str]] = {
    "text":         {"A": "Numpad7", "B": "Numpad8", "tie": "Numpad9"},
    "timing":       {"A": "Numpad4", "B": "Numpad5", "tie": "Numpad6"},
    "diarization":  {"A": "Numpad1", "B": "Numpad2", "tie": "Numpad3"},
    "punctuation":  {"A": "Numpad1", "B": "Numpad2", "tie": "Numpad3"},
}


def get_effective_keys(
    active_dimensions: "tuple[str, ...] | list[str]",
) -> dict[str, dict[str, str]]:
    """Return key mapping for active dimensions.

    Punctuation shares a/s/d with diarization. If both are active at once,
    diarization keeps a/s/d and punctuation falls back to z/x/c.
    """
    both = "diarization" in active_dimensions and "punctuation" in active_dimensions
    result: dict[str, dict[str, str]] = {}
    for dim in active_dimensions:
        if dim == "punctuation" and both:
            result[dim] = {"A": "z", "B": "x", "tie": "c"}
        else:
            result[dim] = DIMENSION_KEYS[dim]
    return result
DIMENSION_CHOICE_COLS = {
    "text": "choice_text",
    "timing": "choice_timing",
    "diarization": "choice_diarization",
    "punctuation": "choice_punctuation",
}
DIMENSION_CHOICES = ("A", "B", "tie")
RATING_MODES = ("overall", "multi_dimension")


def empty_dimension_picks(dimensions: tuple[str, ...] | list[str] = DIMENSIONS) -> dict[str, str | None]:
    return {dim: None for dim in dimensions}


def all_dimensions_selected(
    picks: dict[str, str | None],
    dimensions: tuple[str, ...] | list[str] = DIMENSIONS,
) -> bool:
    return all(picks.get(dim) in DIMENSION_CHOICES for dim in dimensions)


def encode_dimension_choice(dimension: str, choice: str) -> str:
    return f"{dimension}:{choice}"


def parse_dimension_choice(value: str) -> tuple[str, str] | None:
    if ":" not in value:
        return None
    dimension, choice = value.split(":", 1)
    if dimension not in DIMENSIONS or choice not in DIMENSION_CHOICES:
        return None
    return dimension, choice


def dimension_button_label(choice: str) -> str:
    return "Tie" if choice == "tie" else choice


def parse_dimension_submit(
    value: str,
    dimensions: tuple[str, ...] | list[str] = DIMENSIONS,
) -> dict[str, str] | None:
    """Parse submit:text:A,timing:B,diarization:tie from Colab Next callback."""
    if not value.startswith("submit:"):
        return None
    picks: dict[str, str] = {}
    for part in value[len("submit:"):].split(","):
        if ":" not in part:
            return None
        dimension, choice = part.split(":", 1)
        if dimension not in DIMENSIONS or choice not in DIMENSION_CHOICES:
            return None
        picks[dimension] = choice
    if set(picks.keys()) != set(dimensions):
        return None
    return picks


def dimension_rows_html(
    picks: dict[str, str | None],
    dimensions: tuple[str, ...] | list[str] = DIMENSIONS,
) -> str:
    """HTML for dimension rows with A/B/Tie buttons (Colab). Selection updated client-side."""
    kbd_style = (
        "font-size:10px;border:1px solid #9ca3af;border-radius:3px;"
        "padding:0 3px;margin-left:4px;background:#f3f4f6;color:#4b5563;"
    )
    btn_base = "min-width:72px;margin:2px 4px 2px 0;padding:4px 10px;"
    eff_keys = get_effective_keys(dimensions)
    rows: list[str] = ['<table style="border-collapse:collapse;margin:4px 0;">']
    for dim in dimensions:
        label = DIMENSION_LABELS[dim]
        cells = f'<td style="padding:3px 14px 3px 0;font-weight:600;white-space:nowrap;">{html.escape(label)}</td>'
        for choice in DIMENSION_CHOICES:
            selected = picks.get(dim) == choice
            sel_style = "font-weight:600;background:#dcfce7;" if selected else ""
            key = eff_keys[dim][choice]
            cells += (
                f'<td style="padding:2px 2px;">'
                f'<button type="button" data-ab-dim="{html.escape(dim)}" '
                f'data-ab-choice-val="{html.escape(choice)}" '
                f'style="{btn_base}{sel_style}">'
                f'{html.escape(dimension_button_label(choice))}'
                f'<kbd style="{kbd_style}">{html.escape(key)}</kbd>'
                f'</button></td>'
            )
        rows.append(f"<tr>{cells}</tr>")
    rows.append("</table>")
    next_disabled = "" if all_dimensions_selected(picks, dimensions) else " disabled"
    rows.append(
        f'<button type="button" id="ab-next-btn" data-ab-action="next"{next_disabled} '
        f'style="margin-top:10px;padding:6px 16px;">Next item</button>'
    )
    return "\n".join(rows)
