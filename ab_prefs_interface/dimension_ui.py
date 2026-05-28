"""Shared helpers for multi-dimension A/B rating UI (notebook + Colab)."""
from __future__ import annotations

import html

DIMENSIONS = ("text", "timing", "diarization")
DIMENSION_LABELS = {"text": "Text", "timing": "Timing", "diarization": "Diarization"}
DIMENSION_CHOICE_COLS = {
    "text": "choice_text",
    "timing": "choice_timing",
    "diarization": "choice_diarization",
}
DIMENSION_CHOICES = ("A", "B", "tie")
RATING_MODES = ("overall", "multi_dimension")


def empty_dimension_picks() -> dict[str, str | None]:
    return {dim: None for dim in DIMENSIONS}


def all_dimensions_selected(picks: dict[str, str | None]) -> bool:
    return all(picks[dim] in DIMENSION_CHOICES for dim in DIMENSIONS)


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


def dimension_rows_html(picks: dict[str, str | None]) -> str:
    """HTML for 3 dimension rows with A/B/Tie buttons (Colab)."""
    rows: list[str] = []
    for dim in DIMENSIONS:
        label = DIMENSION_LABELS[dim]
        buttons = ""
        for choice in DIMENSION_CHOICES:
            encoded = encode_dimension_choice(dim, choice)
            selected = picks.get(dim) == choice
            style = "font-weight:600;background:#dcfce7;" if selected else ""
            buttons += (
                f'<button type="button" data-ab-choice="{html.escape(encoded)}" '
                f'style="margin:2px 6px 2px 0;padding:4px 12px;{style}">'
                f"{html.escape(dimension_button_label(choice))}</button>"
            )
        rows.append(
            f'<div style="margin:6px 0;"><strong>{html.escape(label)}</strong> {buttons}</div>'
        )
    next_disabled = "" if all_dimensions_selected(picks) else " disabled"
    rows.append(
        f'<button type="button" data-ab-action="next"{next_disabled} '
        f'style="margin-top:10px;padding:6px 16px;">Next item</button>'
    )
    return "\n".join(rows)
