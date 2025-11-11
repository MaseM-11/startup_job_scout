from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Tuple

import gradio as gr
import pandas as pd


INTERESTED_PATH = Path("apply_data/interested_jobs.csv")
APPLIED_PATH = Path("apply_data/applied_jobs.csv")
DISPLAY_COLUMNS = [
    "Company",
    "title",
    "location",
    "date_posted",
    "link",
    "role description",
    "role responsibilities",
    "role reqs",
    "final fit",
    "yrs exp req",
]

PERSIST_COLUMNS = DISPLAY_COLUMNS


def _normalize_value(value: Any) -> str:
    """Return a consistent string representation for persistence."""

    if pd.isna(value):
        return ""
    return str(value)


def _extract_job_record(row: Mapping[str, Any]) -> dict[str, str]:
    """Extract persistent job fields from ``row``."""

    return {column: _normalize_value(row.get(column, "")) for column in PERSIST_COLUMNS}


def _job_identifier(record: Mapping[str, str]) -> str:
    """Return a stable identifier for a job record."""

    normalized = {column: _normalize_value(record.get(column, "")) for column in PERSIST_COLUMNS}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _parse_job_payload(payload: str) -> dict[str, str]:
    """Decode the JSON payload associated with a job component."""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = {}
    return {column: _normalize_value(data.get(column, "")) for column in PERSIST_COLUMNS}


def _format_job_summary(row: pd.Series) -> str:
    """Create a compact markdown summary for the interested workflow."""

    title = row.get("title", "Role")
    company = row.get("Company", "Company")
    location = row.get("location", "")
    posted = row.get("date_posted", "")
    score = row.get("final fit", "")
    years = row.get("yrs exp req", "")

    header = f"{company} — {title}"
    meta_bits: list[str] = []
    if pd.notna(location) and str(location):
        meta_bits.append(f"📍 {location}")
    if pd.notna(posted) and str(posted):
        meta_bits.append(f"🗓️ {posted}")
    if pd.notna(score) and str(score) != "":
        meta_bits.append(f"🏅 Fit score: {score}")
    if pd.notna(years) and str(years) != "":
        meta_bits.append(f"💼 Experience req: {years}")

    meta_line = " • ".join(meta_bits)
    return f"{header}\n{meta_line}" if meta_line else header


def _format_job_markdown(row: pd.Series) -> str:
    """Create detailed markdown for a job row."""

    title = row.get("title", "Role")
    company = row.get("Company", "Company")
    location = row.get("location", "Location unknown")
    posted = row.get("date_posted", "")
    score = row.get("final fit", "")
    years = row.get("yrs exp req", "")

    header = f"### {company} — {title}"
    meta_bits = []
    if location:
        meta_bits.append(f"📍 {location}")
    if posted:
        meta_bits.append(f"🗓️ {posted}")
    if score != "":
        meta_bits.append(f"🏅 Fit score: {score}")
    if years != "":
        meta_bits.append(f"💼 Experience req: {years}")
    meta = "  \n" + "  •  ".join(meta_bits) if meta_bits else ""

    link = row.get("link", "")
    if link:
        link_md = f"[Job link]({link})"
    else:
        link_md = ""

    description = row.get("role description", "")
    responsibilities = row.get("role responsibilities", "")
    requirements = row.get("role reqs", "")

    details = []
    if description:
        details.append(f"**Description**\n{description}")
    if responsibilities:
        details.append(f"**Responsibilities**\n{responsibilities}")
    if requirements:
        details.append(f"**Requirements**\n{requirements}")

    details_md = "\n\n".join(details)
    extras = f"\n\n{link_md}" if link_md else ""

    return f"{header}{meta}\n\n{details_md}{extras}"


def _load_saved_jobs(path: Path) -> dict[str, dict[str, str]]:
    """Return persisted job selections keyed by job identifier."""

    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    available_columns = [column for column in PERSIST_COLUMNS if column in df.columns]
    if not available_columns:
        return {}

    df = df[available_columns]
    selections: dict[str, dict[str, str]] = {}
    for record in df.to_dict(orient="records"):
        normalized = {column: _normalize_value(record.get(column, "")) for column in PERSIST_COLUMNS}
        selections[_job_identifier(normalized)] = normalized
    return selections


def _save_job_records(records: Iterable[Mapping[str, str]], path: Path) -> None:
    """Persist job selections to ``path`` preserving column order."""

    rows = [
        {column: _normalize_value(record.get(column, "")) for column in PERSIST_COLUMNS}
        for record in records
    ]

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=PERSIST_COLUMNS)

    for column in PERSIST_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[PERSIST_COLUMNS]
    df.to_csv(path, index=False)


def _sorted_records(selection: Mapping[str, dict[str, str]]) -> list[dict[str, str]]:
    """Return selection values sorted by their identifier."""

    return [selection[key] for key in sorted(selection)]


def _toggle_job_selection(
    job_record: Mapping[str, str],
    is_checked: bool,
    selection: MutableMapping[str, dict[str, str]],
    storage_path: Path,
    label: str,
) -> Tuple[str, MutableMapping[str, dict[str, str]]]:
    """Update a job selection mapping and persist it to disk."""

    job_key = _job_identifier(job_record)
    if is_checked:
        selection[job_key] = dict(job_record)
    else:
        selection.pop(job_key, None)

    _save_job_records(selection.values(), storage_path)
    return f"{label}: {len(selection)} saved", selection


def _handle_interest_change(
    is_checked: bool,
    job_payload: str,
    interested_state: Iterable[Mapping[str, str]] | None,
    applied_state: Iterable[Mapping[str, str]] | None,
) -> Tuple[
    str,
    list[dict[str, str]],
    list[dict[str, str]],
    gr.Checkbox,
    gr.Markdown,
]:
    """Update interest selection and the applied workflow visibility."""

    job_record = _parse_job_payload(job_payload)
    interested_selection: MutableMapping[str, dict[str, str]] = {
        _job_identifier(record): dict(record)
        for record in (interested_state or [])
    }
    applied_selection: MutableMapping[str, dict[str, str]] = {
        _job_identifier(record): dict(record)
        for record in (applied_state or [])
    }

    status_msg, interested_selection = _toggle_job_selection(
        job_record,
        bool(is_checked),
        interested_selection,
        INTERESTED_PATH,
        "Interested jobs",
    )

    job_key = _job_identifier(job_record)
    applied_visible = bool(is_checked)

    if not applied_visible and job_key in applied_selection:
        applied_selection.pop(job_key, None)
        _save_job_records(applied_selection.values(), APPLIED_PATH)
        status_msg = f"{status_msg} | Applied jobs: {len(applied_selection)} saved"

    applied_value = job_key in applied_selection if applied_visible else False

    checkbox_update = gr.update(
        value=applied_value,
        visible=applied_visible,
    )
    markdown_update = gr.update(visible=applied_visible)

    return (
        status_msg,
        _sorted_records(interested_selection),
        _sorted_records(applied_selection),
        checkbox_update,
        markdown_update,
    )



def _handle_applied_change(
    is_checked: bool,
    job_payload: str,
    applied_state: Iterable[Mapping[str, str]] | None,
) -> Tuple[str, list[dict[str, str]]]:
    """Update applied selection for a job."""

    job_record = _parse_job_payload(job_payload)
    applied_selection: MutableMapping[str, dict[str, str]] = {
        _job_identifier(record): dict(record)
        for record in (applied_state or [])
    }

    status_msg, applied_selection = _toggle_job_selection(
        job_record,
        bool(is_checked),
        applied_selection,
        APPLIED_PATH,
        "Applied jobs",
    )

    return status_msg, _sorted_records(applied_selection)



def build_interface(df: pd.DataFrame) -> gr.Blocks:
    """Create the Gradio Blocks interface for the workflow."""

    interested_map = _load_saved_jobs(INTERESTED_PATH)
    raw_applied_map = _load_saved_jobs(APPLIED_PATH)
    applied_map = {
        key: value for key, value in raw_applied_map.items() if key in interested_map
    }
    if len(applied_map) != len(raw_applied_map):
        _save_job_records(applied_map.values(), APPLIED_PATH)

    interested_records = _sorted_records(interested_map)
    applied_records = _sorted_records(applied_map)
    interested_keys = set(interested_map)
    applied_keys = set(applied_map)

    rows = list(df.iterrows())
    job_records = [_extract_job_record(row) for _, row in rows]
    job_keys = [_job_identifier(record) for record in job_records]
    job_payloads = [
        json.dumps(record, sort_keys=True, ensure_ascii=False) for record in job_records
    ]
    job_summaries = [_format_job_summary(row) for _, row in rows]
    job_markdowns = [_format_job_markdown(row) for _, row in rows]

    with gr.Blocks(title="Job Application Workflow") as demo:
        gr.Markdown(
            """
            ## Job Application Workflow
            Review jobs, track interest, and mark applications in a single space.
            """
        )

        interested_state = gr.State(interested_records)
        applied_state = gr.State(applied_records)
        status = gr.Markdown("Ready.")

        job_payload_components: list[gr.Textbox] = []
        interested_checkboxes: list[gr.Checkbox] = []
        applied_checkboxes: list[gr.Checkbox] = []
        applied_markdowns: list[gr.Markdown] = []

        with gr.Tab("Interested Jobs"):
            gr.Markdown(
                "1️⃣ **Browse the jobs below and check the roles you're interested in.**"
            )
            for job_summary, payload, key in zip(job_summaries, job_payloads, job_keys):
                with gr.Group():
                    gr.Markdown(job_summary)
                    job_payload_component = gr.Textbox(
                        value=payload,
                        visible=False,
                        interactive=False,
                    )
                    job_payload_components.append(job_payload_component)
                    interested_cb = gr.Checkbox(
                        label="Interested",
                        value=key in interested_keys,
                    )
                    interested_checkboxes.append(interested_cb)

        with gr.Tab("Applied Jobs"):
            gr.Markdown(
                "2️⃣ **As you apply, mark those jobs here. Only interested roles appear.**"
            )
            for job_md, key, job_payload_component in zip(
                job_markdowns, job_keys, job_payload_components
            ):
                with gr.Group():
                    applied_md = gr.Markdown(
                        job_md,
                        visible=key in interested_keys,
                    )
                    applied_markdowns.append(applied_md)
                    applied_cb = gr.Checkbox(
                        label="Applied",
                        value=key in applied_keys,
                        visible=key in interested_keys,
                    )
                    applied_checkboxes.append(applied_cb)

        for interested_cb, applied_cb, applied_md, job_payload_component in zip(
            interested_checkboxes,
            applied_checkboxes,
            applied_markdowns,
            job_payload_components,
        ):
            interested_cb.change(
                fn=_handle_interest_change,
                inputs=[
                    interested_cb,
                    job_payload_component,
                    interested_state,
                    applied_state,
                ],
                outputs=[
                    status,
                    interested_state,
                    applied_state,
                    applied_cb,
                    applied_md,
                ],
            )

            applied_cb.change(
                fn=_handle_applied_change,
                inputs=[applied_cb, job_payload_component, applied_state],
                outputs=[status, applied_state],
            )

        gr.Markdown(
            "Saved selections live in `interested_jobs.csv` and `applied_jobs.csv`."
        )

    return demo


def load_jobs(data_path):
    """Load and sort job data for display."""
    df = pd.read_csv(data_path)
    display_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    if display_cols:
        df = df[display_cols + [c for c in df.columns if c not in display_cols]]
    df = df.sort_values(by="final fit", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.index.name = "job_index"
    return df


## ---------------------------------------------------------------- ##
    

def main() -> None:
    input_file = sys.argv[1]
    base_name = Path(input_file).stem
    df = load_jobs(f'scraped_data/jobs_scraped_fitted_{base_name}.csv')
    demo = build_interface(df)
    demo.launch()


if __name__ == "__main__":
    main()
