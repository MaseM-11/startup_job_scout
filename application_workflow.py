from __future__ import annotations

from pathlib import Path
from typing import Iterable, MutableSet, Sequence, Tuple

import gradio as gr
import pandas as pd
import sys
import numpy as np

from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
import os

load_dotenv()

## Google Sheets API setups
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
TAB_NAME_INTERESTED = os.getenv("GOOGLE_SHEET_TAB_INTERESTED")
TAB_NAME_APPLIED = os.getenv("GOOGLE_SHEET_TAB_APPLIED")
# Auth
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
google = gspread.authorize(creds)
# Open sheet and tab
sheet_interested = google.open(SHEET_NAME).worksheet(TAB_NAME_INTERESTED)
sheet_applied = google.open(SHEET_NAME).worksheet(TAB_NAME_APPLIED)

INTERESTED_PATH = Path("apply_data/interested_jobs.csv")
APPLIED_PATH = Path("apply_data/applied_jobs.csv")
INTERESTED_HISTORY_PATH = Path("apply_data/interested_jobs_history.csv")
APPLIED_HISTORY_PATH = Path("apply_data/applied_jobs_history.csv")
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

def ensure_header(ws, columns):
    first = ws.row_values(1)
    if first != columns:
        ws.update(range_name="A1", values=[columns])

ensure_header(sheet_interested, DISPLAY_COLUMNS)
ensure_header(sheet_applied, DISPLAY_COLUMNS)


def _load_index_set(path: Path) -> MutableSet[int]:
    """Return previously saved checkbox indices from ``path``."""
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            return set()
        if not df.empty:
            first_col = df.columns[0]
            try:
                return set(df[first_col].dropna().astype(int).tolist())
            except Exception:
                return set()
    return set()

def _append_job_record(
    idx: int,
    job_records: Sequence[dict[str, object]] | None,
    path: Path,
    sheet,
) -> None:
    """Append the job at ``idx`` to local CSV and append ONLY that row to Google Sheets."""
    if job_records is None:
        return

    # --- pick the selected record safely ---
    try:
        record = job_records[int(idx)]
    except (ValueError, TypeError, IndexError):
        return
    if not isinstance(record, dict):
        return

    new_row = pd.DataFrame([record])

    # --- update local history CSV (idempotent) ---
    try:
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_row], ignore_index=True).drop_duplicates()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)

     # --- order by the SHEET HEADER and coerce to JSON-safe types ---
    header = sheet.row_values(1)  # first row is the header
    def to_jsonable(v):
        if v is None or (isinstance(v, float) and pd.isna(v)): return ""
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v) if np.isfinite(v) else ""
        if isinstance(v, (list, tuple, set)): return " • ".join(map(str, v))
        if isinstance(v, (bool, int, float, str)): return v
        return str(v)

    row_dict = new_row.iloc[0].to_dict()
    row_values = [to_jsonable(row_dict.get(col, "")) for col in header]

    # --- append exactly one row in correct order ---
    sheet.append_row(row_values, value_input_option="USER_ENTERED")


def _save_index_set(indices: Iterable[int], path: Path, column_name: str) -> None:
    """Persist checkbox indices to ``path`` using ``column_name``."""
    df = pd.DataFrame(sorted(indices), columns=[column_name])
    df.to_csv(path, index=False)


def _toggle_selection(
    idx: int,
    is_checked: bool,
    selection: MutableSet[int],
    storage_path: Path,
    column_name: str,
    label: str,
) -> Tuple[str, MutableSet[int]]:
    """Update the ``selection`` set and persist to disk."""
    if is_checked:
        selection.add(idx)
    else:
        selection.discard(idx)

    _save_index_set(selection, storage_path, column_name)
    return f"{label}: {len(selection)} saved", selection


def _handle_interest_change(
    is_checked: bool,
    idx: int,
    interested_state: Iterable[int] | None,
    applied_state: Iterable[int] | None,
    job_records: Sequence[dict[str, object]] | None,
) -> Tuple[
    str,
    list[int],
    list[int],
    gr.Checkbox,
    gr.Markdown,
]:
    """Update interest selection and the applied workflow visibility."""
    interested_selection: MutableSet[int] = set(interested_state or [])
    applied_selection: MutableSet[int] = set(applied_state or [])

    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        return (
            f"Interested jobs: {len(interested_selection)} saved",
            sorted(interested_selection),
            sorted(applied_selection),
            gr.update(),
            gr.update(),
        )

    is_now_interested = bool(is_checked)

    status_msg, interested_selection = _toggle_selection(
        idx_int,
        is_now_interested,
        interested_selection,
        INTERESTED_PATH,
        "interested_index",
        "Interested jobs",
    )

    if is_now_interested:
        _append_job_record(idx_int, job_records, INTERESTED_HISTORY_PATH, sheet_interested)

    applied_visible = is_now_interested

    if not is_now_interested and idx_int in applied_selection:
        applied_selection.discard(idx_int)
        _save_index_set(applied_selection, APPLIED_PATH, "applied_index")
        status_msg = (
            f"{status_msg} | Applied jobs: {len(applied_selection)} saved"
        )

    applied_value = idx_int in applied_selection if applied_visible else False

    checkbox_update = gr.update(
        value=applied_value,
        visible=applied_visible,
    )
    markdown_update = gr.update(visible=applied_visible)

    return (
        status_msg,
        sorted(interested_selection),
        sorted(applied_selection),
        checkbox_update,
        markdown_update,
    )


def _handle_applied_change(
    is_checked: bool,
    idx: int,
    applied_state: Iterable[int] | None,
    job_records: Sequence[dict[str, object]] | None,
) -> Tuple[str, list[int]]:
    """Update applied selection for a job."""
    applied_selection: MutableSet[int] = set(applied_state or [])

    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        return f"Applied jobs: {len(applied_selection)} saved", sorted(applied_selection)

    is_now_applied = bool(is_checked)

    status_msg, applied_selection = _toggle_selection(
        idx_int,
        is_now_applied,
        applied_selection,
        APPLIED_PATH,
        "applied_index",
        "Applied jobs",
    )

    if is_now_applied:
        _append_job_record(idx_int, job_records, APPLIED_HISTORY_PATH, sheet_applied)

    return status_msg, sorted(applied_selection)


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


def build_interface(df: pd.DataFrame) -> gr.Blocks:
    """Create the Gradio Blocks interface for the workflow."""
    interested_indices = _load_index_set(INTERESTED_PATH)
    raw_applied_indices = _load_index_set(APPLIED_PATH)
    applied_indices = {idx for idx in raw_applied_indices if idx in interested_indices}
    if applied_indices != raw_applied_indices:
        _save_index_set(applied_indices, APPLIED_PATH, "applied_index")

    job_summaries = [_format_job_summary(row) for _, row in df.iterrows()]
    job_markdowns = [_format_job_markdown(row) for _, row in df.iterrows()]
    job_records = df.to_dict("records")

    with gr.Blocks(title="Job Application Workflow") as demo:
        gr.Markdown(
            """
            ## Job Application Workflow
            Review jobs, track interest, and mark applications in a single space.
            """
        )

        interested_state = gr.State(sorted(interested_indices))
        applied_state = gr.State(sorted(applied_indices))
        job_data_state = gr.State(job_records)
        status = gr.Markdown("Ready.")

        interested_components: list[tuple[gr.Checkbox, gr.Number]] = []
        applied_components: list[
            tuple[gr.Checkbox, gr.Markdown, gr.Number]
        ] = []

        with gr.Tab("Interested Jobs"):
            gr.Markdown(
                "1️⃣ **Browse the jobs below and check the roles you're interested in.**"
            )
            for idx, job_summary in enumerate(job_summaries):
                with gr.Group():
                    gr.Markdown(job_summary)
                    job_index = gr.Number(value=int(idx), visible=False)
                    interested_cb = gr.Checkbox(
                        label="Interested",
                        value=idx in interested_indices,
                    )
                interested_components.append((interested_cb, job_index))

        with gr.Tab("Applied Jobs"):
            gr.Markdown(
                "2️⃣ **As you apply, mark those jobs here. Only interested roles appear.**"
            )
            for idx, job_md in enumerate(job_markdowns):
                with gr.Group():
                    applied_md = gr.Markdown(
                        job_md,
                        visible=idx in interested_indices,
                    )
                    applied_index = gr.Number(value=int(idx), visible=False)
                    applied_cb = gr.Checkbox(
                        label="Applied",
                        value=idx in applied_indices,
                        visible=idx in interested_indices,
                    )
                applied_components.append((applied_cb, applied_md, applied_index))

        for (interested_cb, interested_idx), (
            applied_cb,
            applied_md,
            applied_idx,
        ) in zip(interested_components, applied_components):
            interested_cb.change(
                fn=_handle_interest_change,
                inputs=[
                    interested_cb,
                    interested_idx,
                    interested_state,
                    applied_state,
                    job_data_state,
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
                inputs=[applied_cb, applied_idx, applied_state, job_data_state],
                outputs=[status, applied_state],
            )

        gr.Markdown(
            "Saved selections live in `interested_jobs.csv` and `applied_jobs.csv`. "
            "All checked jobs are archived in the history CSV files."
        )

    return demo


## ---------------------------------------------------------------- ##
    

def main() -> None:
    input_file = sys.argv[1]
    base_name = Path(input_file).stem
    df = load_jobs(f'scraped_data/jobs_scraped_fitted_{base_name}.csv')
    demo = build_interface(df)
    demo.launch(share=True)


if __name__ == "__main__":
    main()