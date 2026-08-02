"""Data Loader Agent: get the user's file into a dataframe, or say why not.

Business files are messy in predictable ways - the wrong delimiter, a title row
above the headers, four sheets where only one holds data. Each of those is
handled here so the rest of the pipeline can assume a clean rectangle.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState
from .base import Agent, Flow

#: Extensions we can read, mapped to a human name.
SUPPORTED = {
    ".csv": "CSV file",
    ".tsv": "tab-separated file",
    ".txt": "text file",
    ".xlsx": "Excel workbook",
    ".xlsm": "Excel workbook with macros",
    ".xls": "older Excel workbook",
}

#: Encodings tried in order when reading a text file.
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

#: Bytes read when sniffing a text file's shape.
SNIFF_BYTES = 64 * 1024

#: A header row this full of blanks or placeholders is probably not the header.
BAD_HEADER_SHARE = 0.4


class LoadError(Exception):
    """The file could not be read at all."""


def sniff_text_format(path: Path) -> tuple[str, str]:
    """Work out a text file's encoding and delimiter."""
    sample = b""
    for encoding in ENCODINGS:
        try:
            with path.open("rb") as handle:
                sample = handle.read(SNIFF_BYTES)
            sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        else:
            text = sample.decode(encoding)
            break
    else:
        raise LoadError(
            "This file is not saved in a text format we can read. Re-saving it "
            "from Excel as CSV UTF-8 usually fixes it."
        )

    if path.suffix.lower() == ".tsv":
        return encoding, "\t"

    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        return encoding, dialect.delimiter
    except csv.Error:
        # Fall back to whichever candidate appears most often in the first line.
        first_line = text.splitlines()[0] if text.splitlines() else ""
        counts = {candidate: first_line.count(candidate) for candidate in ",;\t|"}
        best = max(counts, key=counts.get)
        return encoding, best if counts[best] else ","


def header_looks_wrong(frame: pd.DataFrame) -> bool:
    """True when the first row was probably a title, not the column names."""
    if frame.empty:
        return False
    names = [str(name) for name in frame.columns]
    placeholders = sum(
        1 for name in names if name.startswith("Unnamed:") or not name.strip()
    )
    return placeholders / len(names) >= BAD_HEADER_SHARE


def read_excel_sheets(path: Path) -> list[str]:
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception as error:  # pragma: no cover - depends on the file
        raise LoadError(f"This workbook could not be opened: {error}") from error


class DataLoaderAgent(Agent):
    stage = "load"
    key = "loader"
    title = "Loading your data"

    persona = AgentPersona(
        role="Data intake specialist",
        goal=(
            "Read the owner's file correctly on the first attempt and tell them "
            "plainly what arrived, so they can spot straight away if it is not "
            "the file they meant to send."
        ),
        backstory=(
            "You have opened thousands of business exports and you know every way "
            "they go wrong: a title row above the headers, four sheets where only "
            "one holds data, semicolons instead of commas. You fix what you can "
            "silently and ask only about the things that would change the numbers."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        path = state.source_path
        if path is None or not Path(path).exists():
            state.fail_stage(self.stage, "No file was provided to analyse.")
            return

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            state.fail_stage(
                self.stage,
                f"{path.name} is a {suffix or 'file with no extension'}, which "
                "cannot be read. Save it as CSV or Excel and try again.",
            )
            return

        state.source_name = path.name
        state.source_format = SUPPORTED[suffix]

        try:
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                frame = yield from self._load_excel(state, path)
            else:
                frame = yield from self._load_text(state, path)
        except LoadError as error:
            state.fail_stage(self.stage, str(error))
            return

        if frame is None:
            state.skip_stage(self.stage, "Loading was skipped, so there is nothing to analyse.")
            return

        frame = self._tidy(frame)
        if frame.empty:
            state.fail_stage(
                self.stage,
                "The file was read but holds no rows once empty lines were removed.",
            )
            return

        state.raw_frame = frame.copy()
        # Taken now, from the file as it arrived: cleaning and feature
        # engineering both change the columns in data-dependent ways, so a
        # fingerprint taken later would never match a previous run.
        from ..analysis.comparison import fingerprint

        state.fingerprint = fingerprint(frame)
        state.set_frame(
            frame,
            self.stage,
            f"Loaded {len(frame):,} rows and {frame.shape[1]} columns from {path.name}.",
        )
        state.finish_stage(
            self.stage,
            f"Read {len(frame):,} rows and {frame.shape[1]} columns from "
            f"{path.name} ({state.source_format}).",
        )

    # -- format-specific loading -------------------------------------------

    def _load_text(self, state: PipelineState, path: Path) -> Any:
        encoding, delimiter = sniff_text_format(path)
        readable = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "bar"}.get(
            delimiter, delimiter
        )
        self.note(
            state,
            f"Read as a {readable}-separated file in {encoding} encoding.",
            delimiter=delimiter,
            encoding=encoding,
        )
        state.load_notes.append(f"Columns separated by {readable}, {encoding} encoding.")

        try:
            frame = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
        except Exception as error:
            raise LoadError(
                f"The file could not be read as a table: {error}"
            ) from error

        if header_looks_wrong(frame):
            frame = yield from self._confirm_header(state, path, frame, encoding, delimiter)
        return frame

    def _load_excel(self, state: PipelineState, path: Path) -> Any:
        sheets = read_excel_sheets(path)
        if not sheets:
            raise LoadError("This workbook has no sheets in it.")

        sheet = sheets[0]
        if len(sheets) > 1:
            sizes = self._sheet_sizes(path, sheets)
            largest = max(sizes, key=lambda name: sizes[name])
            listing = ", ".join(
                f"{name} ({sizes[name]:,} rows)" for name in sheets[:10]
            )
            decision = self.decide(
                topic="Which sheet to analyse",
                question=f"This workbook has {len(sheets)} sheets. Which one holds the data you want analysed?",
                context=(
                    f"The sheets are: {listing}. Sheets with very few rows are "
                    "usually notes, lookup lists or a summary rather than the "
                    "underlying records."
                ),
                suggestion=Option(
                    label=f"Analyse the {largest} sheet",
                    rationale=(
                        f"It has the most rows ({sizes[largest]:,}), which usually "
                        "means it holds the actual records rather than a summary."
                    ),
                    payload={"sheet": largest},
                ),
                alternatives=[
                    Option(
                        label=f"Analyse {name}",
                        rationale=f"{sizes[name]:,} rows.",
                        payload={"sheet": name},
                    )
                    for name in sheets
                    if name != largest
                ][:6],
                custom_prompt="Type the exact name of the sheet you want.",
                skip_effect=(
                    f"We will use the first sheet, {sheets[0]}, without checking "
                    "whether it is the right one."
                ),
                evidence={"sheets": sizes},
            )
            answer = yield decision

            if answer.is_custom and answer.text.strip() in sheets:
                sheet = answer.text.strip()
            elif answer.is_custom:
                self.warn(
                    state,
                    f'There is no sheet called "{answer.text.strip()}", so the '
                    f"largest sheet ({largest}) was used instead.",
                )
                sheet = largest
            elif answer.is_skip:
                sheet = sheets[0]
            else:
                sheet = answer.payload.get("sheet", largest)

        state.load_notes.append(f"Read from the sheet named {sheet}.")
        try:
            frame = pd.read_excel(path, sheet_name=sheet)
        except Exception as error:
            raise LoadError(f"The sheet {sheet} could not be read: {error}") from error

        if header_looks_wrong(frame):
            frame = yield from self._confirm_header(state, path, frame, sheet=sheet)
        return frame

    def _sheet_sizes(self, path: Path, sheets: list[str]) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for name in sheets:
            try:
                sizes[name] = len(pd.read_excel(path, sheet_name=name, usecols=[0]))
            except Exception:
                sizes[name] = 0
        return sizes

    # -- header repair -----------------------------------------------------

    def _confirm_header(
        self,
        state: PipelineState,
        path: Path,
        frame: pd.DataFrame,
        encoding: str | None = None,
        delimiter: str | None = None,
        sheet: str | None = None,
    ) -> Any:
        """Ask before assuming the real headers are further down the file."""
        preview = frame.head(4).to_string(index=False, max_colwidth=18)
        decision = self.decide(
            topic="Where the column names are",
            question=(
                "The top row of this file does not look like column names. "
                "Should we skip it and use the row below instead?"
            ),
            context=(
                "Files exported from accounting and till systems often start with "
                "a title or a date range above the real headings. If we read the "
                "title as headings, every column ends up with a meaningless name.\n\n"
                f"This is what the top of the file looks like:\n{preview}"
            ),
            suggestion=Option(
                label="Skip the first row and use the next one as headings",
                rationale="The current headings are blank or auto-numbered, which "
                "means the real ones are almost certainly on the next line.",
                payload={"skiprows": 1},
            ),
            alternatives=[
                Option(
                    label="Skip the first two rows",
                    rationale="Some exports put a title and a blank line above the headings.",
                    payload={"skiprows": 2},
                ),
                Option(
                    label="Keep the file exactly as it is",
                    rationale="Use it if the current headings are correct for your file.",
                    payload={"skiprows": 0},
                ),
            ],
            custom_prompt="Tell us which row holds the column names, counting from 1.",
            skip_effect="The file is read as-is, with the current headings kept.",
        )
        answer = yield decision

        if answer.is_skip:
            return frame

        skiprows = int(answer.payload.get("skiprows", 1))
        if answer.is_custom:
            digits = "".join(
                character for character in answer.text if character.isdigit()
            )
            skiprows = max(0, int(digits) - 1) if digits else 1
        if skiprows == 0:
            return frame

        try:
            if sheet is not None:
                repaired = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows)
            else:
                repaired = pd.read_csv(
                    path,
                    sep=delimiter,
                    encoding=encoding,
                    skiprows=skiprows,
                    low_memory=False,
                )
        except Exception as error:
            self.warn(state, f"Could not re-read the file skipping {skiprows} rows: {error}")
            return frame

        state.load_notes.append(f"Skipped the first {skiprows} row(s) to find the headings.")
        return repaired

    # -- tidying -----------------------------------------------------------

    @staticmethod
    def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
        """Trim the shape of the loaded frame without changing any values."""
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        frame.columns = [
            str(name).strip() or f"column_{index + 1}"
            for index, name in enumerate(frame.columns)
        ]

        # Duplicate headings make every later lookup ambiguous, so they are made
        # unique here rather than failing much further down the pipeline.
        seen: dict[str, int] = {}
        unique_names = []
        for name in frame.columns:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        frame.columns = unique_names
        return frame.reset_index(drop=True)
