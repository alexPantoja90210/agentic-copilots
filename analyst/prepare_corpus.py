"""
Build the corpus the agent is allowed to see, and prove it cannot see the rest.

    python prepare_corpus.py --workbook "IT Tickets Analysis.xlsb" --out ./corpus
    python prepare_corpus.py --workbook ... --out ./corpus --check-only

Why this file exists before any model code
------------------------------------------
The workbook answers its own questions. `Objective Answers` holds the computed
values, `Subjective Answers` holds the recommendation, the `Dashboard *` sheets
hold both again as charts, and `Tasks.docx` holds the answers AND the Excel
formulas used to get them.

Handing that workbook to an agent hands over the answer inside the evidence.
That is not a hypothetical: it is the defect that invalidated IA-45, where the
incident captions used the vocabulary of the fault being tested. The lesson was
that the leak is invisible from the inside -- re-reading your own prompt does
not reveal it, because you already know the answer and cannot unsee it.

So containment is not a step in preparing the data. It is the reason this file
exists, and everything else here is secondary.

IA-65.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The only two sheets that may reach the agent. An allowlist, not a blocklist:
# a blocklist silently admits any sheet somebody adds to the workbook later,
# and the failure mode of that mistake is a leaked answer nobody notices.
ALLOWED_SHEETS = ("Tickets", "IT Agents")

# Named for the refusal message, so the error tells the operator what it found
# rather than only that it refused. Prefix match covers the four dashboards.
QUARANTINED_PREFIXES = ("Objective Answers", "Subjective Answers", "Dashboard")

# Excel's serial date origin, written down rather than inferred. The 1899-12-30
# offset (not 12-31) absorbs Excel's fictional 1900 leap year.
#
# This constant is here because the two-clocks defect has already appeared three
# times in this project: arm A declaring 03:46 UTC over a series printed at
# 20:46; run_corpus.py printing a lead-in that had already passed; and the
# preflight window. Every one of them was a timestamp interpreted under an
# assumption nobody had written down. It does not get a fourth.
EXCEL_EPOCH = pd.Timestamp("1899-12-30")

# Columns whose raw form carries a numeric prefix ("2 - Normal") and whose
# cleaned twin lives beside it. Both are kept: the pair is the evidence that
# the "Imputated" columns impute nothing.
PAIRED_COLUMNS = (("Severity", "Severity Imputated"),
                  ("Priority", "Priority Imputated"))


class CorpusError(Exception):
    pass


def read_allowed(workbook: Path) -> dict:
    """
    The two permitted sheets, and a refusal if the workbook is not what we think.

    Reads by name. Reading by index would silently shift if a sheet were
    inserted, and the corpus would quietly acquire whatever landed in position
    zero -- which is exactly how an answer sheet reaches an agent.
    """
    if not workbook.exists():
        raise CorpusError("no workbook at %s" % workbook)
    sheets = {}
    for name in ALLOWED_SHEETS:
        try:
            frame = pd.read_excel(workbook, sheet_name=name, engine="pyxlsb")
        except ValueError as exc:
            raise CorpusError(
                "sheet %r is not in %s. The corpus is defined by an allowlist, "
                "so a renamed sheet stops the run rather than being guessed at: %s"
                % (name, workbook.name, exc))
        sheets[name] = frame
    return sheets


def quarantined_sheets(workbook: Path) -> list[str]:
    """Every sheet that must never reach the agent, named for the audit record."""
    every = pd.ExcelFile(workbook, engine="pyxlsb").sheet_names
    return [s for s in every
            if any(s.startswith(p) for p in QUARANTINED_PREFIXES)]


def drop_empty_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Columns that are null in every row, dropped and NAMED.

    Returned rather than logged in passing, because "we dropped a column" is a
    change to the evidence the agent will reason over. An unreported drop is a
    silent edit of the corpus, and this project does not make silent edits.
    """
    empty = [c for c in frame.columns if frame[c].isna().all()]
    return frame.drop(columns=empty), empty


def to_dates(frame: pd.DataFrame, column: str = "Fecha") -> pd.DataFrame:
    """Excel serials to real dates, under a stated epoch. See EXCEL_EPOCH."""
    if column not in frame.columns:
        raise CorpusError("expected a %r column to convert to dates" % column)
    out = frame.copy()
    out[column] = EXCEL_EPOCH + pd.to_timedelta(out[column].astype("int64"), unit="D")
    return out


def imputation_report(frame: pd.DataFrame) -> list[dict]:
    """
    Whether the "Imputated" columns impute anything. They do not.

    The workbook names two columns `Severity Imputated` and `Priority
    Imputated`, and the README makes a point of missing values. Neither is
    accurate: the value counts are identical to the source columns, no row is
    filled that was previously empty, and what actually changed is the label
    text -- a numeric prefix stripped and three spellings corrected.

    This is recorded as a finding rather than corrected, because a column whose
    name misdescribes what it holds is exactly the defect class this project
    keeps finding: a thing named after an operation it does not perform.
    """
    findings = []
    for raw, cleaned in PAIRED_COLUMNS:
        if raw not in frame.columns or cleaned not in frame.columns:
            continue
        findings.append({
            "raw_column": raw,
            "cleaned_column": cleaned,
            "raw_nulls": int(frame[raw].isna().sum()),
            "cleaned_nulls": int(frame[cleaned].isna().sum()),
            "raw_levels": int(frame[raw].nunique(dropna=True)),
            "cleaned_levels": int(frame[cleaned].nunique(dropna=True)),
            # The decisive test: same number of levels, same counts per level,
            # no null filled. If all three hold, nothing was imputed.
            "counts_identical": sorted(frame[raw].value_counts().tolist())
                                == sorted(frame[cleaned].value_counts().tolist()),
            "filled_any_null": bool(frame[raw].isna().sum()
                                    > frame[cleaned].isna().sum()),
        })
    return findings


# Values shorter than this are not treated as leak candidates. "40" is the
# average agent age AND a legitimate value in the Agent Age column, and a check
# that cannot tell those apart fires on every honest corpus. This is a stated
# limit of the scan, not an oversight: a short numeric answer must be checked by
# reading, because no substring rule can distinguish it from data.
MIN_LEAK_CANDIDATE_LEN = 4

# Columns that trip the prose heuristic and are not prose. Declared here with a
# reason each, rather than by loosening the heuristic until it stops
# complaining. Tuning a guard until it is quiet is how guards die: the guard
# survives, its usefulness does not, and nobody notices until the day it should
# have fired. An exception with a written reason can be reviewed; a moved
# threshold cannot.
DECLARED_NON_TEXT = {
    "Email": "an address is near-unique and ~30 characters, which is exactly "
             "what the heuristic keys on, but it is a structured identifier: "
             "an answer value either equals it or it does not. Needed in the "
             "corpus because question 6 asks the agent to extract the domain.",
}


def normalise(value) -> str:
    """Comparable form: lowercased, thousands separators and spaces removed."""
    return str(value).strip().lower().replace(",", "").replace(" ", "")


# The values the agent must never be shown, curated from the documented answers
# in Tasks.docx. Curated, not scraped, and that word carries a correction.
#
# The first version harvested every numeric cell from the quarantined sheets.
# Those sheets hold pivot tables, and a pivot table displays the INPUTS beside
# the outputs -- employee ids, agent ids, years of birth. The scan duly reported
# 148 leaks, every one of them a data value that appears in the answer sheet
# because the answer sheet is built from the data.
#
# That is the same defect a fifth time: a check that cannot distinguish two
# situations it treats as one -- here, "a computed answer" from "a data value
# the answer sheet happens to display". Only the DERIVED values can indicate a
# leak, and knowing which those are requires reading the task file. No harvest
# can do it.
DERIVED_ANSWERS = {
    "53.36507936507937": "average daily ticket volume",
    "-0.040536349147638": "correlation severity vs resolution time",
    "13051": "tickets in 2016",
    "29088": "tickets in 2020",
    "2609": "peak month, December 2020",
}

# Answers that CANNOT be scanned for, and why. Written down rather than omitted,
# because a guard's blind spots belong in the record next to what it covers.
#
#   4.5  / 4.553  daily average resolution time -- below MIN_LEAK_CANDIDATE_LEN,
#                 and "4.5" is three characters that occur in ordinary numbers.
#   40            average agent age -- also a legitimate value in Employee ID,
#                 Agent ID and Agent Age.
#   53            rounded daily volume -- also a legitimate Employee ID and
#                 Agent Age.
#   16            attribute count -- also a legitimate Employee ID, Agent ID and
#                 Resolution Time.
#
# These four are verified by reading the prompt, not by the scan. A scan that
# claimed to cover them would be asserting a guarantee it cannot keep.
UNSCANNABLE_ANSWERS = ("4.5", "4.553", "40", "53", "16")


def answers_from_workbook(workbook: Path) -> list[str]:
    """
    The curated derived answers. The argument is kept for the audit trail: the
    caller names the workbook these answers were read out of.

    The CHECKER may read the answers. The agent may not. That asymmetry is the
    design -- a leak detector that does not know the answer cannot detect a
    leak, which is why IA-59's detector is handed the fault vocabulary it is
    protecting against.
    """
    if not workbook.exists():
        raise CorpusError("no workbook at %s" % workbook)
    return sorted(DERIVED_ANSWERS)


def suspected_text_columns(frame: pd.DataFrame, min_mean_len: int = 40,
                           min_unique_ratio: float = 0.5) -> list[str]:
    """
    Columns that look like free prose, so the caller cannot forget to declare one.

    Prose is LONG and NEARLY UNIQUE. Both conditions matter, and the first
    version of this function used neither: it flagged any column where most
    values contained a space, and duly reported `Issue Type` ("IT Request"),
    `Severity` ("2 - Normal"), `Priority` ("0 - Unassiged") and `Agent Name`
    ("Barbara Grijalva") as free text.

    A space does not separate prose from a multi-word label. Length and
    cardinality do: `Agent Name` has 50 distinct values across 97,498 rows -- a
    ratio of 0.0005 -- and averages fifteen characters. A description column has
    a ratio near 1 and runs to hundreds.

    That was the fourth time in this file's short life that the same defect
    appeared: a check that cannot distinguish two situations it treats as one.
    It is recorded here rather than quietly fixed, because the frequency is the
    finding -- the defect is not rare, it is the default, and only a test that
    names both situations keeps it out.
    """
    text = []
    rows = max(len(frame), 1)
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty or values.dtype.kind not in "OSU":
            continue
        sample = values.astype(str).head(5000)
        mean_len = sample.str.len().mean()
        unique_ratio = values.nunique() / rows
        if mean_len >= min_mean_len or (mean_len >= 20
                                        and unique_ratio >= min_unique_ratio):
            text.append(str(column))
    return text


def leak_scan(frame: pd.DataFrame, answers: list[str],
              text_columns: tuple = (),
              declared_non_text: dict | None = None) -> list[dict]:
    """
    Whether any quarantined answer value reaches the corpus. Two modes, and the
    two modes are the point.

      exact      the cell IS the answer. Correct for identifiers, categories and
                 numbers, where a value either equals the answer or does not.
      substring  the cell CONTAINS the answer. Correct only for free prose.

    Both modes were wrong before they were right, and both failures were the
    same defect wearing different clothes -- a check that cannot distinguish two
    situations it treats as one:

    1. Scanning the SERIALISED CSV reported the human's peak-month figure 2,609
       inside `...,2017-02-02,609,22,...`: a date abutting an employee id,
       joined by the delimiter. The answer was never in the data.
    2. Scanning per cell but by SUBSTRING reported 2,193 hits, every one of them
       a four-digit answer occurring inside a ticket id like
       `TWLENR-8242410468`. Long digit strings contain short digit strings by
       arithmetic, not by leakage.

    A detector that fires on artefacts is worse than no detector: it teaches the
    operator to dismiss its output, and on the day it is right it is dismissed
    too. So the caller must say which columns are prose. There is no safe
    default, because each default is silently wrong on the other kind of column.

    This corpus has no free-text columns -- established by inspection and
    asserted by `suspected_text_columns` -- so the scan here is pure equality.
    That is a fact about this corpus, not a general setting. The Jira work has
    descriptions, and there the substring mode is the one that matters.
    """
    exempt = set(text_columns) | set(declared_non_text or DECLARED_NON_TEXT)
    undeclared = [c for c in suspected_text_columns(frame) if c not in exempt]
    if undeclared:
        raise CorpusError(
            "these columns look like free prose but were not declared as text "
            "columns, so the scan would ask the wrong question about exactly "
            "the place a leak is most likely to hide: %s" % undeclared)

    hits = []
    lookup = set(answers)
    for column in frame.columns:
        uniques = {normalise(v) for v in frame[column].dropna().unique()}
        if str(column) in text_columns:
            for answer in answers:
                for value in uniques:
                    if answer in value:
                        hits.append({"column": str(column), "answer": answer,
                                     "cell": value[:120], "mode": "substring"})
                        break
        else:
            for value in uniques & lookup:
                hits.append({"column": str(column), "answer": value,
                             "cell": value, "mode": "exact"})
    return hits


def assert_quarantine(out_dir: Path, workbook: Path) -> None:
    """
    Refuse if the agent's working root can reach the workbook.

    The agent is given `out_dir`. If the workbook -- which contains the answers
    -- sits inside it, then every containment decision above is decoration: the
    agent has a file-reading tool and the answers are one path away.

    Named paths in the refusal, because a guard that says only "refused" makes
    the operator guess, and a guessing operator disables the guard.
    """
    out_dir = out_dir.resolve()
    workbook = workbook.resolve()
    if out_dir in workbook.parents or out_dir == workbook.parent:
        raise CorpusError(
            "the workbook %s is inside the agent's corpus directory %s. The "
            "agent would be able to read the answer sheets directly. Move the "
            "workbook outside the corpus directory and re-run."
            % (workbook, out_dir))


def prepare(workbook: Path, out_dir: Path) -> dict:
    """Everything above, in order, returning the manifest that records it."""
    assert_quarantine(out_dir, workbook)
    sheets = read_allowed(workbook)

    tickets, dropped_t = drop_empty_columns(sheets["Tickets"])
    tickets = tickets.loc[:, ~tickets.columns.astype(str).str.startswith("Unnamed")]
    tickets = to_dates(tickets)
    agents, dropped_a = drop_empty_columns(sheets["IT Agents"])
    agents = agents.loc[:, ~agents.columns.astype(str).str.startswith("Unnamed")]

    answers = answers_from_workbook(workbook)
    # No free-text columns in this workbook -- see leak_scan's docstring.
    # suspected_text_columns() raises if that ever stops being true.
    leaks = leak_scan(tickets, answers) + leak_scan(agents, answers)
    if leaks:
        raise CorpusError(
            "%d answer value(s) from the quarantined sheets appear inside cells "
            "of the corpus: %s. The corpus is not written. Investigate before "
            "relaxing this -- an answer inside the evidence is the defect that "
            "invalidated IA-45."
            % (len(leaks), leaks[:5]))

    out_dir.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(out_dir / "tickets.csv", index=False)
    agents.to_csv(out_dir / "agents.csv", index=False)

    manifest = {
        "source_workbook": workbook.name,
        "sheets_admitted": list(ALLOWED_SHEETS),
        "sheets_quarantined": quarantined_sheets(workbook),
        "tickets_rows": int(len(tickets)),
        "agents_rows": int(len(agents)),
        "columns_dropped_all_null": {"Tickets": dropped_t, "IT Agents": dropped_a},
        "date_epoch": str(EXCEL_EPOCH.date()),
        "date_range": [str(tickets["Fecha"].min().date()),
                       str(tickets["Fecha"].max().date())],
        "imputation_findings": imputation_report(sheets["Tickets"]),
        # Recorded on every prepare, not only when it fires: a corpus that
        # cannot show it was leak-scanned is a corpus whose blinding nobody can
        # verify later. IA-62 applied to the data instead of the console.
        "leak_scan": {"candidates_checked": len(answers),
                      "hits": leaks},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="run the containment checks and report, write nothing")
    args = ap.parse_args(argv)

    workbook, out_dir = Path(args.workbook), Path(args.out)
    try:
        if args.check_only:
            assert_quarantine(out_dir, workbook)
            held = quarantined_sheets(workbook)
            print("containment OK. %d sheet(s) held back: %s"
                  % (len(held), ", ".join(held)))
            return 0
        manifest = prepare(workbook, out_dir)
    except CorpusError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    print("corpus -> %s" % out_dir)
    print("  %d tickets, %d agents, %s to %s"
          % (manifest["tickets_rows"], manifest["agents_rows"],
             *manifest["date_range"]))
    print("  held back: %s" % ", ".join(manifest["sheets_quarantined"]))
    for finding in manifest["imputation_findings"]:
        if finding["counts_identical"] and not finding["filled_any_null"]:
            print("  NOTE: %r imputes nothing -- identical counts, no null filled."
                  % finding["cleaned_column"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
