"""
Selftest for the corpus preparation and the containment around it.

The property under test is not "the data loaded". It is that the agent cannot
reach the answers, and that nothing was changed in the corpus without saying so.

No network, no key, no spend.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_corpus as pc

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def run(workbook: Path) -> int:
    scratch = Path(tempfile.mkdtemp())
    out = scratch / "corpus"

    # ---- containment: the workbook may not live where the agent can read it --
    bad = scratch / "bad"
    bad.mkdir()
    planted = bad / workbook.name
    shutil.copy(workbook, planted)
    try:
        pc.assert_quarantine(bad, planted)
        check("a workbook inside the corpus directory is refused", False,
              "it was allowed, and the agent could read the answer sheets")
    except pc.CorpusError as exc:
        check("a workbook inside the corpus directory is refused", True)
        check("and the refusal names both paths, so the operator need not guess",
              str(planted) in str(exc) and str(bad) in str(exc), str(exc))

    manifest = pc.prepare(workbook, out)
    tickets_early = pd.read_csv(out / "tickets.csv")

    # ---- the allowlist actually held back the answers ----------------------
    held = manifest["sheets_quarantined"]
    check("the answer sheets are held back",
          "Objective Answers" in held and "Subjective Answers" in held, str(held))
    check("and so are the dashboards, which carry the same answers as charts",
          sum(1 for s in held if s.startswith("Dashboard")) >= 4, str(held))
    check("only the two permitted sheets were written",
          sorted(p.name for p in out.glob("*.csv")) == ["agents.csv", "tickets.csv"],
          str(sorted(p.name for p in out.glob("*.csv"))))

    tickets = tickets_early
    agents = pd.read_csv(out / "agents.csv")

    # ---- the corpus does not contain the answers ---------------------------
    check("the leak scan ran and is recorded in the manifest, fired or not",
          manifest["leak_scan"]["candidates_checked"] > 0,
          str(manifest["leak_scan"]))
    check("no answer value from the quarantined sheets is inside a corpus cell",
          manifest["leak_scan"]["hits"] == [], str(manifest["leak_scan"]["hits"]))

    # The scan is per CELL, and this test is why. Reading the serialised CSV
    # reported a leak of the human's peak-month figure 2,609, found inside
    # "...,2017-02-02,609,22,..." -- a date abutting an employee id, joined by
    # the delimiter. The number was never in the data.
    #
    # This asserts the false positive stays fixed. A leak detector that fires on
    # serialisation artefacts trains the operator to ignore it, and on the day
    # it is right it is ignored too.
    serialised = (out / "tickets.csv").read_text(encoding="utf-8",
                                                 errors="ignore")[:4_000_000]
    check("the artefact that fooled the first version is still present in the "
          "serialised text, so this test is testing something",
          "2,609" in serialised)
    check("and the per-cell scan does not report it",
          not any(h["answer"] == "2609" for h in manifest["leak_scan"]["hits"]),
          str(manifest["leak_scan"]["hits"]))

    # ---- the prose heuristic distinguishes labels from prose ---------------
    # It flagged Issue Type, Severity, Priority and Agent Name in its first
    # version, on the theory that a space means prose. It does not.
    suspected = pc.suspected_text_columns(tickets_early)
    check("multi-word categorical labels are not mistaken for free prose",
          not {"Issue Type", "Severity", "Priority", "Agent Name"} & set(suspected),
          str(suspected))
    prose = tickets_early.head(200).copy()
    prose["description"] = ("Users cannot log in after the SSO change. Stack "
                            "trace attached. Affects 400 users in EMEA. ")
    prose["description"] += prose.index.astype(str)
    check("but a real prose column IS detected",
          "description" in pc.suspected_text_columns(prose),
          str(pc.suspected_text_columns(prose)))
    try:
        pc.leak_scan(prose, ["2609"])
        check("an undeclared prose column stops the scan", False, "it ran anyway")
    except pc.CorpusError as exc:
        check("an undeclared prose column stops the scan", "description" in str(exc))

    # ---- the candidate list holds answers, not data ------------------------
    # It was scraped from the quarantined sheets first, and reported 148 leaks:
    # pivot tables display the inputs beside the outputs, so employee ids came
    # back as "answers". Only DERIVED values can indicate a leak.
    tickets_raw = pd.read_excel(workbook, sheet_name="Tickets", engine="pyxlsb")
    numeric = [c for c in tickets_raw.columns if tickets_raw[c].dtype.kind in "if"]
    collisions = [a for a in pc.DERIVED_ANSWERS
                  if any((tickets_raw[c] == float(a)).any() for c in numeric)]
    check("no scanned answer collides with a legitimate data value",
          collisions == [], str(collisions))
    check("and the answers that DO collide are listed as unscannable, not "
          "silently dropped",
          set(pc.UNSCANNABLE_ANSWERS) >= {"40", "53", "16"},
          str(pc.UNSCANNABLE_ANSWERS))
    check("every scanned answer says what it is, so the list can be reviewed",
          all(len(v) > 10 for v in pc.DERIVED_ANSWERS.values()))

    # ---- exemptions are declared with a reason, not tuned away -------------
    check("Email is exempted by explicit declaration, not by a moved threshold",
          "Email" in pc.DECLARED_NON_TEXT
          and len(pc.DECLARED_NON_TEXT["Email"]) > 40)

    # ---- and the scan can still catch a real leak --------------------------
    planted_frame = tickets_early.head(50).copy()
    planted_frame["notes"] = "peak month was 2609 tickets"
    found = pc.leak_scan(planted_frame, ["2609"], text_columns=("notes",))
    check("a value planted inside a single cell IS caught",
          any(h["column"] == "notes" for h in found), str(found))

    # ---- nothing was changed without saying so -----------------------------
    dropped = manifest["columns_dropped_all_null"]["Tickets"]
    check("the all-null column is dropped AND named in the manifest",
          "Column1" in dropped, str(dropped))
    check("and it is really gone from the corpus", "Column1" not in tickets.columns)

    # ---- dates: the defect that has already appeared three times -----------
    check("the Excel epoch is recorded in the manifest, not left implicit",
          manifest["date_epoch"] == "1899-12-30", manifest["date_epoch"])
    check("a known serial round-trips (42370 = 2016-01-01)",
          str(pd.Timestamp(pc.EXCEL_EPOCH) + pd.Timedelta(days=42370))[:10]
          == "2016-01-01")
    check("the corpus date range is the five full years, not an off-by-one",
          manifest["date_range"] == ["2016-01-01", "2020-12-31"],
          str(manifest["date_range"]))
    check("and every calendar day in that range is present -- no silent gap",
          pd.to_datetime(tickets["Fecha"]).nunique() == 1827,
          str(pd.to_datetime(tickets["Fecha"]).nunique()))

    # ---- the 'Imputated' columns impute nothing ----------------------------
    findings = {f["cleaned_column"]: f for f in manifest["imputation_findings"]}
    for cleaned in ("Severity Imputated", "Priority Imputated"):
        f = findings.get(cleaned, {})
        check("%r fills no null" % cleaned, f.get("filled_any_null") is False, str(f))
        check("%r has identical value counts to its source" % cleaned,
              f.get("counts_identical") is True, str(f))
        check("%r keeps the same number of levels" % cleaned,
              f.get("raw_levels") == f.get("cleaned_levels"), str(f))
    check("both raw and cleaned columns survive into the corpus, so the finding "
          "stays checkable",
          all(c in tickets.columns for c in
              ("Severity", "Severity Imputated", "Priority", "Priority Imputated")))

    # ---- shape, so a silently truncated read cannot pass -------------------
    check("all 97,498 tickets are present", len(tickets) == 97498, str(len(tickets)))
    check("all 50 agents are present", len(agents) == 50, str(len(agents)))

    # ---- a renamed sheet stops the run rather than being guessed at --------
    try:
        pc.read_allowed(Path("does-not-exist.xlsb"))
        check("a missing workbook is refused", False, "it was accepted")
    except pc.CorpusError:
        check("a missing workbook is refused", True)

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python selftest_prepare.py <path to the .xlsb workbook>")
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1])))
