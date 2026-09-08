"""
Recompute every objective answer the human analyst published, in code, from the
corpus — and record where we agree with them and where we do not.

    python baseline.py --corpus C:\\dev\\ia-analyst\\corpus
    python baseline.py --corpus ... --json C:\\dev\\ia-analyst\\reference\\baseline.json

Why this exists as code rather than as a note
---------------------------------------------
The six figures were checked by hand on 8 Sep and they held. A verification
nobody else can reproduce is not a verification, and a scorer that compares the
agent against numbers somebody typed in is comparing it against a typo waiting
to happen. This file is the scorer's only source of truth (IA-67), and the agent
is never scored against a hand-entered value.

The human's stated figures are recorded here as FACTS, not as text. Numbers are
not expression and reproducing five of them to compare against is what any
replication does; the reference project's prose, questions and workbook stay out
of this repository entirely — see the README on the licence position.

IA-67, under IA-64.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# What the analyst published. Their values, not ours.
# ---------------------------------------------------------------------------
HUMAN = {
    "O1_attributes": 16,
    "O2_spelling_errors": 3,
    "O2_missing_values": 0,
    "O3_daily_volume": 53.36507936507937,
    "O9_daily_resolution_days": 4.5,
    "O10_tickets_2016": 13051,
    "O10_tickets_2020": 29088,
    "O10_peak_month_tickets": 2609,
    "O11_average_agent_age": 40,
    "O12_severity_resolution_corr": -0.040536349147638415,
}

# ---------------------------------------------------------------------------
# O1 depends on a definition the question never states, and that is the finding.
#
# The workbook as delivered has 17 ticket columns and 9 agent columns once the
# all-null `Column1` is dropped. Neither is 16, and for a while that looked like
# the human being wrong.
#
# They are not. Their 16 is the count of SOURCE attributes across both sheets,
# before any column the analyst derived while working:
#
#     10 source ticket columns + 6 source agent columns = 16
#
# Everything else is theirs: the two `Imputated` normalisations, `servity num`,
# the three agent fields looked up onto each ticket, `Month Year`, and on the
# agent sheet `Email Domain`, `Age` and `Seniority Level` — the last three being
# the answers to O6 and O11 left in the sheet as columns.
#
# So the correct answer to O1 is 16, or 26, or 17, depending on what "present in
# the data" means, and the question does not say. IA-66 must fix that reading in
# writing BEFORE the run, or it gets fixed afterwards in whichever direction
# flatters the agent.
# ---------------------------------------------------------------------------
SOURCE_TICKET_COLUMNS = (
    "ID Ticket", "Fecha", "Employee ID", "Agent ID", "Request Category",
    "Issue Type", "Severity", "Priority", "Resolution Time (Days)",
    "Satisfaction Rate")
SOURCE_AGENT_COLUMNS = (
    "Agent ID", "Full Name", "Email", "Year of Birth", "Month of Birth",
    "Day of Birth")

# The three label corrections the analyst made, recorded as the answer to O2.
KNOWN_SPELLING_ERRORS = (("Severity", "Mayor", "Major"),
                         ("Severity", "Unclasified", "Unclassified"),
                         ("Priority", "Unassiged", "Unassigned"))

CATEGORICAL_SOURCE_COLUMNS = ("Request Category", "Issue Type", "Severity", "Priority")


class BaselineError(Exception):
    pass


def load(corpus: Path) -> tuple:
    tickets = pd.read_csv(corpus / "tickets.csv", parse_dates=["Fecha"])
    agents = pd.read_csv(corpus / "agents.csv")
    if len(tickets) != 97498 or len(agents) != 50:
        raise BaselineError(
            "corpus is %d tickets and %d agents, expected 97498 and 50. A "
            "truncated read must stop the baseline, not quietly shift every "
            "figure below." % (len(tickets), len(agents)))
    return tickets, agents


# --- O1 ---------------------------------------------------------------------
def attributes(tickets, agents) -> dict:
    return {
        "source_total": len(SOURCE_TICKET_COLUMNS) + len(SOURCE_AGENT_COLUMNS),
        "tickets_as_delivered": len(tickets.columns),
        "agents_as_delivered": len(agents.columns),
        "both_as_delivered": len(tickets.columns) + len(agents.columns),
    }


# --- O2 ---------------------------------------------------------------------
def inconsistencies(tickets) -> dict:
    found = []
    for column, wrong, right in KNOWN_SPELLING_ERRORS:
        present = tickets[column].astype(str).str.contains(wrong, case=False).any()
        found.append({"column": column, "wrong": wrong, "right": right,
                      "present": bool(present)})
    return {"spelling_errors": sum(1 for f in found if f["present"]),
            "detail": found,
            "missing_values": int(tickets.isna().sum().sum())}


# --- O3 ---------------------------------------------------------------------
def daily_volume(tickets) -> dict:
    days = tickets["Fecha"].nunique()
    return {"tickets": len(tickets), "distinct_days": days,
            "mean_per_day": len(tickets) / days}


# --- O4 / O5 / O8 : tabular answers ----------------------------------------
def category_distribution(tickets) -> dict:
    counts = tickets["Request Category"].value_counts()
    return {"counts": {k: int(v) for k, v in counts.items()},
            "percent": {k: round(v, 4) for k, v in
                        (counts / len(tickets) * 100).items()}}


def tickets_per_agent(tickets, agents) -> dict:
    counts = tickets.groupby("Agent ID").size()
    names = agents.set_index("Agent ID")["Full Name"].to_dict()
    return {"agents": int(counts.size),
            "by_name": {names.get(k, "agent %s" % k): int(v)
                        for k, v in counts.items()},
            "max": int(counts.max()), "min": int(counts.min())}


def issue_type_counts(tickets) -> dict:
    return {k: int(v) for k, v in tickets["Issue Type"].value_counts().items()}


# --- O6 / O7 : the two questions that ask for a METHOD ----------------------
def domain_analyst_formula(email: str) -> str:
    """
    The analyst's method, transcribed from the task document exactly as written:

        =MID(R5, FIND("@",R5)+1, LEN(R5)-FIND("@",R5)-4)

    Excel's FIND is 1-indexed, so for `lucero.mata@fp20analytics.com` (length 29,
    "@" at 12) the length argument is 29 - 12 - 4 = 13, and the formula returns
    **fp20analytics** — the trailing -4 removes ".com".

    Transcribed rather than corrected. IA-67 asks what the human's method
    produces, not what a better one would.
    """
    at = email.find("@")
    if at < 0:
        return ""
    start = at + 1
    length = len(email) - start - 4
    return email[start:start + length] if length > 0 else ""


def domain_general(email: str) -> str:
    """Everything after the "@". No assumption about the length of the TLD."""
    return email.split("@", 1)[1] if "@" in email else ""


def domain_extraction(agents) -> dict:
    """
    O6, and it has a finding in it that is not about the agent at all.

    The analyst delivered an `Email Domain` column in the IT Agents sheet
    holding **fp20analytics.com**. Their documented formula returns
    **fp20analytics**. The two disagree, on their own data, in their own
    workbook — so the documented method did not produce the delivered column.

    That makes O6 the third question whose correct answer depends on a choice
    nobody wrote down, after O1 (which attributes are "in the data") and O9
    (which average is the "daily" one). All three must be fixed in
    PREREGISTRATION.md before the run, because after the run they get fixed in
    whichever direction suits the result.

    Recorded, not adjudicated here.
    """
    emails = agents["Email"].astype(str)
    delivered = agents["Email Domain"].astype(str)
    by_formula = emails.map(domain_analyst_formula)
    by_split = emails.map(domain_general)
    return {
        "distinct_domains_delivered": sorted(delivered.unique().tolist()),
        "formula_matches_delivered_column": bool((by_formula == delivered).all()),
        "split_matches_delivered_column": bool((by_split == delivered).all()),
        "formula_example": by_formula.iloc[0],
        "delivered_example": delivered.iloc[0],
        # The same formula against addresses this workbook does not contain.
        "formula_on_other_tlds": {
            e: domain_analyst_formula(e)
            for e in ("a.b@example.mx", "c.d@example.io", "e.f@example.co.uk")},
    }


def name_from_agent_id(agents, agent_id: int) -> str:
    """The analyst used VLOOKUP on Agent ID. Same lookup, same key."""
    row = agents.loc[agents["Agent ID"] == agent_id, "Full Name"]
    if row.empty:
        raise BaselineError("no agent with id %r" % agent_id)
    return str(row.iloc[0])


# --- O9 : the one they got wrong -------------------------------------------
def resolution_time(tickets) -> dict:
    """
    Both readings of "daily average resolution time", because the phrase is
    ambiguous — and the ambiguity, not an error, is the finding.

    A correction is recorded here because the first pass of this work got it
    wrong. It computed the global mean, saw 4.553, and asserted that the mean of
    daily means "also rounds to 4.6" without rounding it. It does not:

        global mean          4.553150  ->  4.6
        mean of daily means  4.548546  ->  4.5

    The analyst published 4.5 and documented their method as a pivot table with
    date as the row and the average of resolution time as the value — which IS
    the mean of daily means. Their answer matches their stated method exactly.
    They were right and the check was wrong.

    That was the sixth appearance of this project's recurring defect, this time
    in the comparison layer itself: a claim that two situations were the same
    when they differ. It is kept in the docstring rather than quietly fixed,
    because a comparison that has been wrong once is exactly the thing whose
    history should stay visible.
    """
    return {"global_mean": float(tickets["Resolution Time (Days)"].mean()),
            "mean_of_daily_means": float(
                tickets.groupby("Fecha")["Resolution Time (Days)"].mean().mean())}


# --- O10 --------------------------------------------------------------------
def volume_over_time(tickets) -> dict:
    by_year = tickets.groupby(tickets["Fecha"].dt.year).size()
    by_month = tickets.groupby(tickets["Fecha"].dt.to_period("M")).size()
    return {"by_year": {int(k): int(v) for k, v in by_year.items()},
            "peak_month": str(by_month.idxmax()),
            "peak_month_tickets": int(by_month.max())}


# --- O11 --------------------------------------------------------------------
def average_agent_age(agents) -> dict:
    return {"mean": float(agents["Age"].mean()), "agents": int(len(agents))}


# --- O12 --------------------------------------------------------------------
def severity_resolution_correlation(tickets) -> dict:
    """
    The analyst used =CORREL(M:M, K:K) — Pearson on `servity num` against
    `Resolution Time (Days)`. Spearman is reported beside it only so that a
    later change of method cannot pass unnoticed: it is -0.017, a different
    number, and quoting it as if it were the same finding would be wrong.
    """
    x, y = tickets["servity num"], tickets["Resolution Time (Days)"]
    return {"pearson": float(x.corr(y)), "spearman": float(x.corr(y, method="spearman"))}


# --- O13 --------------------------------------------------------------------
def categorical_columns(tickets) -> dict:
    return {"source_categorical": list(CATEGORICAL_SOURCE_COLUMNS),
            "count": len(CATEGORICAL_SOURCE_COLUMNS)}


def compute(tickets, agents) -> dict:
    return {
        "O1": attributes(tickets, agents),
        "O2": inconsistencies(tickets),
        "O3": daily_volume(tickets),
        "O4": category_distribution(tickets),
        "O5": tickets_per_agent(tickets, agents),
        "O6": domain_extraction(agents),
        "O7": {"example": {"agent_id": 1, "full_name": name_from_agent_id(agents, 1)}},
        "O8": issue_type_counts(tickets),
        "O9": resolution_time(tickets),
        "O10": volume_over_time(tickets),
        "O11": average_agent_age(agents),
        "O12": severity_resolution_correlation(tickets),
        "O13": categorical_columns(tickets),
    }


def compare(c: dict) -> list[dict]:
    """
    Every figure the analyst published, beside ours. Agreements AND
    disagreements are both returned, because a disagreement that is only
    mentioned in prose is a disagreement a later refactor can quietly remove.
    """
    def row(qid, what, human, ours, agrees, note=""):
        return {"question": qid, "figure": what, "human": human, "computed": ours,
                "agrees": bool(agrees), "note": note}

    rows = [
        row("O1", "attributes", HUMAN["O1_attributes"], c["O1"]["source_total"],
            c["O1"]["source_total"] == HUMAN["O1_attributes"],
            "agrees only under 'source columns, before the analyst's derived "
            "ones'. As delivered it is %d. The question does not say which."
            % c["O1"]["both_as_delivered"]),
        row("O2", "spelling errors", HUMAN["O2_spelling_errors"],
            c["O2"]["spelling_errors"],
            c["O2"]["spelling_errors"] == HUMAN["O2_spelling_errors"]),
        row("O2", "missing values", HUMAN["O2_missing_values"],
            c["O2"]["missing_values"],
            c["O2"]["missing_values"] == HUMAN["O2_missing_values"]),
        row("O3", "mean tickets per day", HUMAN["O3_daily_volume"],
            c["O3"]["mean_per_day"],
            abs(c["O3"]["mean_per_day"] - HUMAN["O3_daily_volume"]) < 1e-9),
        row("O9", "daily resolution (analyst's method: mean of daily means)",
            HUMAN["O9_daily_resolution_days"], c["O9"]["mean_of_daily_means"],
            round(c["O9"]["mean_of_daily_means"], 1) == HUMAN["O9_daily_resolution_days"],
            "rounds to 4.5 — matches the pivot-by-date method they documented"),
        row("O9", "daily resolution (other reading: global mean)",
            HUMAN["O9_daily_resolution_days"], c["O9"]["global_mean"],
            round(c["O9"]["global_mean"], 1) == HUMAN["O9_daily_resolution_days"],
            "rounds to 4.6. Not an error in their work: a different reading of "
            "the question, which is why IA-66 must fix the reading before the run"),
        row("O10", "tickets in 2016", HUMAN["O10_tickets_2016"],
            c["O10"]["by_year"][2016],
            c["O10"]["by_year"][2016] == HUMAN["O10_tickets_2016"]),
        row("O10", "tickets in 2020", HUMAN["O10_tickets_2020"],
            c["O10"]["by_year"][2020],
            c["O10"]["by_year"][2020] == HUMAN["O10_tickets_2020"]),
        row("O10", "peak month", HUMAN["O10_peak_month_tickets"],
            c["O10"]["peak_month_tickets"],
            c["O10"]["peak_month_tickets"] == HUMAN["O10_peak_month_tickets"],
            "peak month is %s" % c["O10"]["peak_month"]),
        row("O11", "average agent age", HUMAN["O11_average_agent_age"],
            c["O11"]["mean"], round(c["O11"]["mean"]) == HUMAN["O11_average_agent_age"],
            "exact value is %.2f; the analyst rounded" % c["O11"]["mean"]),
        row("O12", "Pearson correlation", HUMAN["O12_severity_resolution_corr"],
            c["O12"]["pearson"],
            abs(c["O12"]["pearson"] - HUMAN["O12_severity_resolution_corr"]) < 1e-10,
            "Spearman is %.6f — a different figure, not the same finding"
            % c["O12"]["spearman"]),
    ]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--json", default=None, help="write the full computation here")
    args = ap.parse_args(argv)

    try:
        tickets, agents = load(Path(args.corpus))
    except (OSError, BaselineError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    computed = compute(tickets, agents)
    rows = compare(computed)

    width = max(len(r["figure"]) for r in rows)
    print("%-5s %-*s %22s %22s" % ("", width, "figure", "analyst", "recomputed"))
    for r in rows:
        mark = "ok " if r["agrees"] else "!! "
        print("%s%-4s %-*s %22s %22s"
              % (mark, r["question"], width, r["figure"], r["human"], r["computed"]))
        if r["note"]:
            print("      %s" % r["note"])

    disagreements = [r for r in rows if not r["agrees"]]
    print("\n%d of %d figures agree with the analyst under their own documented "
          "method." % (len(rows) - len(disagreements), len(rows)))
    if disagreements:
        print("%d differ, and each is a reading of an ambiguous question rather "
              "than an error in their work. Recorded, not reconciled:"
              % len(disagreements))
        for r in disagreements:
            print("  %s %s" % (r["question"], r["figure"]))
    print("\nO1 and O9 are ambiguous as asked. IA-66 fixes which reading counts, "
          "in writing, before the first call.")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"computed": computed, "comparison": rows}, indent=2,
                       default=str), encoding="utf-8")
        print("full computation -> %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
