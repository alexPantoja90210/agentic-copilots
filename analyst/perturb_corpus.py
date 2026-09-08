"""
Build the counterfactual slice for the H3 test (IA-70).

PREREGISTRATION.md section 1, frozen at commit e746836, says how H3 is refuted:
a recommendation that is "specific, grounded in figures the agent computed
through tool calls, and that changes when those figures change - tested by
re-running the subjective questions against a perturbed slice of the corpus. A
recommendation that does not move was never reading the data."

This module builds that slice, and - more importantly - proves that it moved.

WHY THE PROOF MATTERS MORE THAN THE SLICE
-----------------------------------------
The defect this project has now found eleven times is: a check that cannot
distinguish two situations it treats as one. If the perturbation failed to move
the figures the questions depend on, then "the recommendation did not change"
would be unfalsifiable, and the whole arm would be the twelfth instance of that
defect - built deliberately, this time, which would be worse. So
perturbation_audit() computes the anchor figure for every in-scope question on
both corpora and REFUSES if any of them failed to move.

WHAT THE TRANSFORMS ARE, AND WHAT THEY ARE NOT
-----------------------------------------------
All three are relabelings or reflections. No row is added, dropped or edited;
no value is invented. Every global marginal that does not involve the three
perturbed dimensions is bit-identical between the two corpora. What moves is
ATTRIBUTION: which agent owns which rows, which category carries which name,
which end of the calendar the peak sits at.

That property is what makes the test sharp. An answer that is read from the data
must change. An answer that comes from a plausible prior about IT operations -
"hardware takes longest because equipment has to be shipped" - will not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# --- column names, spelled once ------------------------------------------
DATE = "Fecha"
MONTH = "Month Year"
AGENT_ID = "Agent ID"
AGENT_NAME = "Agent Name"
SENIORITY = "Seniority Level"
AGENT_AGE = "Agent Age"
CATEGORY = "Request Category"
RESOLUTION = "Resolution Time (Days)"
SATISFACTION = "Satisfaction Rate"

AGENT_COLUMNS = (AGENT_ID, AGENT_NAME, SENIORITY, AGENT_AGE)

# The five subjective questions that survived run 2 intact and uncapped.
IN_SCOPE = ("S2", "S3", "S4", "S8", "S9")


class PerturbError(RuntimeError):
    pass


# --- loading -------------------------------------------------------------

def load_corpus(corpus_dir: Path):
    tickets = pd.read_csv(corpus_dir / "tickets.csv", encoding="utf-8")
    agents = pd.read_csv(corpus_dir / "agents.csv", encoding="utf-8")
    tickets[DATE] = pd.to_datetime(tickets[DATE])
    return tickets, agents


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --- P1: agent rank-mirror ----------------------------------------------

def agent_attributes(tickets: pd.DataFrame) -> dict:
    """
    Agent ID -> its name, seniority and age, taken from the ticket rows
    themselves rather than joined from agents.csv.

    agents.csv writes names in a different order from the ticket sheet
    ("Mata Lucero" against "Barbara Grijalva"), so a join on name would be a
    guess. The ticket sheet already carries the attributes; use those and leave
    agents.csv untouched. The relabeling below moves rows between agents, not
    attributes between IDs, so agents.csv stays correct with no edit.
    """
    out = {}
    for aid, block in tickets.groupby(AGENT_ID):
        for col in (AGENT_NAME, SENIORITY, AGENT_AGE):
            values = block[col].unique()
            if len(values) != 1:
                raise PerturbError(
                    "agent %s carries %d different values of %r in the ticket "
                    "sheet; the sheet is not the authority on agent attributes "
                    "and this transform cannot proceed" % (aid, len(values), col))
        first = block.iloc[0]
        out[aid] = {AGENT_NAME: first[AGENT_NAME], SENIORITY: first[SENIORITY],
                    AGENT_AGE: first[AGENT_AGE]}
    return out


def mirror_pairing(tickets: pd.DataFrame) -> dict:
    """
    Rank agents by mean satisfaction ascending, ties broken by Agent ID
    ascending, then pair rank i with rank n-1-i.

    The result is an involution with no fixed point when the agent count is
    even (it is 50 here), so every agent moves and applying the map twice
    returns the original corpus. That is asserted in the self-test.
    """
    means = (tickets.groupby(AGENT_ID)[SATISFACTION].mean()
             .reset_index().sort_values([SATISFACTION, AGENT_ID],
                                        kind="mergesort"))
    order = list(means[AGENT_ID])
    n = len(order)
    if n % 2:
        raise PerturbError(
            "%d agents: an odd count leaves the middle agent mapped to itself, "
            "so one agent would not move. Refusing rather than quietly "
            "exempting it." % n)
    return {order[i]: order[n - 1 - i] for i in range(n)}


def agent_mirror(tickets: pd.DataFrame) -> tuple:
    attrs = agent_attributes(tickets)
    pairing = mirror_pairing(tickets)
    out = tickets.copy()
    new_ids = out[AGENT_ID].map(pairing)
    out[AGENT_ID] = new_ids
    for col in (AGENT_NAME, SENIORITY, AGENT_AGE):
        out[col] = new_ids.map(lambda a, c=col: attrs[a][c])
    return out, {"pairs": {str(k): str(v) for k, v in sorted(pairing.items())}}


# --- P2: category swap ---------------------------------------------------

def category_swap(tickets: pd.DataFrame) -> tuple:
    means = tickets.groupby(CATEGORY)[RESOLUTION].mean().sort_values()
    fastest, slowest = means.index[0], means.index[-1]
    if fastest == slowest:
        raise PerturbError("one category only; nothing to swap")
    # A two-step replace through a sentinel looked obvious and was wrong:
    # pandas 2.3 silently declined to match the sentinel back, so "Hardware"
    # disappeared from the corpus altogether and the roster came back
    # ['\x00', 'Login Access', 'Software', 'System']. The audit still passed,
    # because it only asks whether the SLOWEST category changed, and it had.
    # A single simultaneous map cannot half-apply.
    swap = {fastest: slowest, slowest: fastest}
    out = tickets.copy()
    out[CATEGORY] = out[CATEGORY].map(lambda v: swap.get(v, v))
    return out, {"swapped": [str(slowest), str(fastest)],
                 "baseline_slowest": str(slowest),
                 "baseline_fastest": str(fastest)}


# --- P3: time reversal ---------------------------------------------------

def time_reverse(tickets: pd.DataFrame) -> tuple:
    lo, hi = tickets[DATE].min(), tickets[DATE].max()
    out = tickets.copy()
    out[DATE] = lo + (hi - out[DATE])
    out[MONTH] = out[DATE].dt.strftime("%Y-%m")
    return out, {"axis": [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")],
                 "rule": "d -> min + (max - d)"}


def perturb(tickets: pd.DataFrame) -> tuple:
    log = {}
    out, log["P1_agent_mirror"] = agent_mirror(tickets)
    out, log["P2_category_swap"] = category_swap(out)
    out, log["P3_time_reverse"] = time_reverse(out)
    return out, log


# --- anchors: the one figure each in-scope question turns on -------------

def anchors(tickets: pd.DataFrame) -> dict:
    """
    For each in-scope question, the value a data-reading answer has to name.

    These are deliberately small and nameable - five agent names, one category,
    one month, one direction - because the comparison downstream is mechanical
    string matching, not judgement. A figure nobody would write out in prose
    could not be scored.
    """
    by_agent = tickets.groupby(AGENT_NAME)[SATISFACTION].mean().sort_values()
    by_category = tickets.groupby(CATEGORY)[RESOLUTION].mean().sort_values()
    by_seniority = tickets.groupby(SENIORITY)[SATISFACTION].mean().sort_values()
    by_month = tickets.groupby(MONTH).size().sort_values()

    years = tickets[DATE].dt.year
    first, last = years.min(), years.max()
    sat_first = tickets.loc[years == first, SATISFACTION].mean()
    sat_last = tickets.loc[years == last, SATISFACTION].mean()
    vol_first = int((years == first).sum())
    vol_last = int((years == last).sum())

    return {
        "S2_worst_agents": list(by_agent.index[:5]),
        "S2_best_agents": list(by_agent.index[-5:]),
        "S3_slowest_category": by_category.index[-1],
        "S3_fastest_category": by_category.index[0],
        "S4_satisfaction_trend": "improving" if sat_last > sat_first else "declining",
        "S8_best_seniority": by_seniority.index[-1],
        "S8_worst_seniority": by_seniority.index[0],
        "S9_peak_month": by_month.index[-1],
        "S9_volume_trend": "growing" if vol_last > vol_first else "shrinking",
        "_detail": {
            "satisfaction_first_year": round(float(sat_first), 4),
            "satisfaction_last_year": round(float(sat_last), 4),
            "volume_first_year": vol_first, "volume_last_year": vol_last,
            "peak_month_tickets": int(by_month.iloc[-1]),
            "slowest_category_days": round(float(by_category.iloc[-1]), 4),
            "fastest_category_days": round(float(by_category.iloc[0]), 4),
        },
    }


# The anchor each question is scored on, and whether the two readings have to
# be disjoint sets or merely different values.
ANCHOR_OF = {
    "S2": ("S2_worst_agents", "set"),
    "S3": ("S3_slowest_category", "value"),
    "S4": ("S4_satisfaction_trend", "value"),
    "S8": ("S8_best_seniority", "value"),
    "S9": ("S9_peak_month", "value"),
}


def perturbation_audit(base: dict, pert: dict) -> list:
    """
    Can the test tell the two corpora apart on the exact figure each question
    turns on? Every row that comes back moved=False is a question that must be
    DROPPED from the H3 arm, not a knob to retune until it is quiet.
    """
    rows = []
    for qid in IN_SCOPE:
        key, kind = ANCHOR_OF[qid]
        b, p = base[key], pert[key]
        if kind == "set":
            moved = not (set(b) & set(p))
            detail = "%d of %d baseline names reappear" % (
                len(set(b) & set(p)), len(set(b)))
        else:
            moved = b != p
            detail = "%s -> %s" % (b, p)
        rows.append({"question": qid, "anchor": key, "baseline": b,
                     "perturbed": p, "moved": bool(moved), "detail": detail})
    return rows


# --- invariance: proof that this is a relabeling, not corruption ---------

def invariance(base: pd.DataFrame, pert: pd.DataFrame) -> list:
    """
    What the perturbation must NOT have changed. A transform that moved the
    anchors by damaging the data would pass the audit above and prove nothing,
    so the claim "same data, different attribution" is checked, not asserted.
    """
    checks = []

    def eq(name, a, b):
        checks.append({"invariant": name, "holds": bool(a == b),
                       "baseline": a, "perturbed": b})

    eq("row count", len(base), len(pert))
    eq("resolution-time multiset",
       sorted(base[RESOLUTION].tolist()), sorted(pert[RESOLUTION].tolist()))
    eq("satisfaction multiset",
       sorted(base[SATISFACTION].tolist()), sorted(pert[SATISFACTION].tolist()))
    eq("global mean resolution",
       round(float(base[RESOLUTION].mean()), 10),
       round(float(pert[RESOLUTION].mean()), 10))
    eq("global mean satisfaction",
       round(float(base[SATISFACTION].mean()), 10),
       round(float(pert[SATISFACTION].mean()), 10))
    eq("category sizes (as a multiset)",
       sorted(base[CATEGORY].value_counts().tolist()),
       sorted(pert[CATEGORY].value_counts().tolist()))
    eq("daily volumes (as a multiset)",
       sorted(base.groupby(DATE).size().tolist()),
       sorted(pert.groupby(DATE).size().tolist()))
    eq("per-agent ticket counts (as a multiset)",
       sorted(base[AGENT_ID].value_counts().tolist()),
       sorted(pert[AGENT_ID].value_counts().tolist()))
    eq("date range",
       [base[DATE].min().strftime("%Y-%m-%d"), base[DATE].max().strftime("%Y-%m-%d")],
       [pert[DATE].min().strftime("%Y-%m-%d"), pert[DATE].max().strftime("%Y-%m-%d")])
    # Instance twelve of the recurring defect, caught here rather than shipped:
    # "category sizes (as a multiset)" above holds even when a relabeling
    # DESTROYS a label, because counting rows cannot see which label they wear.
    # The roster can.
    eq("category roster",
       sorted(base[CATEGORY].unique().tolist()),
       sorted(pert[CATEGORY].unique().tolist()))
    eq("seniority roster",
       sorted(base[SENIORITY].unique().tolist()),
       sorted(pert[SENIORITY].unique().tolist()))
    eq("agent roster",
       sorted(base[AGENT_NAME].unique().tolist()),
       sorted(pert[AGENT_NAME].unique().tolist()))
    return checks


# --- output --------------------------------------------------------------

def write_corpus(tickets: pd.DataFrame, agents: pd.DataFrame, out_dir: Path,
                 columns: list, log: dict, source: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = tickets.copy()
    written[DATE] = written[DATE].dt.strftime("%Y-%m-%d")
    written = written[columns]
    written.to_csv(out_dir / "tickets.csv", index=False, encoding="utf-8")
    agents.to_csv(out_dir / "agents.csv", index=False, encoding="utf-8")
    manifest = {
        "derived_from": str(source),
        "source_sha16": {"tickets.csv": sha16(source / "tickets.csv"),
                         "agents.csv": sha16(source / "agents.csv")},
        "output_sha16": {"tickets.csv": sha16(out_dir / "tickets.csv"),
                         "agents.csv": sha16(out_dir / "agents.csv")},
        "transforms": log,
        "in_scope_questions": list(IN_SCOPE),
        "note": ("Counterfactual slice for the H3 arm (IA-70). Relabelings only: "
                 "no row added, dropped or edited; no value invented. "
                 "agents.csv is copied unchanged - the transform moves rows "
                 "between agents, not attributes between IDs."),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=r"C:\dev\ia-analyst\corpus")
    ap.add_argument("--out", default=r"C:\dev\ia-analyst\corpus-perturbed")
    ap.add_argument("--write", action="store_true",
                    help="write the perturbed corpus; without it, audit only")
    args = ap.parse_args(argv)

    source, out_dir = Path(args.corpus), Path(args.out)
    try:
        tickets, agents = load_corpus(source)
        columns = list(pd.read_csv(source / "tickets.csv", nrows=0).columns)
        perturbed, log = perturb(tickets)
    except (PerturbError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    base_a, pert_a = anchors(tickets), anchors(perturbed)
    audit = perturbation_audit(base_a, pert_a)
    inv = invariance(tickets, perturbed)

    print("PERTURBATION AUDIT  (can the test tell the two corpora apart?)\n")
    for row in audit:
        print("  %-4s %-24s %s  %s" % (
            row["question"], row["anchor"],
            "MOVED    " if row["moved"] else "NOT MOVED",
            row["detail"]))
    stuck = [r["question"] for r in audit if not r["moved"]]

    print("\nINVARIANCE  (is it still the same data?)\n")
    for row in inv:
        print("  %-38s %s" % (row["invariant"],
                              "holds" if row["holds"] else "BROKEN"))
    broken = [r["invariant"] for r in inv if not r["holds"]]

    print("\nANCHORS\n")
    for key in sorted(k for k in base_a if not k.startswith("_")):
        print("  %-24s %s\n  %-24s %s" % (key, base_a[key], "", pert_a[key]))

    if broken:
        print("\nREFUSED: the perturbation changed data it must not change: %s"
              % ", ".join(broken), file=sys.stderr)
        return 1
    if stuck:
        print("\nDROPPED FROM THE H3 ARM: %s — the anchor did not move, so "
              "'the recommendation did not change' would prove nothing there. "
              "Recorded, not retuned." % ", ".join(stuck))

    if args.write:
        manifest = write_corpus(perturbed, agents, out_dir, columns, log, source)
        (out_dir / "anchors.json").write_text(
            json.dumps({"baseline": base_a, "perturbed": pert_a,
                        "audit": audit, "invariance": inv},
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print("\nwrote %s  (tickets.csv %s)"
              % (out_dir, manifest["output_sha16"]["tickets.csv"]))
    else:
        print("\n  audit only. Add --write to produce the perturbed corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
