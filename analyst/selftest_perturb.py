"""
Selftest for the H3 counterfactual slice (IA-70).

The property under test is not "a perturbed file was produced". It is that the
perturbation MOVED the figures the in-scope questions turn on, that it moved
NOTHING ELSE, and - the part that actually matters - that both of those checks
can fail. A guard that cannot fail is the defect this project has now found
twelve times, and the twelfth was found here, in this module's own subject:
`category sizes (as a multiset)` holds even when a relabeling destroys a label.
There is a regression test for exactly that below.

Runs on synthetic frames built in this file. No corpus, no network, no key, no
spend.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturb_corpus as pcx

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def frame(n_agents=12, months=24) -> pd.DataFrame:
    """
    A miniature corpus sized so that every in-scope anchor CAN move.

    The sizing is not cosmetic. An earlier fixture had four agents, one month
    and one year, and three anchors could not move for reasons that had nothing
    to do with the code: a single month has no other month to become the peak,
    a single year has no trend to invert, and with four agents the bottom five
    overlap themselves. The test failed on its own shape and would have been
    "fixed" by loosening the transform. So:

      * twelve agents, so the bottom five can be disjoint from the bottom five
        of the mirrored corpus (any count below ten makes that impossible);
      * twenty-four months across two years, so the peak month and both trends
        have somewhere to move;
      * volume rising with time and later months worked by higher-satisfaction
        agents, so the satisfaction trend and the volume trend both point up -
        matching the real corpus, so the transforms are exercised in the same
        direction they will be used.
    """
    rows = []
    base = pd.Timestamp("2019-01-01")
    for m in range(months):
        date = base + pd.DateOffset(months=m)
        aid = 1 + (m * n_agents) // months
        for rep in range(m + 1):          # volume grows with time
            cat = "Hardware" if rep % 2 else "Login Access"
            rows.append({
                pcx.DATE: date,
                pcx.MONTH: date.strftime("%Y-%m"),
                pcx.AGENT_ID: aid,
                pcx.AGENT_NAME: "Agent %02d" % aid,
                pcx.SENIORITY: "Senior" if aid % 2 else "Mid-Level",
                pcx.AGENT_AGE: 30 + aid,
                pcx.CATEGORY: cat,
                pcx.RESOLUTION: 9 if cat == "Hardware" else 1,
                pcx.SATISFACTION: aid,      # agent k has satisfaction k
            })
    return pd.DataFrame(rows)


def run() -> int:
    t = frame()

    # ---- P1: the pairing is a genuine involution with nobody left out -----
    pairing = pcx.mirror_pairing(t)
    check("the mirror pairing is an involution",
          all(pairing[pairing[a]] == a for a in pairing))
    check("no agent is mapped to itself",
          all(pairing[a] != a for a in pairing),
          "a fixed point means one agent does not move and its answer cannot change")
    check("every agent appears exactly once as a target",
          sorted(pairing.values()) == sorted(pairing))

    # ---- P1: the ranking is inverted, not merely shuffled -----------------
    before = list(t.groupby(pcx.AGENT_NAME)[pcx.SATISFACTION].mean().sort_values().index)
    mirrored, _ = pcx.agent_mirror(t)
    after = list(mirrored.groupby(pcx.AGENT_NAME)[pcx.SATISFACTION].mean()
                 .sort_values().index)
    check("the agent ranking is exactly reversed", after == before[::-1],
          "%s -> %s" % (before, after))

    twice, _ = pcx.agent_mirror(mirrored)
    check("applying the mirror twice returns the original",
          twice.sort_index()[list(t.columns)].equals(t.sort_index()[list(t.columns)]))

    # ---- P1 refuses rather than quietly exempting the middle agent --------
    odd = t[t[pcx.AGENT_ID] != 4]  # 12 agents minus one = odd
    try:
        pcx.mirror_pairing(odd)
        check("an odd agent count is refused", False, "it was accepted")
    except pcx.PerturbError:
        check("an odd agent count is refused", True)

    # ---- P1 refuses when the ticket sheet disagrees with itself -----------
    inconsistent = t.copy()
    inconsistent.loc[inconsistent.index[0], pcx.AGENT_NAME] = "Someone Else"
    try:
        pcx.agent_attributes(inconsistent)
        check("an agent with two names in the sheet is refused", False,
              "it was accepted and one of the two names would have been picked")
    except pcx.PerturbError:
        check("an agent with two names in the sheet is refused", True)

    # ---- P2: exactly two labels trade places, and none is lost ------------
    swapped, log2 = pcx.category_swap(t)
    check("the category roster survives the swap",
          sorted(swapped[pcx.CATEGORY].unique()) == sorted(t[pcx.CATEGORY].unique()),
          str(sorted(swapped[pcx.CATEGORY].unique())))
    check("the slow category and the fast category trade places",
          swapped.groupby(pcx.CATEGORY)[pcx.RESOLUTION].mean().idxmax()
          == t.groupby(pcx.CATEGORY)[pcx.RESOLUTION].mean().idxmin())
    twice2, _ = pcx.category_swap(swapped)
    check("applying the category swap twice returns the original",
          twice2[pcx.CATEGORY].equals(t[pcx.CATEGORY]))

    # ---- P3: the calendar is mirrored, and the month column follows -------
    reversed_, _ = pcx.time_reverse(t)
    check("the date range is unchanged by reversal",
          (reversed_[pcx.DATE].min(), reversed_[pcx.DATE].max())
          == (t[pcx.DATE].min(), t[pcx.DATE].max()))
    check("the earliest rows become the latest",
          reversed_.loc[t[pcx.DATE].idxmin(), pcx.DATE] == t[pcx.DATE].max())
    check("Month Year is recomputed from the new date, not carried over",
          (reversed_[pcx.MONTH] == reversed_[pcx.DATE].dt.strftime("%Y-%m")).all(),
          "a stale month column would put tickets in a month they are not in")
    twice3, _ = pcx.time_reverse(reversed_)
    check("applying the reversal twice returns the original",
          twice3[pcx.DATE].equals(t[pcx.DATE]))

    # ---- the anchors read what they claim to read -------------------------
    a = pcx.anchors(t)
    check("the worst-agent anchor is the five lowest-satisfaction agents",
          a["S2_worst_agents"] == ["Agent %02d" % k for k in range(1, 6)],
          str(a["S2_worst_agents"]))
    check("the slowest-category anchor is the slowest category",
          a["S3_slowest_category"] == "Hardware", str(a["S3_slowest_category"]))
    check("the peak-month anchor is the busiest month",
          a["S9_peak_month"] == t.groupby(pcx.MONTH).size().idxmax())

    # ---- THE AUDIT MUST BE ABLE TO SAY NO ---------------------------------
    same = pcx.perturbation_audit(a, a)
    check("an unchanged corpus is reported as NOT MOVED on every question",
          all(not r["moved"] for r in same),
          "the audit passed a perturbation that changed nothing, which would "
          "make 'the recommendation did not move' unfalsifiable")

    partial = dict(a)
    partial["S2_worst_agents"] = a["S2_worst_agents"][:1] + ["Nobody"] * 4
    row = [r for r in pcx.perturbation_audit(a, partial) if r["question"] == "S2"][0]
    check("a partly overlapping agent set counts as NOT MOVED", not row["moved"],
          "one surviving name lets an unmoved answer score as moved")

    full, _ = pcx.perturb(t)
    moved = pcx.perturbation_audit(a, pcx.anchors(full))
    check("the composed perturbation moves every in-scope anchor",
          all(r["moved"] for r in moved),
          ", ".join(r["question"] for r in moved if not r["moved"]))

    # ---- INVARIANCE MUST BE ABLE TO SAY NO --------------------------------
    check("invariance holds between a corpus and its own perturbation",
          all(r["holds"] for r in pcx.invariance(t, full)),
          ", ".join(r["invariant"] for r in pcx.invariance(t, full) if not r["holds"]))

    # the exact bug this module was written after: a relabeling that DESTROYS
    # a label passes the row-count and multiset checks and fails the roster.
    destroyed = t.copy()
    destroyed[pcx.CATEGORY] = destroyed[pcx.CATEGORY].map(
        lambda v: "\x00" if v == "Hardware" else v)
    inv = {r["invariant"]: r["holds"] for r in pcx.invariance(t, destroyed)}
    check("a destroyed category label is caught", not inv["category roster"],
          "the roster check did not fire")
    check("counting rows alone does NOT catch it", inv["category sizes (as a multiset)"],
          "recorded deliberately: this is why the roster check exists")

    edited = t.copy()
    edited.loc[edited.index[0], pcx.RESOLUTION] = 999
    inv2 = {r["invariant"]: r["holds"] for r in pcx.invariance(t, edited)}
    check("an edited resolution time is caught", not inv2["resolution-time multiset"])
    check("an edited value also moves the global mean",
          not inv2["global mean resolution"])

    dropped = t.iloc[:-1]
    inv3 = {r["invariant"]: r["holds"] for r in pcx.invariance(t, dropped)}
    check("a dropped row is caught", not inv3["row count"])

    # ---- what gets written ------------------------------------------------
    scratch = Path(tempfile.mkdtemp())
    src, out = scratch / "corpus", scratch / "perturbed"
    src.mkdir()
    written = t.copy()
    written[pcx.DATE] = written[pcx.DATE].dt.strftime("%Y-%m-%d")
    written.to_csv(src / "tickets.csv", index=False, encoding="utf-8")
    agents = pd.DataFrame({"Agent ID": list(range(1, 13))})
    agents.to_csv(src / "agents.csv", index=False, encoding="utf-8")
    manifest = pcx.write_corpus(full, agents, out, list(t.columns), {}, src)
    check("the manifest fingerprints both the source and the output",
          manifest["source_sha16"]["tickets.csv"]
          != manifest["output_sha16"]["tickets.csv"],
          "identical fingerprints mean nothing was perturbed")
    check("agents.csv is copied through unchanged",
          manifest["source_sha16"]["agents.csv"]
          == manifest["output_sha16"]["agents.csv"],
          "the transform moves rows between agents, not attributes between IDs")
    reread = pd.read_csv(out / "tickets.csv", encoding="utf-8")
    check("the written corpus keeps the source column order",
          list(reread.columns) == list(t.columns))
    check("the written corpus keeps every row", len(reread) == len(t))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
