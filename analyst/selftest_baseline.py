"""
Selftest for the recomputed baseline.

The property under test is not "the numbers came out". It is that this file can
be trusted as the scorer's only source of truth — which means every figure the
analyst published is asserted against ours, agreements and disagreements alike,
and no later refactor can quietly move either.

No network, no key, no spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as bl

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def run(corpus: Path) -> int:
    tickets, agents = bl.load(corpus)
    c = bl.compute(tickets, agents)
    rows = {(r["question"], r["figure"]): r for r in bl.compare(c)}

    # ---- the corpus is the one these figures were computed against ---------
    check("the baseline refuses a corpus of the wrong size",
          _refuses_truncated(corpus))

    # ---- O1: three readings, and the analyst's is one of them --------------
    check("O1 source-column count reproduces the analyst's 16",
          c["O1"]["source_total"] == 16, str(c["O1"]))
    check("and the corpus as delivered is NOT 16, which is why O1 is ambiguous",
          c["O1"]["both_as_delivered"] != 16, str(c["O1"]))
    check("both readings are recorded, so the ambiguity cannot be lost",
          {"source_total", "both_as_delivered"} <= set(c["O1"]))

    # ---- O2 ----------------------------------------------------------------
    check("all three spelling errors are still present in the corpus",
          c["O2"]["spelling_errors"] == 3, str(c["O2"]["detail"]))
    check("and there are no missing values anywhere",
          c["O2"]["missing_values"] == 0, str(c["O2"]["missing_values"]))

    # ---- O3 ----------------------------------------------------------------
    check("mean tickets per day matches to 1e-9",
          abs(c["O3"]["mean_per_day"] - bl.HUMAN["O3_daily_volume"]) < 1e-9,
          "%r" % c["O3"]["mean_per_day"])
    check("and it is computed over 1,827 distinct days",
          c["O3"]["distinct_days"] == 1827, str(c["O3"]["distinct_days"]))

    # ---- O9: the correction. The analyst was right; the first check was not -
    daily = c["O9"]["mean_of_daily_means"]
    glob = c["O9"]["global_mean"]
    check("the analyst's own method (mean of daily means) rounds to their 4.5",
          round(daily, 1) == 4.5, "%r -> %r" % (daily, round(daily, 1)))
    check("the other reading (global mean) rounds to 4.6, and they are different",
          round(glob, 1) == 4.6 and round(daily, 1) != round(glob, 1),
          "%r vs %r" % (glob, daily))
    # This is the check the first version of the comparison did not make. It
    # asserted both readings rounded to 4.6 without rounding the second one,
    # and reported the analyst as wrong. They were not.
    check("the two readings are asserted separately, never collapsed into one "
          "claim about both",
          ("O9", "daily resolution (analyst's method: mean of daily means)") in rows
          and ("O9", "daily resolution (other reading: global mean)") in rows,
          str(sorted(k[1] for k in rows if k[0] == "O9")))
    check("and the disagreement is labelled a reading, not an error in their work",
          "Not an error in their work" in
          rows[("O9", "daily resolution (other reading: global mean)")]["note"])

    # ---- O10 ---------------------------------------------------------------
    check("tickets in 2016 match", c["O10"]["by_year"][2016] == 13051)
    check("tickets in 2020 match", c["O10"]["by_year"][2020] == 29088)
    check("the peak month is December 2020 with 2,609",
          c["O10"]["peak_month"] == "2020-12" and c["O10"]["peak_month_tickets"] == 2609,
          str(c["O10"]))

    # ---- O11 ---------------------------------------------------------------
    check("average agent age rounds to the analyst's 40",
          round(c["O11"]["mean"]) == 40, "%r" % c["O11"]["mean"])
    check("and the unrounded value is kept, so the rounding stays visible",
          abs(c["O11"]["mean"] - 40.06) < 1e-9, "%r" % c["O11"]["mean"])

    # ---- O12: method matters more than the number --------------------------
    check("Pearson matches the analyst's CORREL to 10 decimal places",
          abs(c["O12"]["pearson"] - bl.HUMAN["O12_severity_resolution_corr"]) < 1e-10,
          "%r" % c["O12"]["pearson"])
    check("Spearman is a DIFFERENT figure, so a change of method cannot pass "
          "unnoticed",
          abs(c["O12"]["spearman"] - c["O12"]["pearson"]) > 0.02,
          "%r vs %r" % (c["O12"]["spearman"], c["O12"]["pearson"]))

    # ---- O6: the analyst's method works here and does not generalise -------
    # ---- O6: the analyst's documented method does not match their own column
    check("every address in the workbook is on one domain",
          c["O6"]["distinct_domains_delivered"] == ["fp20analytics.com"],
          str(c["O6"]["distinct_domains_delivered"]))
    check("the analyst's documented formula does NOT reproduce their own "
          "delivered Email Domain column",
          c["O6"]["formula_matches_delivered_column"] is False,
          str(c["O6"]))
    check("and the difference is exactly the trailing .com the -4 removes",
          c["O6"]["formula_example"] == "fp20analytics"
          and c["O6"]["delivered_example"] == "fp20analytics.com",
          "%r vs %r" % (c["O6"]["formula_example"], c["O6"]["delivered_example"]))
    check("a plain split after the @ DOES reproduce the delivered column",
          c["O6"]["split_matches_delivered_column"] is True, str(c["O6"]))
    others = c["O6"]["formula_on_other_tlds"]
    check("the formula also truncates a .mx address, so it is not general either",
          others["a.b@example.mx"] != "example.mx", str(others))
    check("and .co.uk", others["e.f@example.co.uk"] != "example.co.uk", str(others))
    check("while the general method returns the domain unharmed",
          bl.domain_general("a.b@example.mx") == "example.mx")

    # ---- O7 ----------------------------------------------------------------
    check("the agent-id lookup returns a real name",
          isinstance(c["O7"]["example"]["full_name"], str)
          and len(c["O7"]["example"]["full_name"]) > 2, str(c["O7"]))
    try:
        bl.name_from_agent_id(agents, 99999)
        check("and an unknown agent id is refused rather than guessed", False,
              "it returned something")
    except bl.BaselineError:
        check("and an unknown agent id is refused rather than guessed", True)

    # ---- O4 / O5 / O8: the tabular answers ---------------------------------
    check("the four categories cover every ticket",
          sum(c["O4"]["counts"].values()) == 97498, str(c["O4"]["counts"]))
    check("all 50 agents appear in the per-agent counts",
          c["O5"]["agents"] == 50, str(c["O5"]["agents"]))
    check("and the per-agent counts sum to the corpus",
          sum(c["O5"]["by_name"].values()) == 97498)
    check("the two issue types cover every ticket",
          sum(c["O8"].values()) == 97498, str(c["O8"]))

    # ---- the comparison keeps BOTH kinds of row ----------------------------
    agree = [r for r in rows.values() if r["agrees"]]
    differ = [r for r in rows.values() if not r["agrees"]]
    check("the comparison reports agreements", len(agree) >= 10, str(len(agree)))
    check("and keeps the disagreement rather than dropping it",
          len(differ) == 1, str([(r["question"], r["figure"]) for r in differ]))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


def _refuses_truncated(corpus: Path) -> bool:
    import pandas as pd, tempfile
    scratch = Path(tempfile.mkdtemp())
    pd.read_csv(corpus / "tickets.csv").head(10).to_csv(scratch / "tickets.csv",
                                                        index=False)
    pd.read_csv(corpus / "agents.csv").to_csv(scratch / "agents.csv", index=False)
    try:
        bl.load(scratch)
        return False
    except bl.BaselineError:
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python selftest_baseline.py <corpus directory>")
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1])))
