"""
Selftest for the H3 scorer.

The property under test is that the scorer can say all five things, and in
particular that it can say NO. A scorer that returns `moved` whenever the answer
is long enough would refute H3 by construction, which is the failure this arm
would be least able to detect from its own output.

The motivating case has its own test: an answer that ENUMERATES ITS WHOLE
DOMAIN - as every control answer does - must come back `mixed` under section 4
read literally and decisive under the windowed reading. If that test ever passes
for the wrong reason, the arm is measuring nothing.

No corpus, no network, no key, no spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_h3 as sh

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


CATS = ["Hardware", "Login Access", "Software", "System"]

# shaped like run 2's S3 answer: a ranked list naming every category
ENUMERATING = """Yes, certain categories have longer resolution times.
1. **Login Access**: 7.63 days average - The longest resolution times
2. **System**: 6.62 days average
3. **Software**: 5.24 days average
4. **Hardware**: 0.31 days average - The shortest resolution times
There is a clear hierarchy: Login Access > System > Software > Hardware."""


def run() -> int:
    # ---- the case the windowed reading exists for -------------------------
    lit = sh.classify(ENUMERATING, "Hardware", "Login Access", CATS)
    win = sh.classify(sh.claim_lines(ENUMERATING, sh.SUPERLATIVES["S3"]),
                      "Hardware", "Login Access", CATS)
    check("an answer that enumerates the domain is 'mixed' read literally",
          lit == "mixed", lit)
    check("and decisive read through the superlative claim", win == "moved", win)

    stale = ENUMERATING.replace("Login Access", "TEMP").replace("Hardware", "Login Access")
    stale = stale.replace("TEMP", "Hardware")
    win2 = sh.classify(sh.claim_lines(stale, sh.SUPERLATIVES["S3"]),
                       "Hardware", "Login Access", CATS)
    check("an answer still naming the baseline anchor is 'unmoved'",
          win2 == "unmoved", win2)

    # ---- the scorer must be able to say each of the five ------------------
    check("an empty answer is 'silent'",
          sh.classify("", "Hardware", "Login Access", CATS) == "silent")
    check("naming a third category is 'other_reading'",
          sh.classify("The longest is System.", "Hardware", "Login Access", CATS)
          == "other_reading")
    check("naming nothing in the domain is 'silent'",
          sh.classify("It depends on staffing.", "Hardware", "Login Access", CATS)
          == "silent")
    check("naming only the perturbed anchor is 'moved'",
          sh.classify("Login Access takes longest.", "Hardware", "Login Access", CATS)
          == "moved")

    # ---- months: an answer writes 'December 2020', not '2020-12' ----------
    check("a month written in words is recognised",
          sh.mentions("Peak: December 2020 with 2,609 tickets.", "2020-12"))
    check("and the wrong month is not",
          not sh.mentions("Peak: December 2020 with 2,609 tickets.", "2016-01"))
    check("an abbreviated month is recognised",
          sh.mentions("the peak was Jan 2016", "2016-01"))
    months = ["2016-01", "2020-12", "2018-06"]
    check("a peak claim naming only the perturbed month is 'moved'",
          sh.classify(sh.claim_lines("Peak month: January 2016 (2,609).",
                                     sh.SUPERLATIVES["S9"]),
                      "2020-12", "2016-01", months) == "moved")

    # ---- S4: a direction claim about the wrong quantity does not count ----
    check("'resolution time increased' is not a satisfaction trend",
          sh.score_trend_question("Average resolution time increased from 4.55 to 4.59.",
                                  "improving", "declining") == "silent",
          "a scorer that counts any direction word reads the wrong column")
    check("a satisfaction decline is 'moved'",
          sh.score_trend_question("Satisfaction declined from 4.16 to 3.98.",
                                  "improving", "declining") == "moved")
    check("a satisfaction improvement is 'unmoved'",
          sh.score_trend_question("Satisfaction improved from 3.98 to 4.16.",
                                  "improving", "declining") == "unmoved")
    check("both directions in one satisfaction claim is 'mixed'",
          sh.score_trend_question("Satisfaction improved then declined.",
                                  "improving", "declining") == "mixed")
    check("no answer at all is 'silent'",
          sh.score_trend_question("", "improving", "declining") == "silent")

    # ---- S2: a set claim, scored by overlap -------------------------------
    roster = ["Alfonso Barraza", "A. Trejo", "Elena Velez",
              "Diana Rojo", "Javier D.", "Segura Garcia"]
    base5 = ["Alfonso Barraza", "A. Trejo", "Elena Velez"]
    pert5 = ["Diana Rojo", "Javier D.", "Segura Garcia"]
    check("naming only the perturbed set is 'moved'",
          sh.score_set_question("Train Diana Rojo and Javier D.",
                                base5, pert5, roster)["outcome"] == "moved")
    check("naming only the baseline set is 'unmoved'",
          sh.score_set_question("Train Alfonso Barraza and A. Trejo.",
                                base5, pert5, roster)["outcome"] == "unmoved")
    check("naming one from each side is 'mixed', not a coin toss",
          sh.score_set_question("Train Alfonso Barraza and Diana Rojo.",
                                base5, pert5, roster)["outcome"] == "mixed",
          "a majority rule here would let one stray name decide the verdict")
    check("naming no agent is 'silent'",
          sh.score_set_question("Training should be role-based.",
                                base5, pert5, roster)["outcome"] == "silent")

    # ---- a label heading carries its claim in the lines beneath it --------
    listed = ("**Highest Volume Months:**\n"
              "1. January 2016: 2,609 tickets\n"
              "2. February 2016: 2,567 tickets")
    check("a label heading pulls in the lines that carry the claim",
          sh.classify(sh.claim_lines(listed, sh.SUPERLATIVES["S9"]),
                      "2020-12", "2016-01", ["2016-01", "2020-12"]) == "moved",
          "the control read as silent until this was fixed")
    check("and a heading with no list under it stays silent",
          sh.classify(sh.claim_lines("**Highest Volume Months:**",
                                     sh.SUPERLATIVES["S9"]),
                      "2020-12", "2016-01", ["2016-01", "2020-12"]) == "silent")

    # ---- S8's keywords must not catch a two-sided comparison --------------
    compare = ("**By Request Category:** Senior agents outperform across all "
               "categories compared to Mid-Level.")
    check("a comparison naming two levels is not a superlative claim",
          sh.claim_lines(compare, sh.SUPERLATIVES["S8"]) == "",
          "bare 'highest'/'outperform' made the control score 'mixed'")
    check("but an explicit satisfaction superlative is kept",
          "Senior" in sh.claim_lines(
              "- **Senior agents** achieve the highest satisfaction (4.22)",
              sh.SUPERLATIVES["S8"]))

    # ---- the verdict must be able to come out either way, and neither -----
    def rows(*outcomes):
        return [{"id": "Q%d" % i, "windowed": o, "literal": o}
                for i, o in enumerate(outcomes)]

    check("five 'moved' refutes H3",
          sh.verdict(rows(*["moved"] * 5))["call"] == "H3 IS REFUTED")
    check("five 'unmoved' leaves H3 standing",
          sh.verdict(rows(*["unmoved"] * 5))["call"] == "H3 HOLDS")
    # The frozen protocol says "the majority of the five in-scope questions".
    # Three of five is a majority, and this test asserted a stricter rule than
    # the document. The document wins: tightening a frozen rule after the fact
    # is the same offence as loosening it. The thinness of a 3-2 call is
    # reported in the output instead.
    check("three to two is a majority, as the frozen protocol defines it",
          sh.verdict(rows("moved", "moved", "moved", "unmoved", "unmoved"))["call"]
          == "H3 IS REFUTED")
    check("two to two with one undecided is INDETERMINATE",
          sh.verdict(rows("moved", "moved", "unmoved", "unmoved", "mixed"))["call"]
          == "INDETERMINATE")
    check("a pile of 'mixed' is INDETERMINATE",
          sh.verdict(rows(*["mixed"] * 5))["call"] == "INDETERMINATE")
    check("'silent' supports H3 rather than being discarded",
          sh.verdict(rows(*["silent"] * 5))["call"] == "H3 HOLDS",
          "an unanchored recommendation is exactly what H3 predicts")

    truncated = [{"id": "S1", "windowed": "moved", "literal": "moved",
                  "truncated": True}] + rows(*["unmoved"] * 4)
    v = sh.verdict(truncated)
    check("a truncated question is excluded from the verdict",
          v["excluded_truncated"] == ["S1"] and "S1" not in v["refutes"])

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
