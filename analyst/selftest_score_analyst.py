"""
Selftest for the scorer and the token economics.

The property under test is that the score is MECHANICAL: the same answers must
produce the same outcome whoever runs it, and no rule may be softened after
seeing a result. So every tolerance rule is tested at its boundary, both
outcomes that are easy to conflate are tested apart, and the two parsing
defects found while building this are asserted fixed.

No network, no key, no spend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def rec(qid, answer, kind="objective", calls=1, result="", tokens=(3000, 400)):
    return {"id": qid, "kind": kind, "answer": answer,
            "followed_contract": answer is not None,
            "abstained": bool(answer and answer.upper().startswith("CANNOT")),
            "tool_calls": calls, "exchanges": 2,
            "input_tokens": tokens[0], "output_tokens": tokens[1], "seconds": 6.0,
            "calls": [{"error": False, "result": result}] if calls else []}


def run(baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    exp = sc.expected_values(baseline)

    def one(qid, answer, **kw):
        return sc.score([rec(qid, answer, **kw)], baseline)[0]

    # ---- the two parsing defects found while building this ----------------
    check("a comma-separated list yields every figure, not just the last",
          sc.numbers("IT Request 73220, IT Error 24278") == [73220.0, 24278.0],
          str(sc.numbers("IT Request 73220, IT Error 24278")))
    check("thousands separators are removed rather than splitting the number",
          13051.0 in sc.numbers("13,051 in 2016"), str(sc.numbers("13,051 in 2016")))
    check("a negative is kept negative",
          sc.numbers("-0.0405 weak") == [-0.0405], str(sc.numbers("-0.0405 weak")))
    check("O10 targets what the analyst PUBLISHED, three figures, not all five "
          "years the data can yield",
          len(exp["volume_years"]) == 3, str(exp["volume_years"]))

    # ---- rule B: the mean band, at its boundary ---------------------------
    target = exp["daily_volume"]
    check("a mean inside the 0.5 per cent band is correct",
          one("O3", "about %.2f per day" % (target * 1.004))["outcome"] == "correct")
    check("and outside it is wrong",
          one("O3", "about %.2f per day" % (target * 1.02))["outcome"] == "wrong")

    # ---- rule C: the correlation band separates Pearson from Spearman -----
    check("the analyst's Pearson is correct",
          one("O12", "-0.0405, very weak")["outcome"] == "correct")
    check("and Spearman is scored other_reading, not wrong — it is a different "
          "method, not a mistake",
          one("O12", "-0.0170")["outcome"] == "other_reading",
          str(one("O12", "-0.0170")))

    # ---- §5: the rejected reading of O9 is other_reading -------------------
    check("the analyst's method (4.5485) is correct",
          one("O9", "4.55 days")["outcome"] == "correct")
    # The rules cannot separate O9's two readings: they are 0.1 per cent apart
    # and rule B's band is 0.5 per cent. So the global mean scores `correct`.
    # Asserted as it IS, not as §5 hoped, and the scorer reports the gap.
    check("O9's rejected reading is INSIDE the frozen tolerance, so it scores "
          "correct — the rules cannot resolve that distinction",
          one("O9", "4.553 days")["outcome"] == "correct",
          str(one("O9", "4.553 days")))
    audit = {a["question"]: a for a in sc.resolution_audit(exp)}
    check("and the scorer says so: O9 is reported NOT separable",
          audit["O9"]["separable"] is False, str(audit["O9"]))
    check("while O12 IS separable — Pearson and Spearman are 0.024 apart against "
          "a 0.005 band",
          audit["O12"]["separable"] is True, str(audit["O12"]))
    check("and O1 is separable, 16 against 26",
          audit["O1"]["separable"] is True, str(audit["O1"]))
    check("and a value that is neither is wrong",
          one("O9", "7 days")["outcome"] == "wrong")

    # ---- O1: 16 counts, 26 is the rejected reading ------------------------
    check("O1 at 16 is correct", one("O1", "16 attributes")["outcome"] == "correct")
    check("O1 at 26 is other_reading, because the question never said which",
          one("O1", "26 attributes")["outcome"] == "other_reading")

    # ---- the four outcomes that are easy to conflate ----------------------
    check("no ANSWER line is no_contract",
          one("O3", None)["outcome"] == "no_contract")
    check("an abstention is its own outcome, not a wrong answer",
          one("O3", "CANNOT COMPUTE")["outcome"] == "abstained")
    check("a RIGHT answer with no tool call is unsupported, not correct — a "
          "number reached without touching the data is luck",
          one("O3", "53.365", calls=0)["outcome"] == "unsupported",
          str(one("O3", "53.365", calls=0)))
    check("the same answer WITH a tool call is correct",
          one("O3", "53.365", calls=1)["outcome"] == "correct")

    # ---- rule E: a method is scored by what the agent actually ran --------
    check("a method question is correct when the agent's own tool call returned "
          "the right thing",
          one("O6", "I split on the @", result="fp20analytics.com")["outcome"] == "correct")
    check("and wrong when nothing it ran produced it",
          one("O6", "I would use a formula", result="nothing here")["outcome"] == "wrong")

    # ---- rule D: a table needs every figure, not most of them -------------
    counts = exp["issue_types"]
    every = ", ".join("%s %d" % (k, v) for k, v in counts.items())
    check("a table with every count is correct",
          one("O8", every)["outcome"] == "correct", every)
    check("and a table missing one is wrong — a distribution with a wrong "
          "bucket is a wrong distribution",
          one("O8", "IT Request %d" % list(counts.values())[0])["outcome"] == "wrong")

    # ---- subjective questions are never scored numerically ----------------
    srow = sc.score([rec("S1", "Invest in training.", kind="subjective")], baseline)[0]
    check("a subjective question carries no outcome at all",
          srow["outcome"] is None, str(srow))

    # ---- token economics ---------------------------------------------------
    rows = sc.score([
        rec("O1", "16", tokens=(2000, 200)),
        rec("O3", "53.365", tokens=(2000, 200)),
        rec("O9", "999 days", tokens=(2000, 200)),          # wrong, and its cost
        rec("S1", "Invest.", kind="subjective", tokens=(8000, 800)),
    ], baseline)
    econ = sc.tokenomics(rows)
    check("tokens per CORRECT answer is reported, not tokens per question",
          econ["tokens_per_correct"] == 3300.0, str(econ["tokens_per_correct"]))
    check("tokens spent on wrong answers are counted separately, because that "
          "is the waste line",
          econ["tokens_on_wrong_or_no_contract"] == 2200, str(econ))
    check("an open-ended question is priced against a bounded one at the same rung",
          econ["subjective_multiplier"] == 4.0, str(econ["subjective_multiplier"]))
    check("and no USD figure appears anywhere in the economics",
          not any("usd" in k.lower() for k in econ), str(list(econ)))

    # ---- the verdict is read, not negotiated ------------------------------
    passing = [rec("O%d" % i, "x") for i in range(1, 14)]
    for row in sc.score(passing, baseline):
        row["outcome"] = "correct"
    check("11 of 13 clears the pre-registered bar",
          "did NOT" not in sc.verdict([{"kind": "objective", "outcome": "correct"}] * 11
                                      + [{"kind": "objective", "outcome": "wrong"}] * 2))
    check("10 of 13 does not, and the miss is framed as publishable",
          "did NOT" in sc.verdict([{"kind": "objective", "outcome": "correct"}] * 10
                                  + [{"kind": "objective", "outcome": "wrong"}] * 3))

    rendered = sc.render(rows, econ, {"preregistration_commit": "abc123456789",
                                      "harness": {"commit": "def123456789"}})
    check("the report names the rules it applied", "abc123456789" in rendered)
    check("and states that the corpus is synthetic",
          "synthetic" in rendered, rendered[-200:])
    check("and carries no dollar sign", "$" not in rendered)

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python selftest_score_analyst.py <path to baseline.json>")
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1])))
