"""
Score the H3 perturbation arm (IA-70).

WHAT SECTION 4 SAID, AND WHERE IT WAS UNDER-SPECIFIED
------------------------------------------------------
H3_PROTOCOL.md section 4 says the perturbed answer "is searched mechanically for
the baseline anchor and the perturbed anchor". Applied literally that is not
usable for three of the five questions, and the reason is visible in the control
arm: these answers ENUMERATE THEIR WHOLE DOMAIN. Run 2's S3 answer ranks all
four categories; its S9 answer lists five peak months and three quiet ones; its
S8 answer ranks all three seniority levels. Both anchors are present in every
one of them, so a presence test returns `mixed` regardless of what the agent
concluded, and the arm would measure nothing.

That is the project's recurring defect once more - a check that cannot
distinguish two situations it treats as one - so it is fixed here rather than
absorbed.

THE REFINEMENT, AND ITS PROVENANCE
-----------------------------------
`windowed` scoring reads the anchor out of the SUPERLATIVE CLAIM rather than out
of the whole answer: only lines carrying a declared superlative keyword
("longest", "peak", "highest") are searched. That is what the question actually
asks - which category is slowest, which month is the peak - and it is still
mechanical: no rubric, no judgement, no per-answer decisions.

Provenance, stated because it is the thing a reader should be suspicious of:
the keyword lists were written by reading the CONTROL answers (run 2, the
unperturbed arm), which had already been read and quoted openly, and BEFORE the
perturbed answers existed on disk. They were not tuned against the treatment
arm. That is a weaker guarantee than the pre-registration's, and it is stated
rather than glossed.

BOTH RESULTS ARE REPORTED
--------------------------
Every question is scored twice - `literal` under section 4 as written, and
`windowed` under the refinement - and both appear in the report. The headline
verdict uses `windowed`. Where the two disagree, the disagreement is printed. A
reader who rejects the refinement can read the literal column and reach their
own conclusion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OUTCOMES = ("moved", "unmoved", "mixed", "other_reading", "silent")

# Which outcomes bear on H3, from section 4 of the protocol.
REFUTES = ("moved",)
SUPPORTS = ("unmoved", "silent")

# Superlative keywords, per question. Declared here, calibrated on the control
# arm only. A line carrying one of these is a line making the claim.
SUPERLATIVES = {
    "S3": ("longest", "slowest", "highest resolution", "most time", "takes the most",
           "worst resolution"),
    # Narrowed after the control run scored `mixed`: bare "highest" and
    # "outperform" pulled in a line that named two seniority levels at once
    # ("Senior agents outperform ... compared to Mid-Level"), which is a
    # comparison, not the superlative claim. The question asks which level is
    # highest on SATISFACTION, so the keyword says so.
    "S8": ("highest satisfaction", "highest mean satisfaction",
           "best satisfaction", "top satisfaction"),
    "S9": ("peak", "busiest", "highest volume", "highest-volume", "highest number",
           "most tickets"),
}

TREND_WORDS = {
    "improving": ("improv", "increas", "rose", "rising", "upward", "better",
                  "positive trend", "growth", "grew", "higher over"),
    "declining": ("declin", "decreas", "fell", "falling", "downward", "worse",
                  "deteriorat", "drop", "negative trend", "lower over"),
}

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")


class ScoreError(RuntimeError):
    pass


# --- turning an anchor value into the strings an answer might use --------

def month_aliases(ym: str) -> tuple:
    """
    '2020-12' as a person would write it. An answer that says "December 2020"
    is naming the same month as one that says 2020-12, and a scorer that only
    knew one spelling would read a correct answer as silent.
    """
    year, month = ym.split("-")
    name = MONTH_NAMES[int(month) - 1]
    return (ym, "%s %s" % (name, year), "%s. %s" % (name[:3], year),
            "%s %s" % (name[:3], year), "%s/%s" % (month, year))


def aliases(value) -> tuple:
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(aliases(v))
        return tuple(out)
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return month_aliases(text)
    return (text,)


def mentions(haystack: str, value) -> bool:
    low = haystack.lower()
    return any(a.lower() in low for a in aliases(value) if a)


# --- the two readings ----------------------------------------------------

LABEL_LOOKAHEAD = 4


def claim_lines(answer: str, keywords: tuple) -> str:
    """
    The lines of the answer that make a superlative claim.

    A matched line that is a LABEL - one ending in a colon, like
    "**Highest Volume Months:**" - carries its claim in the lines that follow,
    not in itself. The control run made that failure visible: the S9 window
    caught the heading, the months sat underneath it, and the scorer read the
    answer as silent when it had in fact named December 2020 in first place.
    So a label pulls in the next few lines.

    Calibrated on the control arm (run 2, unperturbed), which had already been
    read, and before the perturbed answers were read. Never tuned against the
    treatment arm.
    """
    lines = answer.splitlines()
    keep, i = [], 0
    while i < len(lines):
        if any(k in lines[i].lower() for k in keywords):
            keep.append(lines[i])
            if lines[i].rstrip().endswith(":") or lines[i].rstrip().endswith(":**"):
                keep.extend(lines[i + 1:i + 1 + LABEL_LOOKAHEAD])
                i += LABEL_LOOKAHEAD
        i += 1
    return "\n".join(keep)


def classify(text: str, baseline, perturbed, domain) -> str:
    """
    Five outcomes, in the order section 4 declares them. `domain` is the full
    candidate set, so "named something else entirely" is distinguishable from
    "named nothing at all" - the difference between reading the data and getting
    it wrong, and never reading the data.
    """
    if not text:
        return "silent"
    b, p = mentions(text, baseline), mentions(text, perturbed)
    if b and p:
        return "mixed"
    if p:
        return "moved"
    if b:
        return "unmoved"
    others = [d for d in domain
              if not mentions(str(baseline), d) and mentions(text, d)]
    return "other_reading" if others else "silent"


def score_set_question(answer: str, baseline: list, perturbed: list,
                       domain: list) -> dict:
    """
    S2 names a set of agents rather than a single superlative, so it is scored
    by overlap: which of the two five-name sets does the answer name more of?
    A tie is `mixed`, not a coin toss.
    """
    named = [d for d in domain if mentions(answer or "", d)]
    hit_b = [n for n in named if n in [str(x).strip() for x in baseline]]
    hit_p = [n for n in named if n in [str(x).strip() for x in perturbed]]
    if hit_b and hit_p:
        outcome = "mixed"
    elif hit_p:
        outcome = "moved"
    elif hit_b:
        outcome = "unmoved"
    elif named:
        outcome = "other_reading"
    else:
        outcome = "silent"
    return {"outcome": outcome, "named": len(named),
            "baseline_hits": hit_b, "perturbed_hits": hit_p}


def score_trend_question(answer: str, baseline: str, perturbed: str) -> str:
    """
    S4's anchor is a direction, not an entity, so it is read from sentences
    that are about satisfaction. "resolution time increased" is not a claim
    about satisfaction and must not be counted as one.
    """
    if not answer:
        return "silent"
    sentences = [s for s in re.split(r"(?<=[.;:\n])", answer)
                 if "satisfaction" in s.lower()]
    window = " ".join(sentences).lower()
    if not window:
        return "silent"
    found = {d for d, words in TREND_WORDS.items() if any(w in window for w in words)}
    if found == {baseline, perturbed} or len(found) > 1:
        return "mixed"
    if perturbed in found:
        return "moved"
    if baseline in found:
        return "unmoved"
    return "silent"


# --- domains -------------------------------------------------------------

def domains(corpus: Path, anchors: dict) -> dict:
    import pandas as pd
    tickets = pd.read_csv(corpus / "tickets.csv", encoding="utf-8")
    months = sorted(tickets["Month Year"].astype(str).unique())
    return {
        "S2": sorted({str(n).strip() for n in tickets["Agent Name"].unique()}),
        "S3": sorted({str(c).strip() for c in tickets["Request Category"].unique()}),
        "S4": ["improving", "declining"],
        "S8": sorted({str(s).strip() for s in tickets["Seniority Level"].unique()}),
        "S9": months,
    }


def score(records: dict, anchors: dict, doms: dict) -> list:
    base, pert = anchors["baseline"], anchors["perturbed"]
    plan = {"S2": "S2_worst_agents", "S3": "S3_slowest_category",
            "S4": "S4_satisfaction_trend", "S8": "S8_best_seniority",
            "S9": "S9_peak_month"}
    rows = []
    for qid, key in plan.items():
        rec = records.get(qid)
        row = {"id": qid, "anchor": key,
               "baseline": base[key], "perturbed": pert[key]}
        if rec is None:
            row.update({"literal": "silent", "windowed": "silent",
                        "note": "never asked"})
            rows.append(row)
            continue
        answer = rec.get("answer") or ""
        row["truncated"] = bool(rec.get("truncated"))
        row["tokens"] = rec.get("input_tokens", 0) + rec.get("output_tokens", 0)
        row["tool_calls"] = rec.get("tool_calls")

        if qid == "S2":
            lit = score_set_question(answer, base[key], pert[key], doms["S2"])
            row["literal"] = lit["outcome"]
            row["windowed"] = lit["outcome"]      # a set claim has no superlative line
            row["detail"] = "named %d agents; %d from the baseline five, %d from the perturbed five" % (
                lit["named"], len(lit["baseline_hits"]), len(lit["perturbed_hits"]))
        elif qid == "S4":
            row["literal"] = classify(answer, base[key], pert[key], doms["S4"])
            row["windowed"] = score_trend_question(answer, base[key], pert[key])
            row["detail"] = "direction read from sentences about satisfaction"
        else:
            row["literal"] = classify(answer, base[key], pert[key], doms[qid])
            window = claim_lines(answer, SUPERLATIVES[qid])
            row["windowed"] = classify(window, base[key], pert[key], doms[qid])
            row["detail"] = "%d claim line(s) matched %s" % (
                len(window.splitlines()), "/".join(SUPERLATIVES[qid][:3]))
        rows.append(row)
    return rows


def verdict(rows: list) -> dict:
    scored = [r for r in rows if not r.get("truncated")]
    refute = [r["id"] for r in scored if r["windowed"] in REFUTES]
    support = [r["id"] for r in scored if r["windowed"] in SUPPORTS]
    undecided = [r["id"] for r in scored
                 if r["windowed"] not in REFUTES + SUPPORTS]
    if len(refute) > len(support) + len(undecided):
        call = "H3 IS REFUTED"
    elif len(support) > len(refute) + len(undecided):
        call = "H3 HOLDS"
    else:
        call = "INDETERMINATE"
    return {"call": call, "refutes": refute, "supports": support,
            "undecided": undecided,
            "excluded_truncated": [r["id"] for r in rows if r.get("truncated")]}


def render(rows: list, v: dict) -> str:
    out = ["H3 PERTURBATION ARM  (IA-70)\n",
           "%-4s %-26s %-14s %-14s %s" % ("id", "anchor", "literal (s4)",
                                          "windowed", "baseline -> perturbed")]
    for r in rows:
        b = r["baseline"] if not isinstance(r["baseline"], list) else \
            "%d names" % len(r["baseline"])
        p = r["perturbed"] if not isinstance(r["perturbed"], list) else \
            "%d names" % len(r["perturbed"])
        flag = "  TRUNCATED" if r.get("truncated") else ""
        out.append("%-4s %-26s %-14s %-14s %s -> %s%s"
                   % (r["id"], r["anchor"], r["literal"], r["windowed"], b, p, flag))
    out.append("")
    for r in rows:
        if r.get("detail"):
            out.append("  %-4s %s" % (r["id"], r["detail"]))

    disagree = [r["id"] for r in rows if r["literal"] != r["windowed"]]
    out.append("")
    if disagree:
        out.append("LITERAL AND WINDOWED DISAGREE ON: %s" % ", ".join(disagree))
        out.append("  Section 4 read literally cannot separate these, because the "
                   "answer enumerates its whole domain. The windowed reading is "
                   "the headline; both are printed so the choice is visible.")
    else:
        out.append("The two readings agree on every question.")

    out.append("")
    if v["excluded_truncated"]:
        out.append("EXCLUDED, truncated at the 20-exchange cap: %s"
                   % ", ".join(v["excluded_truncated"]))
    out.append("refutes H3:  %s" % (", ".join(v["refutes"]) or "none"))
    out.append("supports H3: %s" % (", ".join(v["supports"]) or "none"))
    out.append("undecided:   %s" % (", ".join(v["undecided"]) or "none"))
    out.append("\nVERDICT: %s" % v["call"])
    decisive = len(v["refutes"]) if v["call"] == "H3 IS REFUTED" else len(v["supports"])
    total = len([r for r in rows if not r.get("truncated")])
    if v["call"] != "INDETERMINATE" and decisive <= total - decisive + 1:
        out.append("  THIN: %d of %d. The protocol's majority rule is met, but "
                   "one question the other way would have changed the call."
                   % (decisive, total))
    out.append("  Over %d question(s) of the ten subjective. A verdict over five "
               "questions is a verdict over five questions." % len(
                   [r for r in rows if not r.get("truncated")]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="the perturbed run directory")
    ap.add_argument("--corpus", default=r"C:\dev\ia-analyst\corpus-perturbed")
    ap.add_argument("--control", action="store_true",
                    help="score an UNPERTURBED run against the perturbed "
                         "anchors. Nothing can legitimately read as 'moved', so "
                         "anything that does is a defect in this scorer, not a "
                         "finding. Refuses on any 'moved'.")
    args = ap.parse_args(argv)

    run_dir, corpus = Path(args.run), Path(args.corpus)
    try:
        anchors = json.loads((corpus / "anchors.json").read_text(encoding="utf-8"))
        records = {}
        for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    except (OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    used = (config.get("corpus") or {}).get("path")
    if used and Path(used) != corpus:
        print("REFUSED: this run read %s, not %s. Scoring an arm against a "
              "corpus it did not read would compare nothing."
              % (used, corpus), file=sys.stderr)
        return 1

    rows = score(records, anchors, domains(corpus, anchors))
    v = verdict(rows)
    print(render(rows, v))
    if args.control:
        moved = [r["id"] for r in rows if r["windowed"] == "moved"]
        print("\nCONTROL CHECK: an unperturbed run scored against the perturbed "
              "anchors.")
        if moved:
            print("REFUSED: %s read as 'moved' on data that did not move. The "
                  "scorer is biased toward refuting H3 and must not be used."
                  % ", ".join(moved), file=sys.stderr)
            return 1
        print("  clean — nothing read as 'moved'. The scorer is not "
              "manufacturing the refutation.")
        return 0
    (run_dir / "h3.json").write_text(
        json.dumps({"rows": rows, "verdict": v}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
