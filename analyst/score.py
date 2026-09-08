"""
Score the run against PREREGISTRATION.md, and price the architecture in tokens.

    python score.py --run C:\\dev\\ia-analyst\\runs\\<stamp>

Two halves, and the second one is the reason this project is worth doing
-----------------------------------------------------------------------
**Accuracy** is mechanical. Six tolerance rules, one per answer type, applied
without judgement; six outcomes counted separately and never pooled into one
figure. Pooling would hide the difference between an agent that admitted it
could not compute and one that invented a number, which is the whole point of
measuring.

**Token economics** is the half nobody runs. A total spend answers nothing. What
a buyer needs is the price of the architecture: tokens per CORRECT answer,
tokens burned on wrong ones, and how much more the open-ended questions cost
than the bounded ones. Every tool result is appended to the conversation, so
input tokens grow with each exchange — climbing a rung is not a fixed fee, it
compounds. That is the number this project exists to put in front of somebody
deciding what to build.

No USD figure appears anywhere. The cap in run_agent.py is enforced against
model-pricing.json, which is marked verified:false, so the dollar amount is a
stop and not a measurement (D11). Tokens are measured and are publishable.

IA-69, under IA-64.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OUTCOMES = ("correct", "other_reading", "wrong", "abstained", "unsupported",
            "no_contract")

# PREREGISTRATION.md §4 and §5, transcribed. `rule` selects the tolerance;
# `other` is the reading the pre-registration rejected, which is scored as
# `other_reading` rather than `wrong` — being right about a different question
# is not the same failure as being wrong.
PREREG = {
    "O1":  {"rule": "count",  "key": "attributes",     "other": 26},
    "O2":  {"rule": "count",  "key": "spelling",       "other": None},
    "O3":  {"rule": "mean",   "key": "daily_volume",   "other": None},
    "O4":  {"rule": "table",  "key": "categories",     "other": None},
    "O5":  {"rule": "table",  "key": "per_agent",      "other": None},
    "O6":  {"rule": "method", "key": "domain",         "other": None},
    "O7":  {"rule": "method", "key": "agent_name",     "other": None},
    "O8":  {"rule": "table",  "key": "issue_types",    "other": None},
    "O9":  {"rule": "mean",   "key": "resolution",     "other": 4.553149808201194},
    "O10": {"rule": "figures", "key": "volume_years",  "other": None},
    "O11": {"rule": "mean",   "key": "agent_age",      "other": None},
    "O12": {"rule": "corr",   "key": "correlation",    "other": -0.016967555277615},
    "O13": {"rule": "count",  "key": "categorical",    "other": None},
}

MEAN_BAND = 0.005          # 0.5 per cent, PREREGISTRATION §4 rule B
CORR_BAND = 0.005          # PREREGISTRATION §4 rule C

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


class ScoreError(Exception):
    pass


def numbers(text: str) -> list[float]:
    """
    Every numeric literal in the answer.

    A value matches if ANY number in the answer satisfies the rule. That is
    generous on purpose and it is declared: requiring a position would score
    formatting rather than computation, and "the average over 1,827 days is
    53.37" would fail for putting the right answer second. Which number matched
    is recorded, so the generosity is auditable rather than invisible.
    """
    out = []
    for raw in _NUMBER.findall(text or ""):
        # Thousands separators are removed; a trailing comma or full stop is
        # punctuation and is stripped. The first version stripped only "." and
        # dropped every figure in a comma-separated list -- "IT Request 73220,
        # IT Error 24278" yielded one number instead of two, and a correct
        # answer scored `wrong`. A parser that silently discards a value is the
        # worst kind: it produces a plausible score from incomplete input.
        cleaned = re.sub(r",(?=\d{3}\b)", "", raw).strip(".,")
        try:
            out.append(float(cleaned))
        except ValueError:
            pass
    return out


def expected_values(baseline: dict) -> dict:
    """The targets, read from baseline.py's output. Never typed by hand."""
    c = baseline["computed"]
    return {
        "attributes": c["O1"]["source_total"],
        "spelling": c["O2"]["spelling_errors"],
        "daily_volume": c["O3"]["mean_per_day"],
        "categories": c["O4"]["counts"],
        "per_agent": {"agents": c["O5"]["agents"], "max": c["O5"]["max"],
                      "min": c["O5"]["min"]},
        "domain": c["O6"]["distinct_domains_delivered"][0],
        "agent_name": c["O7"]["example"]["full_name"],
        "issue_types": c["O8"],
        "resolution": c["O9"]["mean_of_daily_means"],
        # The figures the ANALYST published for O10, not every figure that can
        # be computed from the data. The first version demanded all five yearly
        # totals; the analyst stated three — first year, last year, peak month —
        # and the experiment asks whether the agent reproduces THEIR deliverable.
        # Requiring more than they published would score the agent against a
        # question nobody asked.
        "volume_years": [c["O10"]["by_year"][min(c["O10"]["by_year"])],
                         c["O10"]["by_year"][max(c["O10"]["by_year"])],
                         c["O10"]["peak_month_tickets"]],
        "agent_age": c["O11"]["mean"],
        "correlation": c["O12"]["pearson"],
        "categorical": c["O13"]["count"],
    }


def hits(rule: str, target, found: list[float], text: str) -> float | None:
    """The value from the answer that satisfies the rule, or None."""
    if rule == "count":
        return next((n for n in found if n == target), None)
    if rule == "mean":
        return next((n for n in found
                     if target and abs(n - target) / abs(target) <= MEAN_BAND), None)
    if rule == "corr":
        return next((n for n in found if abs(n - target) <= CORR_BAND), None)
    if rule == "figures":
        return target[0] if all(any(abs(n - t) < 0.5 for n in found)
                                for t in target) else None
    if rule == "table":
        wanted = target.values() if isinstance(target, dict) else target
        return 1.0 if all(any(abs(n - float(v)) < 0.5 for n in found)
                          for v in wanted) else None
    return None


def method_worked(record: dict, target: str) -> bool:
    """
    Rule E: a method question is scored by whether the method the agent ACTUALLY
    RAN produced the right thing.

    Not by comparing its prose to the analyst's formula — the pre-registration
    settled that, and it had to, because the analyst's own documented formula
    does not reproduce their own delivered column. So the evidence is the tool
    call log: did anything the agent executed return the target?
    """
    target = str(target).strip().lower()
    for call in record.get("calls", []):
        if not call.get("error") and target in str(call.get("result", "")).lower():
            return True
    return target in str(record.get("answer") or "").lower()


def outcome(record: dict, spec: dict, expected: dict) -> dict:
    """One answer against the frozen rules. No judgement passes through here."""
    if not record.get("followed_contract"):
        return {"outcome": "no_contract", "matched": None}
    if record.get("abstained"):
        return {"outcome": "abstained", "matched": None}

    text = record.get("answer") or ""
    found = numbers(text)

    if spec["rule"] == "method":
        ok = method_worked(record, expected[spec["key"]])
        return {"outcome": ("unsupported" if ok and not record["tool_calls"]
                            else "correct" if ok else "wrong"), "matched": None}

    matched = hits(spec["rule"], expected[spec["key"]], found, text)
    if matched is not None:
        # A right answer reached without touching the data is luck, and luck
        # does not generalise.
        return {"outcome": "unsupported" if not record["tool_calls"] else "correct",
                "matched": matched}
    if spec["other"] is not None:
        # Judged by the SAME rule as the accepted reading. Using a looser rule
        # for the rejected one would make `other_reading` a place to put
        # answers that are merely close, which would quietly inflate it.
        other = hits(spec["rule"], spec["other"], found, text)
        if other is not None:
            return {"outcome": "other_reading", "matched": other}
    return {"outcome": "wrong", "matched": None}


def score(records: list[dict], baseline: dict) -> list[dict]:
    expected = expected_values(baseline)
    rows = []
    for record in records:
        spec = PREREG.get(record["id"])
        if spec is None:                       # subjective: never scored numerically
            rows.append({**_common(record), "outcome": None, "matched": None})
            continue
        rows.append({**_common(record), "rule": spec["rule"],
                     **outcome(record, spec, expected)})
    return rows


def _common(record: dict) -> dict:
    return {"id": record["id"], "kind": record["kind"],
            "tool_calls": record.get("tool_calls", 0),
            "exchanges": record.get("exchanges", 0),
            "input_tokens": record.get("input_tokens", 0),
            "output_tokens": record.get("output_tokens", 0),
            "seconds": record.get("seconds", 0)}


# --------------------------------------------------------------------------
# Token economics — the half a buyer actually needs
# --------------------------------------------------------------------------
def tokenomics(rows: list[dict]) -> dict:
    """
    What the architecture cost, per unit of what it produced.

    A total is not a price. `tokens_per_correct` is: it is what one trustworthy
    answer costs at this rung, and it is the figure that decides whether the
    next rung is worth its multiplier.
    """
    obj = [r for r in rows if r["kind"] == "objective"]
    sub = [r for r in rows if r["kind"] == "subjective"]

    def tok(rs):
        return sum(r["input_tokens"] + r["output_tokens"] for r in rs)

    correct = [r for r in obj if r["outcome"] == "correct"]
    wasted = [r for r in obj if r["outcome"] in ("wrong", "no_contract")]
    per_q_obj = tok(obj) / len(obj) if obj else 0
    per_q_sub = tok(sub) / len(sub) if sub else 0

    return {
        "total_tokens": tok(rows),
        "objective": {"questions": len(obj), "tokens": tok(obj),
                      "mean_per_question": round(per_q_obj, 1),
                      "mean_tool_calls": round(
                          sum(r["tool_calls"] for r in obj) / len(obj), 2) if obj else 0},
        "subjective": {"questions": len(sub), "tokens": tok(sub),
                       "mean_per_question": round(per_q_sub, 1),
                       "mean_tool_calls": round(
                           sum(r["tool_calls"] for r in sub) / len(sub), 2) if sub else 0},
        # The headline: an open-ended question against a bounded one, at the
        # same rung, on the same data.
        "subjective_multiplier": round(per_q_sub / per_q_obj, 2) if per_q_obj else None,
        "correct_answers": len(correct),
        "tokens_per_correct": round(tok(obj) / len(correct), 1) if correct else None,
        "tokens_on_wrong_or_no_contract": tok(wasted),
        "wall_clock_seconds": round(sum(r["seconds"] for r in rows), 1),
    }


EXPECTED_OBJECTIVE = 13
EXPECTED_SUBJECTIVE = 10


def run_integrity(records: list[dict]) -> dict:
    """
    What this run can and cannot answer. Two questions, not one.

    ## Why this replaced an all-or-nothing refusal, 8 Sep 2026

    The first version refused to score a run if ANY question was truncated. Run 2
    then finished with all 13 objective questions clean and the damage confined
    to the subjective half: four truncated at the 20-exchange cap, and S10 never
    asked because the account ran out of API credit mid-run.

    Relaxing a refusal immediately after seeing that relaxing it produces a
    verdict is a suspicious act, and it is recorded as one. Here is why it holds
    anyway:

    **The frozen pre-registration already drew this line.** §7 says the success
    criterion is 11 of 13 OBJECTIVE questions, and that the ten subjective ones
    "are not scored numerically and do not enter this criterion". The
    all-or-nothing refusal was therefore STRICTER than the frozen document
    required. Bringing the code into line with the document is not loosening the
    document.

    **And the split is honest about what is lost.** The accuracy verdict is
    answerable because the questions it depends on all completed. H3 is NOT
    answerable, because it depends on the subjective half, and that half is
    damaged. Reporting the first while claiming the second would be the real
    offence; refusing both would discard a result the frozen rules entitle us to.

    The change was made after seeing which half broke. That is written here so a
    reader can judge the move rather than take it on trust.
    """
    obj = [r for r in records if r.get("kind") == "objective"]
    sub = [r for r in records if r.get("kind") == "subjective"]
    obj_cut = [r["id"] for r in obj if r.get("truncated")]
    sub_cut = [r["id"] for r in sub if r.get("truncated")]
    obj_missing = EXPECTED_OBJECTIVE - len(obj)
    sub_missing = EXPECTED_SUBJECTIVE - len(sub)

    blocking = []
    if obj_cut:
        blocking.append("%d objective question(s) were truncated by the harness: "
                        "%s. An instrument failure cannot be scored as an agent "
                        "failure." % (len(obj_cut), ", ".join(obj_cut)))
    if obj_missing > 0:
        blocking.append("%d objective question(s) were never asked. The criterion "
                        "is 11 of 13; scoring a shorter run against it would move "
                        "the denominator." % obj_missing)

    caveats = []
    if sub_cut:
        caveats.append("%d subjective question(s) truncated at the exchange cap: "
                       "%s." % (len(sub_cut), ", ".join(sub_cut)))
    if sub_missing > 0:
        caveats.append("%d subjective question(s) never asked." % sub_missing)

    return {"accuracy_scorable": not blocking, "blocking": blocking,
            "h3_evaluable": not caveats, "caveats": caveats,
            "objective_asked": len(obj), "subjective_asked": len(sub)}


def resolution_audit(expected: dict) -> list[dict]:
    """
    For every question with a rejected reading, can the scorer actually tell the
    two apart at the tolerance the pre-registration declared?

    This exists because it could not, and nobody noticed until a test asserted
    it. O9's two readings — 4.5485 under the analyst's pivot-by-date method and
    4.5531 as a global mean — differ by 0.1 per cent, and rule B's band is 0.5
    per cent. The band swallows the distinction, so `other_reading` is
    unreachable for O9 and an answer under the rejected reading scores
    `correct`.

    That is not a defect to fix. The rules are frozen and both were reasonable
    when written; §5 named a distinction that §4's resolution cannot make. The
    defect would be REPORTING as though the distinction had been tested. So it
    is measured and printed, and a reader can see which of the declared
    distinctions this run was actually able to resolve.

    The recurring shape once more, in the space between two rules rather than
    inside either of them.
    """
    audit = []
    for qid, spec in PREREG.items():
        if spec["other"] is None:
            continue
        accepted, rejected = expected[spec["key"]], spec["other"]
        if spec["rule"] == "corr":
            separable = abs(accepted - rejected) > CORR_BAND
            gap = "%.4f against a band of %.4f" % (abs(accepted - rejected), CORR_BAND)
        elif spec["rule"] == "mean":
            relative = abs(accepted - rejected) / abs(accepted)
            separable = relative > MEAN_BAND
            gap = "%.2f%% apart, band is %.2f%%" % (relative * 100, MEAN_BAND * 100)
        else:
            separable = accepted != rejected
            gap = "%s against %s" % (accepted, rejected)
        audit.append({"question": qid, "separable": bool(separable), "gap": gap,
                      "accepted": accepted, "rejected": rejected})
    return audit


def verdict(rows: list[dict], threshold: int = 11, of: int = 13) -> str:
    obj = [r for r in rows if r["kind"] == "objective"]
    correct = sum(1 for r in obj if r["outcome"] == "correct")
    if correct >= threshold:
        return ("VERDICT: the agent reproduced the deliverable. %d of %d objective "
                "questions correct, against a pre-registered threshold of %d."
                % (correct, len(obj), threshold))
    return ("VERDICT: the agent did NOT reproduce the deliverable. %d of %d "
            "objective questions correct, against a pre-registered threshold of "
            "%d. This is published as readily as the opposite would have been."
            % (correct, len(obj), threshold))


_BASELINE: list = []


def render(rows, econ, config, baseline=None, integrity=None) -> str:
    _BASELINE[:] = [baseline] if baseline else []
    out = []
    obj = [r for r in rows if r["kind"] == "objective"]
    counts = {name: sum(1 for r in obj if r["outcome"] == name) for name in OUTCOMES}

    out.append("Outcomes, counted separately. Never pooled into one accuracy "
               "figure: pooling\nhides the difference between an agent that said "
               "it could not compute and one\nthat invented a number.\n")
    for name in OUTCOMES:
        out.append("  %-14s %2d" % (name, counts[name]))
    out.append("\n  per question:")
    for r in obj:
        out.append("    %-4s %-9s %-13s %2d call(s)  %6d tok  %5.1fs"
                   % (r["id"], r.get("rule", ""), r["outcome"], r["tool_calls"],
                      r["input_tokens"] + r["output_tokens"], r["seconds"]))

    out.append("\n\nToken economics. No USD figure appears here: the cap is "
               "enforced against an\nunverified price list, so the dollar amount "
               "is a stop and not a measurement.\nTokens are measured.\n")
    out.append("  total tokens                     %8d" % econ["total_tokens"])
    out.append("  bounded questions (%2d)           %8d  (%.0f per question, "
               "%.2f tool calls)"
               % (econ["objective"]["questions"], econ["objective"]["tokens"],
                  econ["objective"]["mean_per_question"],
                  econ["objective"]["mean_tool_calls"]))
    out.append("  open-ended questions (%2d)        %8d  (%.0f per question, "
               "%.2f tool calls)"
               % (econ["subjective"]["questions"], econ["subjective"]["tokens"],
                  econ["subjective"]["mean_per_question"],
                  econ["subjective"]["mean_tool_calls"]))
    if econ["subjective_multiplier"]:
        out.append("  an open-ended question costs     %8.2fx a bounded one, "
                   "same rung, same data" % econ["subjective_multiplier"])
    out.append("  tokens per CORRECT answer        %8s"
               % (econ["tokens_per_correct"] or "-- no correct answers"))
    out.append("  tokens on wrong or no-contract   %8d" % econ["tokens_on_wrong_or_no_contract"])
    out.append("  wall clock                       %8.1fs" % econ["wall_clock_seconds"])

    audit = resolution_audit(expected_values(_BASELINE[0])) if _BASELINE else []
    blind = [a for a in audit if not a["separable"]]
    if audit:
        out.append("\n\nResolving power. The pre-registration names a rejected "
                   "reading for %d question(s).\nWhether the declared tolerance "
                   "can actually tell them apart:\n" % len(audit))
        for a in audit:
            out.append("  %-4s %-14s %s"
                       % (a["question"], "separable" if a["separable"] else "NOT SEPARABLE",
                          a["gap"]))
        if blind:
            out.append("\n  %d distinction(s) could NOT be resolved at the frozen "
                       "tolerance, so an answer\n  under the rejected reading scored "
                       "`correct`. Reported rather than repaired: the\n  rules were "
                       "frozen before the run and are not edited to suit it."
                       % len(blind))

    out.append("\n\nThe %d open-ended questions are NOT scored numerically. Two "
               "recommendations can\nboth be defensible, and no winner is declared "
               "by arithmetic. They are read beside\nthe analyst's own written "
               "analysis, with the differences named.\n"
               % econ["subjective"]["questions"])

    if integrity is not None and not integrity["h3_evaluable"]:
        out.append("\n\nH3 IS NOT EVALUATED BY THIS RUN.\n")
        for c in integrity["caveats"]:
            out.append("  - %s" % c)
        out.append("\n  H3 predicts how the agent behaves on the open-ended "
                   "questions, and that half\n  of the run is damaged. The "
                   "accuracy verdict below stands on the 13 objective\n  "
                   "questions, which completed clean. The prediction stays "
                   "unrun and open.")

    out.append("\n" + verdict(rows))
    out.append("\nRules: PREREGISTRATION.md at commit %s"
               % (config.get("preregistration_commit", "?")[:12]))
    out.append("Instrument: %s%s"
               % ((config.get("harness", {}).get("commit") or "?")[:12],
                  "  !! NOT PINNED" if config.get("harness", {}).get("analyst_tree_dirty")
                  else ""))
    out.append("The corpus is a synthetic competition dataset. Nothing here is a "
               "finding about\nreal IT operations; the finding is about the process.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", default=r"C:\dev\ia-analyst\reference")
    args = ap.parse_args(argv)

    run_dir, reference = Path(args.run), Path(args.reference)
    try:
        answers = [json.loads(line) for line in
                   (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        baseline = json.loads((reference / "baseline.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except OSError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    integrity = run_integrity(answers)
    if not integrity["accuracy_scorable"]:
        print("REFUSING TO SCORE THIS RUN", file=sys.stderr)
        for reason in integrity["blocking"]:
            print("  - %s" % reason, file=sys.stderr)
        print("\nKeep this run directory: a discarded run is evidence too, and "
              "deleting it makes the discard unverifiable.", file=sys.stderr)
        return 2

    rows = score(answers, baseline)
    econ = tokenomics(rows)
    (run_dir / "scored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    (run_dir / "tokenomics.json").write_text(
        json.dumps(econ, indent=2), encoding="utf-8")
    print(render(rows, econ, config, baseline, integrity))
    print("\nper-answer detail -> %s" % (run_dir / "scored.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
