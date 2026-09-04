"""
Score the two arms without knowing which arm produced which answer.

    python score_arms.py --rate   --run C:\\dev\\ia-pilot\\runs\\<stamp>
    python score_arms.py --report --run C:\\dev\\ia-pilot\\runs\\<stamp>

Two passes, in this order, and the order is the point
-----------------------------------------------------
**--rate** shows the answers shuffled, one at a time, carrying nothing but an
opaque token. No arm, no incident id, no label, no prompt. The prompt is
withheld deliberately: arm B's is four times longer, so showing it would name
the arm on sight. The rater supplies a usefulness score and says which fault the
answer describes.

**--report** unblinds: it joins the ratings to the answers, the answers to the
ground-truth log, and prints the table. Nothing here can influence what was
rated, because the rating already exists on disk.

What the blinding cannot do, stated because it will be asked
-----------------------------------------------------------
An arm B answer can cite datapoints only arm B was given. A rater who notices
that has effectively identified the arm. **Removing the label hides the name,
not the inference**, and no amount of code fixes it. It is a threat to the
usefulness metric -- not to the accuracy metric, which is mechanical and does
not pass through a human at all. The write-up says so.

The accuracy scoring is deliberately not a boolean
--------------------------------------------------
"Wrong" hides the difference between an agent that named the wrong service and
one that admitted it could not tell. The first is the failure this pilot cares
about; the second is the behaviour the control exists to reward. They are
counted separately.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import ground_truth as gt

# The corpus size the pre-registration was written against (IA-45, amended
# 4 Sep). Below it, this tool refuses to state a verdict on H1 -- the threshold
# "arm B wins at least 4" is a claim about twelve incidents, and restating it
# over six would be the adjustment the pre-registration exists to prevent.
PREREGISTERED_N = 12
PREREGISTERED_WINS = 4

OUTCOMES = ("correct", "wrong_node", "false_positive", "abstained", "no_contract")


class ScoreError(Exception):
    pass


def outcome(claimed: str | None, fault: str, fault_role: str) -> str:
    """
    One answer against its label. Mechanical: no judgement passes through here.

      correct         named the service that failed, or said INSUFFICIENT for
                      the control
      wrong_node      named a service that did not fail -- confident and wrong
      false_positive  named a service during the control, where nothing happened
      abstained       said INSUFFICIENT when something did fail -- wrong, but
                      wrong in the direction that does not mislead an on-call
      no_contract     produced no ROOT CAUSE line at all
    """
    if claimed is None:
        return "no_contract"
    claim = claimed.strip().lower()
    if fault in gt.SIGNAL_FREE_FAULTS:
        return "correct" if claim == "insufficient" else "false_positive"
    if claim == "insufficient":
        return "abstained"
    return "correct" if claim == fault_role.strip().lower() else "wrong_node"


def blind_order(answers: list[dict], seed: int) -> list[dict]:
    """
    The answers shuffled reproducibly, each carrying an opaque token.

    The seed is recorded so the shuffle can be reproduced and audited. A shuffle
    nobody can reproduce is indistinguishable from an ordering chosen to suit
    the result.
    """
    tokens = [dict(record, token="R%03d" % index)
              for index, record in enumerate(answers)]
    rng = random.Random(seed)
    rng.shuffle(tokens)
    return tokens


def join(answers: list[dict], ratings: list[dict], truth: list[dict]) -> list[dict]:
    """Answers + their labels + whatever the blind pass recorded."""
    labels = {row["incident_id"]: row for row in truth if row.get("record") == "open"}
    by_token = {row["token"]: row for row in ratings}
    joined = []
    for index, answer in enumerate(answers):
        token = "R%03d" % index
        label = labels.get(answer["incident_id"])
        if label is None:
            raise ScoreError(
                "%s has an answer but no ground-truth record. An answer with no "
                "label cannot be scored, and inventing one is the failure this "
                "pilot exists to avoid." % answer["incident_id"])
        rating = by_token.get(token, {})
        joined.append({
            "incident_id": answer["incident_id"],
            "arm": answer["arm"],
            "fault": label["fault"],
            "fault_role": label.get("node_role"),
            "claimed_cause": answer.get("claimed_cause"),
            "outcome": outcome(answer.get("claimed_cause"), label["fault"],
                               label.get("node_role") or ""),
            "usefulness": rating.get("usefulness"),
            "described_fault": rating.get("described_fault"),
            "input_tokens": (answer.get("usage") or {}).get("input_tokens"),
            "output_tokens": (answer.get("usage") or {}).get("output_tokens"),
        })
    return joined


def per_class(rows: list[dict]) -> dict:
    """Accuracy per fault class, per arm. Never pooled -- see IA-59 on IA-45."""
    table = {}
    for row in rows:
        bucket = table.setdefault(row["fault"], {})
        arm = bucket.setdefault(row["arm"], {name: 0 for name in OUTCOMES})
        arm[row["outcome"]] += 1
    return table


def wins(rows: list[dict]) -> dict:
    """Incidents where one arm was correct and the other was not."""
    by_incident = {}
    for row in rows:
        by_incident.setdefault(row["incident_id"], {})[row["arm"]] = row["outcome"]
    b_wins, a_wins, tied = [], [], []
    for incident_id, arms in sorted(by_incident.items()):
        a_ok = arms.get("arm_a") == "correct"
        b_ok = arms.get("arm_b") == "correct"
        if b_ok and not a_ok:
            b_wins.append(incident_id)
        elif a_ok and not b_ok:
            a_wins.append(incident_id)
        else:
            tied.append(incident_id)
    return {"arm_b": b_wins, "arm_a": a_wins, "tied": tied,
            "incidents": len(by_incident)}


def cost(rows: list[dict]) -> dict:
    """Input and output tokens per arm. H2 is measured here, in tokens."""
    out = {}
    for row in rows:
        arm = out.setdefault(row["arm"], {"input": 0, "output": 0, "answers": 0})
        arm["input"] += row["input_tokens"] or 0
        arm["output"] += row["output_tokens"] or 0
        arm["answers"] += 1
    return out


def render(rows: list[dict]) -> str:
    lines = []
    table = per_class(rows)
    lines.append("Accuracy by fault class. Never pooled: with F3 dominating the "
                 "corpus a single")
    lines.append("headline number would mostly measure how many stop faults were "
                 "injected.\n")
    lines.append("  class  arm    correct  wrong_node  false_pos  abstained  no_contract")
    for fault in sorted(table):
        for arm in sorted(table[fault]):
            counts = table[fault][arm]
            lines.append("  %-5s  %-5s  %7d  %10d  %9d  %9d  %11d"
                         % (fault, arm, counts["correct"], counts["wrong_node"],
                            counts["false_positive"], counts["abstained"],
                            counts["no_contract"]))

    result = wins(rows)
    lines.append("\nPer-incident comparison over %d incident(s):" % result["incidents"])
    lines.append("  arm B correct where arm A was not: %d  %s"
                 % (len(result["arm_b"]), ", ".join(result["arm_b"]) or "-"))
    lines.append("  arm A correct where arm B was not: %d  %s"
                 % (len(result["arm_a"]), ", ".join(result["arm_a"]) or "-"))
    lines.append("  neither or both:                   %d" % len(result["tied"]))

    spend = cost(rows)
    lines.append("\nCost, in tokens. H2 is measured here and needs no price list:")
    for arm in sorted(spend):
        item = spend[arm]
        lines.append("  %-5s  %7d input  %6d output  over %d answers"
                     % (arm, item["input"], item["output"], item["answers"]))
    if "arm_a" in spend and "arm_b" in spend and spend["arm_a"]["input"]:
        ratio = spend["arm_b"]["input"] / spend["arm_a"]["input"]
        lines.append("  arm B costs %.1fx the input tokens of arm A." % ratio)

    rated = [row["usefulness"] for row in rows if row["usefulness"] is not None]
    if rated:
        for arm in sorted({row["arm"] for row in rows}):
            scores = [row["usefulness"] for row in rows
                      if row["arm"] == arm and row["usefulness"] is not None]
            if scores:
                lines.append("\n  %s usefulness: mean %.2f over %d rated"
                             % (arm, sum(scores) / len(scores), len(scores)))
    else:
        lines.append("\nNo usefulness ratings found. Run --rate first, or the "
                     "secondary metric is simply absent -- which is reported as "
                     "absent, not as zero.")

    lines.append("\n" + verdict(result))
    return "\n".join(lines)


def verdict(result: dict) -> str:
    """
    What may and may not be concluded, given how many incidents there are.

    The pre-registered criterion -- arm B correct where arm A was not, in at
    least 4 incidents -- is a claim about a corpus of twelve. Restating it over
    six would be adjusting the threshold to the sample, which is the move the
    pre-registration exists to prevent. So below the target this refuses to
    give a verdict at all rather than giving a smaller one.
    """
    n = result["incidents"]
    if n < PREREGISTERED_N:
        return (
            "VERDICT: none. %d incidents, and the pre-registration is written "
            "against %d.\nThis run is a pipeline validation (IA-45, amended "
            "4 Sep): it shows the machinery works\nend to end. It is not "
            "evidence for or against H1, and the counts above must not be\n"
            "quoted as if it were. Batch 2 is already specified and is not "
            "changed by what is\nabove." % (n, PREREGISTERED_N))
    if len(result["arm_b"]) >= PREREGISTERED_WINS:
        return ("VERDICT: H1 supported. Arm B was correct where arm A was not "
                "in %d of %d incidents,\nagainst a pre-registered threshold of "
                "%d." % (len(result["arm_b"]), n, PREREGISTERED_WINS))
    return ("VERDICT: H1 NOT supported. Arm B was correct where arm A was not "
            "in %d of %d incidents,\nagainst a pre-registered threshold of %d. "
            "This is published as readily as the\nopposite would have been."
            % (len(result["arm_b"]), n, PREREGISTERED_WINS))


def rate(blind: list[dict], ask=input, show=print) -> list[dict]:
    """The blind pass. Only the answer text is shown -- never the prompt."""
    ratings = []
    for index, record in enumerate(blind, 1):
        show("\n" + "=" * 68)
        show("%s   (%d of %d)" % (record["token"], index, len(blind)))
        show("=" * 68)
        show(record["answer"])
        show("-" * 68)
        usefulness = ask("usefulness 1-5 (blank to skip): ").strip()
        described = ask("fault described (cpu / stopped / credits / none): ").strip()
        ratings.append({
            "token": record["token"],
            "usefulness": int(usefulness) if usefulness.isdigit() else None,
            "described_fault": described or None,
        })
    return ratings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="the run directory from run_arms.py")
    ap.add_argument("--log", default=str(gt.DEFAULT_LOG))
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--rate", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.exists():
        print("no answers at %s" % answers_path, file=sys.stderr)
        return 1
    answers = [json.loads(line) for line in
               answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.rate:
        ratings = rate(blind_order(answers, args.seed))
        (run_dir / "ratings.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ratings) + "\n",
            encoding="utf-8")
        (run_dir / "shuffle.json").write_text(
            json.dumps({"seed": args.seed, "n": len(answers)}, indent=2),
            encoding="utf-8")
        print("\n%d ratings -> %s" % (len(ratings), run_dir / "ratings.jsonl"))
        return 0

    if not args.report:
        print("nothing to do: pass --rate or --report")
        return 0

    ratings_path = run_dir / "ratings.jsonl"
    ratings = ([json.loads(line) for line in
                ratings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
               if ratings_path.exists() else [])
    truth = gt.read_all(Path(args.log))
    rows = join(answers, ratings, truth)
    (run_dir / "scored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    print(render(rows))
    print("\nper-answer detail -> %s" % (run_dir / "scored.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
