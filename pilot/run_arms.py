"""
Ask the model each incident twice -- once per arm -- and keep everything.

    python run_arms.py --plan                    # what it would send, no calls
    python run_arms.py --go --limit 1            # one incident, both arms
    python run_arms.py --go

The two rules this file exists to enforce
-----------------------------------------
**Each arm is its own conversation.** Arm A and arm B are separate API calls
with separate message lists and nothing shared. If they shared a conversation,
arm B would see arm A's answer and the comparison would be measuring memory
instead of context. This is the reason the code looks repetitive: the repetition
is the isolation.

**Everything except the incident text is identical between the arms.** Same
model, same system prompt, same temperature, same max tokens, same output
contract. IA-48 already guarantees arm A's text appears verbatim inside arm B,
so the only difference reaching the model is the added context -- and any
measured difference can be attributed to it.

What is deliberately absent
---------------------------
No tools, no retrieval, no agentic loop. The paper's baseline is a single
question answered from what it was given, and arm B differs by what it was
given, not by what it may go and fetch. A tool-using arm B would be a different
and much easier experiment.

D6: the key comes from the operator's environment (ANTHROPIC_API_KEY), never
from a repository secret. This runs locally, by hand, watched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_budget as ab
import ground_truth as gt

MODEL = os.environ.get("PILOT_MODEL", "claude-sonnet-4-5")
MAX_OUTPUT_TOKENS = 900
TEMPERATURE = 0.0

# Identical for both arms. It states the output contract and nothing about the
# fault catalogue: naming the possible faults would hand over the answer space,
# and F0's correct answer has to remain reachable without being listed.
SYSTEM = (
    "You are an on-call engineer reviewing a cloud incident. Answer from the "
    "information given and nothing else. Do not speculate about data you were "
    "not shown, and do not assume a cause is present simply because an incident "
    "was opened.\n\n"
    "Reply in at most 150 words, then end with exactly one final line:\n"
    "ROOT CAUSE: <service name>\n"
    "or\n"
    "ROOT CAUSE: INSUFFICIENT\n\n"
    "Use INSUFFICIENT when the information does not support attributing a cause "
    "to a specific service. Naming a service you cannot justify is worse than "
    "saying you cannot tell."
)

DEFAULT_INCIDENTS = Path(r"C:\dev\ia-pilot\incidents.jsonl")
DEFAULT_RUNS = Path(r"C:\dev\ia-pilot\runs")

ARMS = ("arm_a", "arm_b")


class RunError(Exception):
    pass


def load_incidents(path: Path) -> list[dict]:
    if not path.exists():
        raise RunError("no incident file at %s. Run build_incidents.py first." % path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RunError("%s is empty." % path)
    return rows


def refuse_unusable(incidents: list[dict]) -> list[str]:
    """
    Reasons not to spend money on this corpus.

    Checked before the first call, because discovering a contaminated corpus
    after twelve API calls costs the calls and teaches nothing new -- the file
    already said so.
    """
    reasons = []
    dirty = [i["incident_id"] for i in incidents if i.get("contaminated_by")]
    if dirty:
        reasons.append(
            "%d incident(s) carry another incident's fault inside the context "
            "their prompt shows (IA-61): %s. Scoring these mixes 'the context "
            "did not help' with 'the context contained something else'."
            % (len(dirty), ", ".join(dirty)))
    unusable = [i["incident_id"] for i in incidents if not i.get("usable")]
    if unusable:
        reasons.append("no usable metric datapoints: %s" % ", ".join(unusable))
    classes = {i["fault"] for i in incidents}
    roles = {i["fault_role"] for i in incidents}
    if len(classes) < 2 or len(roles) < 2:
        reasons.append(
            "the corpus has %d fault class(es) and %d cause node(s). An agent "
            "answering with one fixed rule would score as well as one that "
            "reasons, and the result would not distinguish them."
            % (len(classes), len(roles)))
    return reasons


def ask(client, prompt: str, budget) -> tuple[str, dict]:
    """One question, one answer, no tools, no history."""
    budget.begin_iteration()
    budget.before_call()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    budget.record_response(response)
    text = "".join(block.text for block in response.content
                   if getattr(block, "type", "") == "text")
    usage = getattr(response, "usage", None)
    return text, {"input_tokens": getattr(usage, "input_tokens", None),
                  "output_tokens": getattr(usage, "output_tokens", None),
                  "stop_reason": getattr(response, "stop_reason", None)}


def claimed_cause(text: str) -> str | None:
    """
    The service named on the final ROOT CAUSE line, or None if absent.

    Extraction only. It does not decide whether the answer is right, and it does
    not fall back to hunting for a service name in the prose: an answer that did
    not follow the contract is recorded as not following it, because guessing
    what the model meant is the scorer's job and the scorer is a person (IA-50).
    """
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("ROOT CAUSE:"):
            return stripped.split(":", 1)[1].strip()
    return None


def run(incidents, client, budget, out_dir: Path, verbose=True) -> list[dict]:
    records = []
    for incident in incidents:
        for arm in ARMS:
            # A fresh message list every time. Nothing carries between arms, and
            # nothing carries between incidents.
            text, usage = ask(client, incident[arm], budget)
            record = {
                "incident_id": incident["incident_id"],
                "arm": arm,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "asked_at": datetime.now(timezone.utc).isoformat(),
                "prompt": incident[arm],
                "answer": text,
                "claimed_cause": claimed_cause(text),
                "followed_contract": claimed_cause(text) is not None,
                "usage": usage,
                # The label is NOT written here. It lives in the incident file
                # and in the ground-truth log; copying it beside the answer is
                # how a scorer ends up reading it by accident (IA-50 scores
                # blind).
            }
            records.append(record)
            if verbose:
                print("  %-28s %-5s -> %s%s" % (
                    incident["incident_id"], arm,
                    record["claimed_cause"] or "(no ROOT CAUSE line)",
                    "" if record["followed_contract"] else "  CONTRACT NOT FOLLOWED"))

    out_dir.mkdir(parents=True, exist_ok=True)
    answers = out_dir / "answers.jsonl"
    with answers.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "budget.json").write_text(
        json.dumps(budget.report(), indent=2), encoding="utf-8")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incidents", default=str(DEFAULT_INCIDENTS))
    ap.add_argument("--out", default=None,
                    help="run directory; defaults to a timestamped one")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N incidents, for a smoke test")
    ap.add_argument("--plan", action="store_true",
                    help="show what would be sent and exit; makes no calls")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=400_000)
    ap.add_argument("--max-usd", type=float, default=2.00)
    ap.add_argument("--allow-unusable", action="store_true",
                    help="run against a corpus the checks reject. There is no "
                         "good reason; it exists so that using it is a visible "
                         "choice recorded in the run directory.")
    args = ap.parse_args(argv)

    incidents = load_incidents(Path(args.incidents))
    if args.limit:
        incidents = incidents[:args.limit]

    print("%d incident(s) x %d arms = %d calls, model %s, temperature %s"
          % (len(incidents), len(ARMS), len(incidents) * len(ARMS), MODEL, TEMPERATURE))
    for incident in incidents:
        print("  %-28s %-3s arm_a %5d chars   arm_b %6d chars%s"
              % (incident["incident_id"], incident["fault"],
                 len(incident["arm_a"]), len(incident["arm_b"]),
                 "   CONTAMINATED" if incident.get("contaminated_by") else ""))

    reasons = refuse_unusable(incidents)
    if reasons:
        print("\nThis corpus does not support the comparison:", file=sys.stderr)
        for reason in reasons:
            print("  - %s" % reason, file=sys.stderr)
        if not args.allow_unusable:
            print("\nRefusing to spend calls on it. Fix the corpus, or pass "
                  "--allow-unusable and say so in the write-up.", file=sys.stderr)
            return 1
        print("\n!! Continuing anyway because --allow-unusable was passed.",
              file=sys.stderr)

    if not args.go:
        print("\n  --plan only. Add --go to make the calls.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in this shell. D6: the key lives in "
              "the operator's environment, never in a repository secret.",
              file=sys.stderr)
        return 1

    try:
        import anthropic
    except ImportError:
        print("anthropic is not installed. python -m pip install anthropic",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else (
        DEFAULT_RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    gt.check_location(out_dir / "answers.jsonl")   # never inside a repo

    budget = ab.RunBudget(MODEL, max_iterations=len(incidents) * len(ARMS) + 1,
                          max_tokens=args.max_tokens, max_usd=args.max_usd,
                          max_output_per_call=MAX_OUTPUT_TOKENS)
    print()
    records = run(incidents, anthropic.Anthropic(), budget, out_dir)

    print("\n" + budget.summary())
    broke = [r for r in records if not r["followed_contract"]]
    if broke:
        print("\n!! %d answer(s) did not end with a ROOT CAUSE line. They are "
              "recorded as such, not repaired." % len(broke), file=sys.stderr)
    print("\n%d answers -> %s" % (len(records), out_dir))
    print("Scoring is IA-50 and is done blind: the answers carry no label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
