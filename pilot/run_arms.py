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
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_budget as ab
import ground_truth as gt

MODEL = os.environ.get("PILOT_MODEL", "claude-sonnet-4-5")
MAX_OUTPUT_TOKENS = 900

# Sampling is NOT controlled, and that is a finding rather than an omission.
#
# This runner was written to pass temperature=0 for reproducibility. The
# installed SDK (anthropic 1.0.0) does not expose `temperature` on
# messages.create at all -- the parameter list is max_tokens, messages, model,
# cache_control, container, inference_geo, metadata, output_config,
# service_tier, stop_sequences, stream, system, thinking, tool_choice, tools,
# user_profile_id -- and `output_config` carries only `effort` and `format`.
#
# `extra_body` exists and temperature could be smuggled through it. It is
# deliberately NOT done: a parameter the SDK no longer declares is a parameter
# nobody can verify was applied, and the run record would then claim a
# temperature the calls may not have used. An absent control that is declared
# beats a present one that cannot be checked.
#
# What this costs, stated plainly: both arms get identical treatment, so the
# comparison between them stands. What is lost is REPRODUCIBILITY -- running the
# pilot twice may not produce the same answers, and every number is one sample
# per arm rather than a mean. The write-up says so, and so does every record
# this file writes.
SAMPLING = "model default; temperature not exposed by anthropic 1.0.0"

# Structured output (`output_config.format`) is available and is deliberately
# unused. Forcing JSON would make extraction reliable, and would also change the
# task: the paper's baseline answers in prose, and whether the model honours a
# stated output contract is itself something worth measuring rather than
# enforcing. `followed_contract` in each record is that measurement.

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


def refuse_unusable(selected: list[dict], corpus: list[dict] | None = None) -> list[str]:
    """
    Reasons not to spend money, checked before the first call.

    Two kinds of reason, and they are judged against two different things --
    which is the whole point of the second argument.

    **Per incident**, against what will actually be asked: a contaminated
    prompt, or a window with no usable datapoints. Those are properties of the
    incidents being sent.

    **Per corpus**, against the whole file: whether the fault classes and cause
    nodes vary enough that a fixed rule could not score as well as reasoning.
    That is a property of the CORPUS, not of any subset of it. Judging it on the
    slice made `--limit 1` refuse itself -- one incident has one class and one
    node by definition -- and the obvious way out would have been
    `--allow-unusable`, which also switches off the contamination check. A guard
    that pushes the operator into disabling a different guard is a bad guard.
    """
    corpus = corpus if corpus is not None else selected
    reasons = []
    dirty = [i["incident_id"] for i in selected if i.get("contaminated_by")]
    if dirty:
        reasons.append(
            "%d incident(s) carry another incident's fault inside the context "
            "their prompt shows (IA-61): %s. Scoring these mixes 'the context "
            "did not help' with 'the context contained something else'."
            % (len(dirty), ", ".join(dirty)))
    unusable = [i["incident_id"] for i in selected if not i.get("usable")]
    if unusable:
        reasons.append("no usable metric datapoints: %s" % ", ".join(unusable))
    classes = {i["fault"] for i in corpus}
    roles = {i["fault_role"] for i in corpus}
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


def arm_order(incident_id: str, seed: int) -> tuple:
    """
    Which arm is asked first, decided per incident and reproducibly.

    IA-49 asks for the order to be randomised. Always asking arm A first means
    any drift in the service over a session -- load, latency, a model rolled
    forward mid-run -- lands systematically on arm B. The bias would be small
    and it would be entirely one-directional, which is the worst kind.

    Keyed on the incident id rather than on call position, so the order for a
    given incident is the same however the corpus is sliced with --limit.
    """
    order = list(ARMS)
    random.Random("%s:%d" % (incident_id, seed)).shuffle(order)
    return tuple(order)


def run(incidents, client, budget, out_dir: Path, verbose=True,
        verbose_arms=False, seed=0) -> list[dict]:
    """
    Ask every incident once per arm, writing each answer as it arrives.

    Records are appended to disk immediately. The first version built the whole
    list and wrote it after the loop: a failure on the ninth call would have
    thrown away eight answers that had already been paid for. IA-49 criterion 1
    says a missing transcript is a missing data point -- it cannot be that if
    the transcripts only exist in memory.

    What the console is allowed to say (IA-62)
    ------------------------------------------
    This function used to print one line per ANSWER carrying the arm and the
    claimed cause. `score_arms.py --rate` then went to considerable trouble to
    hide exactly that, and the runner had already printed it. Blinding
    implemented at the scorer while the producer prints the answer key is
    blinding in one file, not in the system.

    So progress is now reported once per INCIDENT, after both arms are done,
    and it names the incident and how many answers followed the output
    contract. Not the arm. Not the claimed cause.

    Per-incident rather than per-answer for a reason that is not obvious:
    `arm_order` is deterministic in (incident_id, seed) and the seed is written
    into run_config.json, so two lines printed in the order they were asked
    would still name the arms to anyone who ran the same one-line function.
    Withholding the label and leaking the order is the same defect wearing a
    different hat.

    `verbose_arms=True` restores the old output for debugging. It is recorded
    in run_config.json by the caller, so a run whose blinding was voided says
    so in its own artefacts instead of relying on somebody remembering.

    Residual, stated rather than hidden: the capped count is per incident, and
    arm B's prompt is the long one, so a capped answer is a good guess for arm
    B. That is the inference score_arms.py already documents -- removing the
    label hides the name, not the inference. Capped answers are excluded from
    the comparison anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    records = []
    handle = answers_path.open("w", encoding="utf-8")
    try:
        for incident in incidents:
            here = []
            for arm in arm_order(incident["incident_id"], seed):
                # A fresh message list every time. Nothing carries between arms,
                # and nothing carries between incidents.
                text, usage = ask(client, incident[arm], budget)
                record = {
                    "incident_id": incident["incident_id"],
                    "arm": arm,
                    "model": MODEL,
                    "sampling": SAMPLING,
                    "asked_at": datetime.now(timezone.utc).isoformat(),
                    "prompt": incident[arm],
                    "answer": text,
                    "claimed_cause": claimed_cause(text),
                    "followed_contract": claimed_cause(text) is not None,
                    # A truncated answer is not a wrong answer. Scoring one as
                    # wrong would flatter arm A, whose prompt is twenty times
                    # shorter and which will never be the one that runs out.
                    "capped": usage.get("stop_reason") == "max_tokens",
                    "usage": usage,
                    # The label is NOT written here. It lives in the incident
                    # file and in the ground-truth log; copying it beside the
                    # answer is how a scorer ends up reading it by accident
                    # (IA-50 scores blind).
                }
                records.append(record)
                here.append(record)
                # Written and flushed per answer regardless of what is printed:
                # IA-62 delays the operator SEEING the mapping, it does not
                # withhold it from the run directory.
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if verbose_arms:
                    print("  !! %-28s %-5s -> %s%s%s" % (
                        incident["incident_id"], arm,
                        record["claimed_cause"] or "(no ROOT CAUSE line)",
                        "" if record["followed_contract"] else "  CONTRACT NOT FOLLOWED",
                        "  CAPPED" if record["capped"] else ""))
            if verbose:
                print("  %-28s  contract %d/%d%s%s" % (
                    incident["incident_id"],
                    sum(1 for r in here if r["followed_contract"]), len(here),
                    "  !! CONTRACT NOT FOLLOWED"
                    if any(not r["followed_contract"] for r in here) else "",
                    "  %d CAPPED" % sum(1 for r in here if r["capped"])
                    if any(r["capped"] for r in here) else ""))
    finally:
        handle.close()

    (out_dir / "budget.json").write_text(
        json.dumps(budget.report(), indent=2), encoding="utf-8")
    return records


def reconcile(records: list[dict], budget) -> list[str]:
    """
    The budget's own totals against the sum of what the transcripts recorded.

    IA-49 criterion 3: if these disagree, the discrepancy IS the finding. Two
    counters that are supposed to track the same thing and do not means one of
    them is measuring something else, and neither can be quoted until it is
    known which.
    """
    report = budget.report()
    problems = []
    for field in ("input_tokens", "output_tokens"):
        from_records = sum((r["usage"] or {}).get(field) or 0 for r in records)
        if from_records != report[field]:
            problems.append(
                "%s: the budget counted %d, the transcripts sum to %d (difference %+d)"
                % (field, report[field], from_records, from_records - report[field]))
    if report["calls"] != len(records):
        problems.append("calls: the budget counted %d, %d transcripts were written"
                        % (report["calls"], len(records)))
    return problems


def run_config(incidents, seed: int, caps: dict, verbose_arms: bool) -> dict:
    """
    What the run records about itself, beside its answers.

    A separate function because IA-49 criterion 2 and IA-62 criterion 3 both
    say a control that lives only in the code cannot be checked against the run
    it was supposed to bound -- and a control that lives only inside `main()`
    cannot be checked by a test either.
    """
    return {
        "model": MODEL, "sampling": SAMPLING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "caps": caps,
        "arm_order_seed": seed,
        "incidents": [i["incident_id"] for i in incidents],
        "system_prompt": SYSTEM,
        # IA-62 criterion 3. Written on every run, true or false, because the
        # useful thing is not the flag but the RECORD: a scorer reading this
        # directory months later can tell whether the usefulness rating from it
        # may be called blind, without asking anyone what they remember.
        "verbose_arms": verbose_arms,
        "blinding": (
            "VOIDED: --verbose-arms printed the arm-to-answer mapping to the "
            "console during this run. The usefulness rating collected from it "
            "is not blind and must be reported with that caveat. The accuracy "
            "metric is mechanical and is unaffected."
            if verbose_arms else
            "console output withheld the arm and the claimed cause, so the "
            "usefulness rating may be collected blind. This says nothing about "
            "what a rater may infer from an answer's own content -- see "
            "score_arms.py."),
    }


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
    ap.add_argument("--seed", type=int, default=20260906,
                    help="decides which arm is asked first for each incident. "
                         "Recorded in run_config.json so the order can be "
                         "reproduced and audited.")
    ap.add_argument("--verbose-arms", action="store_true",
                    help="print which arm produced which answer while the run "
                         "is in flight. This VOIDS the blind usefulness rating "
                         "(IA-62) and is recorded in run_config.json so the run "
                         "declares it rather than depending on memory. The "
                         "accuracy metric is mechanical and is unaffected.")
    ap.add_argument("--allow-unusable", action="store_true",
                    help="run against a corpus the checks reject. There is no "
                         "good reason; it exists so that using it is a visible "
                         "choice recorded in the run directory.")
    args = ap.parse_args(argv)

    corpus = load_incidents(Path(args.incidents))
    incidents = corpus[:args.limit] if args.limit else corpus
    if args.limit:
        print("--limit %d: asking %d of %d incidents. The corpus-level checks "
              "still judge all %d."
              % (args.limit, len(incidents), len(corpus), len(corpus)))

    print("%d incident(s) x %d arms = %d calls, model %s\n  sampling: %s"
          % (len(incidents), len(ARMS), len(incidents) * len(ARMS), MODEL, SAMPLING))
    for incident in incidents:
        print("  %-28s %-3s arm_a %5d chars   arm_b %6d chars%s"
              % (incident["incident_id"], incident["fault"],
                 len(incident["arm_a"]), len(incident["arm_b"]),
                 "   CONTAMINATED" if incident.get("contaminated_by") else ""))

    reasons = refuse_unusable(incidents, corpus)
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
    # The caps and the arm-order seed belong in the results, not only in this
    # file: IA-49 criterion 2. A cap that lives only in the code cannot be
    # checked against the run it was supposed to bound.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(
        run_config(incidents, args.seed,
                   {"iterations": len(incidents) * len(ARMS) + 1,
                    "tokens": args.max_tokens, "usd": args.max_usd},
                   args.verbose_arms),
        indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    if args.verbose_arms:
        print("!! --verbose-arms: the arm-to-answer mapping will be printed. "
              "The usefulness\n!! rating from this run is NOT blind, and "
              "run_config.json records that.\n", file=sys.stderr)
    records = run(incidents, anthropic.Anthropic(), budget, out_dir,
                  verbose_arms=args.verbose_arms, seed=args.seed)

    problems = reconcile(records, budget)
    if problems:
        print("\n!! The budget and the transcripts disagree. That discrepancy is "
              "the finding:", file=sys.stderr)
        for problem in problems:
            print("!!   %s" % problem, file=sys.stderr)
        print("!! Neither figure can be quoted until it is known which one is "
              "measuring something else.", file=sys.stderr)

    print("\n" + budget.summary())
    broke = [r for r in records if not r["followed_contract"]]
    if broke:
        print("\n!! %d answer(s) did not end with a ROOT CAUSE line. They are "
              "recorded as such, not repaired." % len(broke), file=sys.stderr)
    capped = [r for r in records if r["capped"]]
    if capped:
        print("!! %d answer(s) hit the output cap and are marked `capped`. They "
              "are excluded from the accuracy count: a truncated answer scored "
              "as wrong would flatter arm A, whose prompt never runs out."
              % len(capped), file=sys.stderr)
    print("\n%d answers -> %s" % (len(records), out_dir))
    print("Scoring is IA-50 and is done blind: the answers carry no label, and "
          "this console\nnamed neither the arm nor the claimed cause (IA-62)."
          if not args.verbose_arms else
          "Scoring is IA-50. The usefulness rating from this run is NOT blind: "
          "--verbose-arms\nprinted the mapping above. run_config.json says so too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
