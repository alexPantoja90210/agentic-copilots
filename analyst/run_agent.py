"""
Ask the agent all 23 questions, one fresh conversation each, and keep everything.

    python run_agent.py --plan                 # what would be sent, no calls
    python run_agent.py --go --limit 1         # one question, for a smoke test
    python run_agent.py --go

Three rules this file exists to enforce
---------------------------------------
**Each question is its own conversation.** No memory carries between questions.
Question 12 must not be answerable because question 3 already loaded the data.

**The agent computes; it does not recall.** Its only route to a number is a tool
call, and every call is recorded. An answer with no call behind it is written
with `tool_calls: 0` so the scorer can mark it `unsupported` — a right answer
reached without touching the data is luck, and luck does not generalise.

**The console is not the answer key.** Progress names the question and whether
the output contract was followed. Not the answer, not the value, not the
reasoning. That is IA-62's lesson applied at design time instead of as a patch.

D6: the key comes from the operator's environment (ANTHROPIC_API_KEY), never
from a repository secret. This runs locally, by hand, watched.

IA-68, under IA-64. The rules it is run under are PREREGISTRATION.md, whose
commit hash is written into run_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_budget as ab
import agent_tools as at

MODEL = os.environ.get("PILOT_MODEL", "claude-sonnet-4-5")
MAX_OUTPUT_TOKENS = 1500

# Per question. A question that has not resolved in this many exchanges is
# recorded as `no_contract` rather than being allowed to spend without bound.
MAX_EXCHANGES = 8

SAMPLING = "model default; temperature not exposed by the installed SDK"

# The cap is enforced against model-pricing.json, which is marked verified:false.
# It is a stop, not a measurement: it prevents a loop from spending without
# bound. NO USD FIGURE FROM THIS RUN IS PUBLISHABLE (D11). Tokens are.
COST_CAP_USD = 5.00
COST_CAP_NOTE = ("enforced against model-pricing.json, which is marked "
                 "verified:false. The cap is a stop, not a measurement. No USD "
                 "figure from this run may be published; tokens may.")

SYSTEM = (
    "You are a data analyst. You are given two tables and one question about "
    "them, and you answer it by querying the data — never from memory or "
    "assumption.\n\n"
    "Use the tools. Compute what you need. If a question asks how you would do "
    "something, answer with the method and demonstrate it on the data.\n\n"
    "End your reply with exactly one final block that begins:\n"
    "ANSWER: <your answer>\n"
    "or\n"
    "ANSWER: CANNOT COMPUTE\n\n"
    "Use CANNOT COMPUTE when the data does not support an answer. Saying so is "
    "a better outcome than producing a number you cannot justify from what you "
    "queried."
)

ANSWER_PREFIX = "ANSWER:"


class RunError(Exception):
    pass


# What counts as part of the instrument. An untracked .py under analyst/ could
# be a module the run imports; an untracked .idea/ file cannot.
INSTRUMENT_SUFFIXES = (".py", ".md", ".json")


def head_commit(repo_root: Path) -> dict:
    """
    The commit the harness is running from, and whether the INSTRUMENT is pinned.

    The run already refuses unless the RULES are committed. Recording the rules
    while leaving the instrument unnamed applies the principle to half the
    problem: answers produced by uncommitted code cannot be traced to the
    version that produced them.

    The first version of this asked `git status --porcelain -- analyst` and
    called any output dirty. It then reported the instrument unpinned because
    PyCharm had written `analyst/.idea/`. That is the recurring defect again --
    a check that cannot distinguish two situations it treats as one -- and it
    is the kind that matters most here, because a warning that fires on IDE
    noise is a warning the operator learns to scroll past.

    So the three cases are separated and all three are recorded:

      tracked_modified   source under analyst/ changed since its commit
      untracked_source   a new .py/.md/.json that the run might be importing
      untracked_other    everything else, reported and NOT counted as dirty

    A dirty tree still does not stop the run. It is recorded, so a run whose
    instrument was not pinned says so in its own artefacts.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=20)
        status = subprocess.run(["git", "status", "--porcelain", "--", "analyst"],
                                cwd=str(repo_root), capture_output=True,
                                text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "analyst_tree_dirty": None, "error": str(exc)}

    tracked, new_source, other = [], [], []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if line.startswith("??"):
            (new_source if path.endswith(INSTRUMENT_SUFFIXES) else other).append(path)
        else:
            tracked.append(path)
    return {
        "commit": head.stdout.strip() or None,
        "tracked_modified": tracked,
        "untracked_source": new_source,
        "untracked_other": other,
        "analyst_tree_dirty": bool(tracked or new_source),
    }


def preregistration_commit(repo_root: Path) -> str:
    """
    The commit that last touched PREREGISTRATION.md.

    Recorded so a run can prove which version of the rules it ran under. If it
    cannot be determined, the run refuses: a run that cannot name its rules is
    a run whose rules can be rewritten afterwards.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "analyst/PREREGISTRATION.md"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunError("could not read the pre-registration commit: %s" % exc)
    commit = out.stdout.strip()
    if not commit:
        raise RunError(
            "PREREGISTRATION.md has no commit in %s. The rules must be "
            "committed before the first call, and the run must be able to name "
            "which version it ran under." % repo_root)
    return commit


def load_questions(reference: Path) -> list[dict]:
    path = reference / "questions.json"
    if not path.exists():
        raise RunError("no questions.json at %s. Run extract_questions.py first."
                       % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(q, kind="objective") for q in data["objective"]]
    rows += [dict(q, kind="subjective") for q in data["subjective"]]
    if len(rows) != 23:
        raise RunError("expected 23 questions, found %d" % len(rows))
    return rows


def load_corpus(corpus: Path):
    import pandas as pd
    tickets = pd.read_csv(corpus / "tickets.csv", parse_dates=["Fecha"])
    agents = pd.read_csv(corpus / "agents.csv")
    if len(tickets) != 97498 or len(agents) != 50:
        raise RunError("corpus is %d x %d, expected 97498 x 50"
                       % (len(tickets), len(agents)))
    return tickets, agents


def build_prompt(question: dict) -> str:
    """
    What the agent is sent. The question and nothing else.

    No hint of the expected shape of the answer, no worked example, and above
    all nothing from the reference directory beyond the question text itself.
    `assert_no_leak` checks that last part against the baseline rather than
    trusting this function.
    """
    return question["question"]


def assert_no_leak(prompts: list[str], reference: Path) -> None:
    """
    No baseline figure may appear in any prompt.

    The harness process reads the quarantine — it has to, the questions live
    there. The AGENT must not. This is the same asymmetry as IA-65: the checker
    may know the answer, the subject may not. Checked here rather than assumed,
    because IA-45 was invalidated by a leak nobody could see from the inside.
    """
    path = reference / "baseline.json"
    if not path.exists():
        raise RunError("no baseline.json at %s. Run baseline.py first." % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    values = set()
    for row in data.get("comparison", []):
        for key in ("human", "computed"):
            text = str(row.get(key, "")).strip()
            if len(text) >= 4:
                values.add(text)
    blob = "\n".join(prompts)
    hits = sorted(v for v in values if v in blob)
    if hits:
        raise RunError(
            "%d baseline value(s) appear in the prompts: %s. The answer is "
            "inside the evidence, which is the defect that invalidated IA-45."
            % (len(hits), hits[:5]))


def claimed_answer(text: str) -> str | None:
    """
    Everything after the last ANSWER: marker, or None if there is no marker.

    Extraction only. It does not decide whether the answer is right and it does
    not go hunting for a number in the prose: an answer that did not follow the
    contract is recorded as not following it. Guessing what the model meant is
    the scorer's job, and the scorer applies a pre-registered rule.
    """
    index = text.rfind(ANSWER_PREFIX)
    if index < 0:
        return None
    return text[index + len(ANSWER_PREFIX):].strip()


def ask(client, question: dict, tickets, agents, budget) -> dict:
    """One question, one fresh conversation, tools available, everything logged."""
    messages = [{"role": "user", "content": build_prompt(question)}]
    calls, exchanges, text = [], 0, ""
    started = time.time()

    while exchanges < MAX_EXCHANGES:
        budget.begin_iteration()
        budget.before_call()
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM,
            tools=at.TOOLS, messages=messages)
        budget.record_response(response)
        exchanges += 1

        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text")
        uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for use in uses:
            budget.record_tool(use.name)
            out, is_error = at.dispatch(use.name, use.input or {}, tickets, agents)
            calls.append({"tool": use.name, "input": use.input,
                          "result": out[:2000], "error": is_error})
            results.append({"type": "tool_result", "tool_use_id": use.id,
                            "content": out, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    answer = claimed_answer(text)
    return {
        "id": question["id"], "kind": question["kind"],
        "question": question["question"],
        "asked_at": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.time() - started, 2),
        "exchanges": exchanges,
        "tool_calls": len(calls),
        "calls": calls,
        "reply": text,
        "answer": answer,
        # An absent answer is recorded as absent. It is never written as an
        # empty value, which a scorer could not tell from a legitimate blank.
        "followed_contract": answer is not None,
        "abstained": bool(answer and answer.strip().upper().startswith("CANNOT COMPUTE")),
        "hit_exchange_cap": exchanges >= MAX_EXCHANGES and answer is None,
    }


def run(questions, client, tickets, agents, budget, out_dir: Path,
        verbose=True) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / "answers.jsonl").open("w", encoding="utf-8")
    records = []
    try:
        for question in questions:
            record = ask(client, question, tickets, agents, budget)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if verbose:
                # IA-62: the question and whether the contract was followed.
                # Not the answer. Not the value. Watchable, not an answer key.
                print("  %-4s %-11s %2d tool call(s)  %5.1fs  contract %s%s"
                      % (record["id"], record["kind"], record["tool_calls"],
                         record["seconds"],
                         "ok" if record["followed_contract"] else "NOT FOLLOWED",
                         "  (abstained)" if record["abstained"] else ""))
    finally:
        handle.close()
    (out_dir / "budget.json").write_text(
        json.dumps(budget.report(), indent=2), encoding="utf-8")
    return records


def reconcile(records: list[dict], budget) -> list[str]:
    """
    The budget's own count of calls against the transcripts. IA-49 criterion 3:
    if these disagree the discrepancy IS the finding, and neither figure may be
    quoted until it is known which one is measuring something else.
    """
    report = budget.report()
    problems = []
    exchanges = sum(r["exchanges"] for r in records)
    if report["calls"] != exchanges:
        problems.append("calls: the budget counted %d, the transcripts record %d "
                        "exchanges" % (report["calls"], exchanges))
    return problems


def run_config(questions, out_dir, prereg_commit, caps, harness=None) -> dict:
    return {
        "model": MODEL, "sampling": SAMPLING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_exchanges_per_question": MAX_EXCHANGES,
        "caps": caps,
        "cost_cap_note": COST_CAP_NOTE,
        "preregistration_commit": prereg_commit,
        # IA-68: the instrument names itself too, not only its rules.
        "harness": harness or {},
        "tools": [t["name"] for t in at.TOOLS],
        "questions": [q["id"] for q in questions],
        "system_prompt": SYSTEM,
        # IA-62 applied here from the start: this runner has no mode that prints
        # answers, so there is no flag to record.
        "console_prints_answers": False,
    }


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=r"C:\dev\ia-analyst\corpus")
    ap.add_argument("--reference", default=r"C:\dev\ia-analyst\reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--max-usd", type=float, default=COST_CAP_USD)
    args = ap.parse_args(argv)

    reference, corpus = Path(args.reference), Path(args.corpus)
    try:
        questions = load_questions(reference)
        if args.limit:
            questions = questions[:args.limit]
        assert_no_leak([build_prompt(q) for q in questions], reference)
        tickets, agents = load_corpus(corpus)
        prereg = preregistration_commit(here.parent)
    except RunError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    objective = sum(1 for q in questions if q["kind"] == "objective")
    print("%d question(s): %d objective, %d subjective. Model %s."
          % (len(questions), objective, len(questions) - objective, MODEL))
    print("  pre-registration commit %s" % prereg[:12])
    harness = head_commit(here.parent)
    print("  harness commit          %s%s"
          % ((harness.get("commit") or "unknown")[:12],
             "  !! INSTRUMENT NOT PINNED: %s"
             % ", ".join(harness.get("tracked_modified", [])
                         + harness.get("untracked_source", []))
             if harness.get("analyst_tree_dirty") else ""))
    ignored = harness.get("untracked_other") or []
    if ignored:
        print("  (%d untracked non-source file(s) under analyst/, not part of "
              "the instrument: %s)" % (len(ignored), ", ".join(ignored[:3])))
    print("  corpus %d tickets x %d agents" % (len(tickets), len(agents)))
    print("  prompts leak-checked against baseline.json: clean")
    print("  cost cap $%.2f — %s" % (args.max_usd, COST_CAP_NOTE))

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
        Path(r"C:\dev\ia-analyst\runs") /
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    caps = {"tokens": args.max_tokens, "usd": args.max_usd,
            "iterations": len(questions) * MAX_EXCHANGES + 1}
    budget = ab.RunBudget(MODEL, max_iterations=caps["iterations"],
                          max_tokens=args.max_tokens, max_usd=args.max_usd,
                          max_output_per_call=MAX_OUTPUT_TOKENS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config(questions, out_dir, prereg, caps,
                              head_commit(here.parent)), indent=2,
                   ensure_ascii=False), encoding="utf-8")

    print()
    records = run(questions, anthropic.Anthropic(), tickets, agents, budget, out_dir)

    problems = reconcile(records, budget)
    if problems:
        print("\n!! The budget and the transcripts disagree. That discrepancy is "
              "the finding:", file=sys.stderr)
        for problem in problems:
            print("!!   %s" % problem, file=sys.stderr)

    print("\n" + budget.summary())
    broke = [r for r in records if not r["followed_contract"]]
    if broke:
        print("\n!! %d answer(s) did not end with an ANSWER line. Recorded as "
              "such, not repaired." % len(broke), file=sys.stderr)
    no_tools = [r for r in records if r["tool_calls"] == 0]
    if no_tools:
        print("!! %d answer(s) were produced with no tool call. The scorer will "
              "mark those `unsupported` however right they look." % len(no_tools),
              file=sys.stderr)
    print("\n%d answers -> %s" % (len(records), out_dir))
    print("Scoring is IA-69 and applies PREREGISTRATION.md at commit %s."
          % prereg[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
