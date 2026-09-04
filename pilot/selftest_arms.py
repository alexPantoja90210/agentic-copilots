"""
Selftest for the two-arm runner.

The properties under test are the ones that decide whether the experiment
measures anything:

  - each arm is its own conversation, so arm B cannot see arm A's answer;
  - everything except the incident text is identical between the arms;
  - the answers carry no label, because IA-50 scores them blind;
  - a corpus that cannot support the comparison is refused BEFORE the calls.

No network, no key, no spend: the client is a fake that records what it was
asked and answers from a script.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# agent_budget lives at the repository root, beside the copilots that share it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_budget as ab
import run_arms as ra

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


class FakeUsage:
    def __init__(self):
        self.input_tokens = 1000
        self.output_tokens = 60


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage()
        self.stop_reason = "end_turn"


class FakeMessages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        return FakeResponse(self.outer.answers.pop(0) if self.outer.answers
                            else "Looks like db.\nROOT CAUSE: db")


class FakeClient:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = list(answers or [])
        self.messages = FakeMessages(self)


def incident(iid="F3-db-1", fault="F3", role="db", contaminated=None):
    return {"incident_id": iid, "fault": fault, "fault_role": role,
            "symptomatic_node": "web", "usable": True,
            "contaminated_by": contaminated or [],
            "arm_a": "TITLE %s\n\nSummary of %s." % (iid, iid),
            "arm_b": "TITLE %s\n\nSummary of %s.\n\nGRAPH\nMETRICS" % (iid, iid)}


def run() -> int:
    out = Path(tempfile.mkdtemp())

    client = FakeClient(["a.\nROOT CAUSE: app", "b.\nROOT CAUSE: db"])
    budget = ab.RunBudget(ra.MODEL, max_iterations=10, max_tokens=1_000_000)
    records = ra.run([incident()], client, budget, out / "r1", verbose=False)

    # ---- isolation: two calls, neither carrying the other's history ----
    check("one incident produces one call per arm", len(client.calls) == 2,
          "%d calls" % len(client.calls))
    check("every call sends exactly one user message and no history",
          all(len(c["messages"]) == 1 and c["messages"][0]["role"] == "user"
              for c in client.calls),
          str([len(c["messages"]) for c in client.calls]))
    first_answer = records[0]["answer"]
    check("arm B's prompt does not contain arm A's answer",
          first_answer not in client.calls[1]["messages"][0]["content"],
          "arm B would be measuring memory, not context")

    # ---- everything but the incident text is identical ----
    a_call, b_call = client.calls
    for field in ("model", "system", "temperature", "max_tokens"):
        check("the arms share the same %s" % field, a_call[field] == b_call[field],
              "%r vs %r" % (a_call[field], b_call[field]))
    check("and differ only in the message they carry",
          a_call["messages"][0]["content"] != b_call["messages"][0]["content"])
    check("arm A's text is inside arm B's, as IA-48 guarantees",
          a_call["messages"][0]["content"] in b_call["messages"][0]["content"])

    # ---- the system prompt must not hand over the answer space ----
    system = a_call["system"]
    check("the system prompt names no fault class",
          not any(token in system for token in ("F0", "F1", "F2", "F3",
                                                "cpu_saturation", "credit",
                                                "instance_unavailable")),
          system)
    check("and leaves 'cannot tell' reachable", "INSUFFICIENT" in system)

    # ---- the answers carry no label ----
    for record in records:
        check("the record for %s carries no label" % record["arm"],
              "fault_role" not in record and "fault" not in record,
              str(sorted(record)))
    check("what was asked is kept verbatim, not summarised",
          records[0]["prompt"] == client.calls[0]["messages"][0]["content"])

    # ---- the ROOT CAUSE line is extracted, never inferred ----
    check("a well-formed answer yields its claimed cause",
          records[0]["claimed_cause"] == "app" and records[1]["claimed_cause"] == "db",
          str([r["claimed_cause"] for r in records]))
    check("INSUFFICIENT survives as itself, not as a missing answer",
          ra.claimed_cause("nothing stands out.\nROOT CAUSE: INSUFFICIENT")
          == "INSUFFICIENT")
    check("an answer with no contract line is recorded as not following it",
          ra.claimed_cause("I think db failed, honestly.") is None,
          "guessing what the model meant is the human scorer's job")
    check("the last ROOT CAUSE line wins, not the first mention",
          ra.claimed_cause("ROOT CAUSE: app\nOn reflection:\nROOT CAUSE: db") == "db")

    # ---- what must be refused before any money is spent ----
    contaminated = [incident(contaminated=[{"incident_id": "OTHER", "fault": "F3",
                                            "node_role": "app"}])]
    reasons = ra.refuse_unusable(contaminated)
    check("a contaminated corpus is refused", any("IA-61" in r for r in reasons),
          str(reasons))

    one_class = [incident("A", "F3", "db"), incident("B", "F3", "db")]
    reasons = ra.refuse_unusable(one_class)
    check("a corpus with one fault class and one node is refused",
          any("fixed rule" in r for r in reasons), str(reasons))

    varied = [incident("A", "F3", "db"), incident("B", "F1", "web")]
    check("a varied, clean corpus is accepted", ra.refuse_unusable(varied) == [],
          str(ra.refuse_unusable(varied)))

    blind = [incident("A", "F3", "db"), incident("B", "F1", "web")]
    blind[0]["usable"] = False
    check("an incident with no usable datapoints is refused",
          any("no usable" in r for r in ra.refuse_unusable(blind)),
          str(ra.refuse_unusable(blind)))

    # ---- the budget is consulted, not merely reported ----
    tight = ab.RunBudget(ra.MODEL, max_iterations=1, max_tokens=1_000_000)
    try:
        ra.run([incident()], FakeClient(), tight, out / "r2", verbose=False)
        check("the iteration cap stops the run", False, "it ran past the cap")
    except ab.IterationCapExceeded:
        check("the iteration cap stops the run", True)

    written = json.loads((out / "r1" / "answers.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    check("answers are written where a blind scorer can read them",
          written["incident_id"] == "F3-db-1" and "answer" in written)
    check("and the budget report travels with them",
          (out / "r1" / "budget.json").exists())

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
