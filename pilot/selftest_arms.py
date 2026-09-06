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
    check("neither arm's prompt contains the other's answer",
          all(other["answer"] not in call["messages"][0]["content"]
              for other in records for call in client.calls),
          "an arm that sees the other's answer measures memory, not context")

    # ---- everything but the incident text is identical ----
    # Which call is which arm is no longer decided by position: IA-49 asks for
    # the order to be randomised, so it comes from the records. This check used
    # to unpack client.calls positionally and broke the moment the shuffle
    # landed the other way -- correctly, and worth keeping in mind: any test
    # that assumes arm A goes first is now testing the shuffle by accident.
    by_arm = {record["arm"]: index for index, record in enumerate(records)}
    a_call = client.calls[by_arm["arm_a"]]
    b_call = client.calls[by_arm["arm_b"]]
    for field in ("model", "system", "max_tokens"):
        check("the arms share the same %s" % field, a_call[field] == b_call[field],
              "%r vs %r" % (a_call[field], b_call[field]))
    check("the two calls pass exactly the same set of parameters",
          set(a_call) == set(b_call), "%s vs %s" % (sorted(a_call), sorted(b_call)))

    # anthropic 1.0.0 does not accept temperature. Passing it would raise; and
    # smuggling it through extra_body would put a value in the record that
    # nobody could verify was applied. Neither happens, and a test says so
    # rather than a comment.
    check("no sampling parameter is passed that the SDK does not declare",
          "temperature" not in a_call and "extra_body" not in a_call,
          str(sorted(a_call)))
    check("and the record says the sampling was not controlled",
          records[0]["sampling"] == ra.SAMPLING and "not exposed" in ra.SAMPLING,
          records[0].get("sampling"))
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

    # ---- IA-49 crit. 1: an answer is on disk the moment it arrives ----
    # The first version built the whole list and wrote it after the loop. A
    # failure on the ninth call would have thrown away eight answers already
    # paid for, and criterion 1 says a missing transcript is a missing data
    # point -- which it cannot be if the transcripts only live in memory.
    class ExplodingClient(FakeClient):
        def __init__(self, fail_on):
            FakeClient.__init__(self)
            self.fail_on = fail_on

        class _Messages(FakeMessages):
            pass

    boom = FakeClient()
    original_create = boom.messages.create

    def create_then_fail(**kwargs):
        if len(boom.calls) >= 3:
            raise RuntimeError("the service went away")
        return original_create(**kwargs)

    boom.messages.create = create_then_fail
    crash_dir = out / "crash"
    b2 = ab.RunBudget(ra.MODEL, max_iterations=99, max_tokens=1_000_000)
    try:
        ra.run([incident("A"), incident("B")], boom, b2, crash_dir, verbose=False)
    except RuntimeError:
        pass
    written = (crash_dir / "answers.jsonl").read_text(encoding="utf-8").strip()
    check("answers written before the failure survive it",
          len(written.splitlines()) == 3,
          "%d line(s) on disk after a crash on the 4th call" % len(written.splitlines()))

    # ---- IA-49: the arm asked first is randomised, per incident ----
    orders = {ra.arm_order("INC-%02d" % i, seed=1) for i in range(30)}
    check("the arm asked first is not always the same",
          len(orders) == 2, str(orders))
    check("the order for one incident is reproducible",
          ra.arm_order("INC-07", 1) == ra.arm_order("INC-07", 1))
    check("a different seed can give a different order",
          {ra.arm_order("INC-%02d" % i, 1) for i in range(30)}
          == {("arm_a", "arm_b"), ("arm_b", "arm_a")})
    check("both arms are always asked, whatever the order",
          all(set(o) == set(ra.ARMS) for o in orders))

    # ---- IA-49 crit. 3: the two counters must agree ----
    good = FakeClient()
    b3 = ab.RunBudget(ra.MODEL, max_iterations=99, max_tokens=1_000_000)
    recs = ra.run([incident("A")], good, b3, out / "rec", verbose=False)
    check("a clean run reconciles", ra.reconcile(recs, b3) == [],
          str(ra.reconcile(recs, b3)))
    tampered = [dict(r) for r in recs]
    tampered[0]["usage"] = dict(tampered[0]["usage"], input_tokens=1)
    check("a mismatch between budget and transcripts is reported",
          any("input_tokens" in p for p in ra.reconcile(tampered, b3)),
          "two counters that disagree mean one is measuring something else")

    # ---- IA-49 crit. 4: a truncated answer is marked, not scored ----
    class TruncatingClient(FakeClient):
        def __init__(self):
            FakeClient.__init__(self)

    trunc = FakeClient()
    inner = trunc.messages.create

    def create_truncated(**kwargs):
        response = inner(**kwargs)
        response.stop_reason = "max_tokens"
        return response

    trunc.messages.create = create_truncated
    b4 = ab.RunBudget(ra.MODEL, max_iterations=99, max_tokens=1_000_000)
    capped = ra.run([incident("A")], trunc, b4, out / "cap", verbose=False)
    check("an answer stopped by the output cap is marked capped",
          all(r["capped"] for r in capped))
    check("and a normal answer is not", not any(r["capped"] for r in recs))

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

    # A slice of a varied corpus is judged on the corpus, not on the slice.
    # Checking diversity on the slice made --limit 1 refuse itself, and the
    # obvious escape (--allow-unusable) would also have switched off the
    # contamination check.
    one = [varied[0]]
    check("a single-incident slice of a varied corpus is allowed",
          ra.refuse_unusable(one, varied) == [], str(ra.refuse_unusable(one, varied)))
    check("but that same incident alone as the whole corpus is refused",
          any("fixed rule" in r for r in ra.refuse_unusable(one)),
          "diversity is a property of the corpus; nothing else changed")
    dirty_slice = [incident("D", "F3", "db",
                            contaminated=[{"incident_id": "X", "fault": "F3",
                                           "node_role": "app"}])]
    check("a contaminated incident is still refused inside a varied corpus",
          any("IA-61" in r for r in ra.refuse_unusable(dirty_slice, varied)),
          "contamination is per incident and must survive the split")

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
