"""
Selftest for the agent harness and its sandbox.

The properties under test are the ones that decide whether the run measures
anything: the agent cannot leave its namespace, cannot see the answers, cannot
be credited for a number it did not compute, and the console cannot become the
answer key.

No network, no key, no spend: the client is a fake that answers from a script.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_budget as ab
import agent_tools as at
import run_agent as ra

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


# ---- a fake client that speaks the tool-use protocol ----------------------
class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Usage:
    def __init__(self):
        self.input_tokens, self.output_tokens = 900, 120


class Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content, self.stop_reason, self.usage = content, stop_reason, Usage()


class Messages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kw):
        # Snapshot the message list. Storing the reference recorded what the
        # conversation grew into, not what was sent -- so a check on "no history
        # was carried in" was reading the history added afterwards and failing a
        # harness that was correct. The instrument was measuring the wrong
        # object, which is the defect this project keeps finding, this time in a
        # test fixture.
        self.outer.calls.append(dict(kw, messages=list(kw["messages"])))
        return self.outer.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.calls, self.script = [], list(script)
        self.messages = Messages(self)


def text_then(answer):
    return Response([Block(type="text", text="Working.\n\nANSWER: %s" % answer)])


def tool_then(expression, answer):
    return [
        Response([Block(type="text", text="Let me compute that."),
                  Block(type="tool_use", id="t1", name="run_query",
                        input={"expression": expression})], "tool_use"),
        text_then(answer),
    ]


def corpus():
    tickets = pd.DataFrame({
        "ID Ticket": ["A", "B", "C", "D"],
        "Fecha": pd.to_datetime(["2016-01-01"] * 4),
        "Request Category": ["System", "System", "Software", "Hardware"],
        "Resolution Time (Days)": [1, 3, 5, 7],
        "Import Date": ["x", "y", "z", "w"],   # a column that trips a naive check
    })
    agents = pd.DataFrame({"Agent ID": [1, 2], "Full Name": ["Ann", "Bo"],
                           "Email": ["a@x.com", "b@x.com"]})
    return tickets, agents


def run() -> int:
    tickets, agents = corpus()

    # ---- the sandbox refuses what leaves the namespace --------------------
    for bad, why in [
        ("__import__('os').system('ls')", "dunder import"),
        ("tickets.__class__.__mro__", "dunder attribute walk"),
        ("open('/etc/passwd').read()", "file access"),
        ("os.listdir('.')", "an unavailable name"),
        ("tickets.groupby('x').size(); 1", "two statements"),
    ]:
        try:
            at.check_expression(bad)
            check("refuses %s" % why, False, bad)
        except at.ToolError:
            check("refuses %s" % why, True)

    # ---- and allows what it must, including the trap ----------------------
    for good, why in [
        ("tickets.groupby('Request Category').size()", "a groupby"),
        ("tickets['Resolution Time (Days)'].mean()", "a mean"),
        ("len(agents)", "a builtin"),
        ("tickets[['Import Date']].head(2)", "a COLUMN whose name contains 'import'"),
        ("pd.merge(tickets, agents, how='cross').shape", "a pandas call"),
    ]:
        try:
            at.check_expression(good)
            check("allows %s" % why, True)
        except at.ToolError as exc:
            check("allows %s" % why, False, "%s -> %s" % (good, exc))

    # The trap above is the point: a substring rule rejecting "import" would
    # also reject the column, and a rule loose enough for the column would
    # admit the keyword. Parsing tells them apart; searching text cannot.
    out, err = at.dispatch("run_query", {"expression": "tickets[['Import Date']].shape"},
                           tickets, agents)
    check("and the 'Import Date' column really evaluates", err is False and "4" in out,
          "%r" % out)

    # ---- a bad expression is data for the agent, not a crash --------------
    out, err = at.dispatch("run_query", {"expression": "tickets['nope'].mean()"},
                           tickets, agents)
    check("a failing query returns an error to the agent instead of crashing",
          err is True and "KeyError" in out, "%r" % out)

    # ---- describe_corpus gives schema, never an aggregate -----------------
    described = at.describe_corpus(tickets, agents)
    check("describe_corpus names the columns", "Request Category" in described)
    check("and gives no aggregate the questions ask for",
          "mean" not in described.lower() and "4.0" not in described, described[:200])

    # ---- the output contract ----------------------------------------------
    check("an answer is extracted from the last ANSWER marker",
          ra.claimed_answer("ANSWER: first\nmore\nANSWER: last") == "last")
    check("no marker means no answer, not an empty one",
          ra.claimed_answer("I think it is 53.") is None)
    check("CANNOT COMPUTE survives as itself",
          ra.claimed_answer("ANSWER: CANNOT COMPUTE") == "CANNOT COMPUTE")

    # ---- a full question, with a tool call --------------------------------
    scratch = Path(tempfile.mkdtemp())
    client = FakeClient(tool_then("tickets['Resolution Time (Days)'].mean()", "4.0"))
    budget = ab.RunBudget(ra.MODEL, max_iterations=20, max_tokens=1_000_000)
    q = {"id": "O9", "kind": "objective", "question": "What is the average?"}
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        recs = ra.run([q], client, tickets, agents, budget, scratch / "r1")
    rec, printed = recs[0], console.getvalue()

    check("the tool call is recorded with its expression and result",
          rec["tool_calls"] == 1 and "Resolution Time" in rec["calls"][0]["input"]["expression"],
          str(rec["calls"]))
    check("the answer is captured", rec["answer"] == "4.0", str(rec["answer"]))
    check("the contract is recorded as followed", rec["followed_contract"] is True)
    check("wall-clock seconds are recorded", isinstance(rec["seconds"], float))
    # Tokens per question, not only per run: a total cannot say what a rung
    # costs or what a correct answer costs.
    check("input and output tokens are recorded per question",
          rec["input_tokens"] == 1800 and rec["output_tokens"] == 240,
          "%r / %r" % (rec["input_tokens"], rec["output_tokens"]))
    check("and tokens per exchange, which is what makes a rung expensive",
          rec["tokens_per_exchange"] == 1020.0, str(rec["tokens_per_exchange"]))
    check("each question is its own conversation, with no history carried in",
          client.calls[0]["messages"][0]["role"] == "user"
          and len(client.calls[0]["messages"]) == 1, str(len(client.calls[0]["messages"])))
    check("the tools are offered on every call",
          all("tools" in c for c in client.calls))

    # ---- IA-62: the console is not the answer key -------------------------
    check("the console names the question", "O9" in printed, printed)
    check("and says whether the contract was followed", "contract ok" in printed, printed)
    check("and never prints the answer", "4.0" not in printed, printed)
    check("nor the model's prose", "Working" not in printed, printed)

    # ---- but the run directory keeps everything ---------------------------
    on_disk = json.loads((scratch / "r1" / "answers.jsonl").read_text(encoding="utf-8"))
    check("the answer and every tool call are written to answers.jsonl in full",
          on_disk["answer"] == "4.0" and len(on_disk["calls"]) == 1, str(on_disk)[:200])

    # ---- an answer with no tool call is recorded as such ------------------
    client2 = FakeClient([text_then("53.4")])
    b2 = ab.RunBudget(ra.MODEL, max_iterations=20, max_tokens=1_000_000)
    with contextlib.redirect_stdout(io.StringIO()):
        rec2 = ra.run([q], client2, tickets, agents, b2, scratch / "r2")[0]
    check("a right-looking answer with no tool call is recorded with tool_calls 0, "
          "so the scorer can mark it unsupported",
          rec2["tool_calls"] == 0 and rec2["answer"] == "53.4", str(rec2)[:160])

    # ---- an answer that never follows the contract ------------------------
    client3 = FakeClient([Response([Block(type="text", text="It is about 53.")])])
    b3 = ab.RunBudget(ra.MODEL, max_iterations=20, max_tokens=1_000_000)
    with contextlib.redirect_stdout(io.StringIO()):
        rec3 = ra.run([q], client3, tickets, agents, b3, scratch / "r3")[0]
    check("no ANSWER line is recorded as not following the contract, and the "
          "answer stays None rather than becoming an empty string",
          rec3["followed_contract"] is False and rec3["answer"] is None, str(rec3)[:160])

    # ---- abstention is its own outcome ------------------------------------
    client4 = FakeClient([text_then("CANNOT COMPUTE")])
    b4 = ab.RunBudget(ra.MODEL, max_iterations=20, max_tokens=1_000_000)
    with contextlib.redirect_stdout(io.StringIO()):
        rec4 = ra.run([q], client4, tickets, agents, b4, scratch / "r4")[0]
    check("an abstention is flagged and still counts as following the contract",
          rec4["abstained"] is True and rec4["followed_contract"] is True)

    # ---- truncation is not the same thing as breaking the contract --------
    # The first full run reported eleven "contract NOT FOLLOWED" and every one
    # was this harness cutting the agent off mid-work. Two situations reported
    # as one, in the field that decides whether the agent complied.
    talker = FakeClient([Response(
        [Block(type="text", text="still working"),
         Block(type="tool_use", id="t%d" % i, name="run_query",
               input={"expression": "len(tickets)"})], "tool_use")
        for i in range(ra.MAX_EXCHANGES + 2)])
    b5 = ab.RunBudget(ra.MODEL, max_iterations=200, max_tokens=5_000_000)
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        cut = ra.run([q], talker, tickets, agents, b5, scratch / "cut")[0]
    check("an agent stopped at the exchange cap is marked truncated",
          cut["truncated"] is True, str(cut)[:160])
    check("and NOT no_contract — that field is reserved for the agent finishing "
          "without an ANSWER line",
          cut["no_contract"] is False, str(cut)[:160])
    check("the console says the harness truncated it, not that the agent failed",
          "TRUNCATED BY THE HARNESS" in console.getvalue(), console.getvalue())

    quiet = FakeClient([Response([Block(type="text", text="It is about 53.")])])
    b6 = ab.RunBudget(ra.MODEL, max_iterations=20, max_tokens=1_000_000)
    with contextlib.redirect_stdout(io.StringIO()):
        ended = ra.run([q], quiet, tickets, agents, b6, scratch / "ended")[0]
    check("an agent that finishes with no ANSWER line IS no_contract",
          ended["no_contract"] is True and ended["truncated"] is False, str(ended)[:160])

    # ---- the leak check on prompts ----------------------------------------
    ref = scratch / "reference"
    ref.mkdir()
    (ref / "baseline.json").write_text(json.dumps({"comparison": [
        {"question": "O3", "figure": "mean", "human": 53.36507936507937,
         "computed": 53.36507936507937}]}), encoding="utf-8")
    ra.assert_no_leak(["What is the average daily ticket volume over time?"], ref)
    check("a clean prompt set passes the leak check", True)
    try:
        ra.assert_no_leak(["The mean is 53.36507936507937. Confirm it."], ref)
        check("a prompt carrying a baseline value is refused", False, "it passed")
    except ra.RunError as exc:
        check("a prompt carrying a baseline value is refused",
              "53.36507936507937" in str(exc), str(exc))

    # ---- the run must be able to name the rules it ran under --------------
    try:
        ra.preregistration_commit(scratch)
        check("a repo with no committed pre-registration is refused", False,
              "it returned a commit")
    except ra.RunError:
        check("a repo with no committed pre-registration is refused", True)

    # ---- budget and reconciliation ----------------------------------------
    check("the budget counted the calls it made", b4.report()["calls"] == 1,
          str(b4.report()["calls"]))
    check("a clean run reconciles", ra.reconcile([rec4], b4) == [])
    rec4_bad = dict(rec4, exchanges=99)
    check("and a mismatch between budget and transcripts is reported",
          ra.reconcile([rec4_bad], b4) != [])

    # ---- the instrument names itself, not only its rules ------------------
    harness = ra.head_commit(scratch)
    check("head_commit reports rather than raising outside a repository",
          isinstance(harness, dict) and "commit" in harness, str(harness))

    # The distinction that the first version could not make: IDE metadata is
    # not an unpinned instrument, and a new module is.
    import subprocess as sp
    repo = Path(tempfile.mkdtemp())
    (repo / "analyst").mkdir()
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"]):
        sp.run(cmd, cwd=str(repo), capture_output=True)
    (repo / "analyst" / "run_agent.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    sp.run(["git", "commit", "-qm", "init"], cwd=str(repo), capture_output=True)
    check("a clean analyst tree is reported pinned",
          ra.head_commit(repo)["analyst_tree_dirty"] is False,
          str(ra.head_commit(repo)))

    (repo / "analyst" / ".idea").mkdir()
    (repo / "analyst" / ".idea" / "misc.xml").write_text("<x/>", encoding="utf-8")
    ide = ra.head_commit(repo)
    check("untracked IDE metadata does NOT make the instrument unpinned",
          ide["analyst_tree_dirty"] is False, str(ide))
    check("but it is still reported rather than hidden",
          any(".idea" in p for p in ide["untracked_other"]), str(ide))

    (repo / "analyst" / "helper.py").write_text("y = 2\n", encoding="utf-8")
    newmod = ra.head_commit(repo)
    check("an untracked .py DOES make the instrument unpinned — the run might "
          "be importing it",
          newmod["analyst_tree_dirty"] is True, str(newmod))

    (repo / "analyst" / "run_agent.py").write_text("x = 2\n", encoding="utf-8")
    edited = ra.head_commit(repo)
    check("and a modified tracked source file does too",
          any("run_agent.py" in p for p in edited["tracked_modified"]), str(edited))
    cfg = ra.run_config([q], scratch, "deadbeef", {"usd": 5.0},
                        {"commit": "cafe1234", "analyst_tree_dirty": True,
                         "uncommitted": [" M analyst/run_agent.py"]})
    check("run_config records the harness commit, so answers can be traced to "
          "the code that produced them",
          cfg["harness"]["commit"] == "cafe1234", str(cfg.get("harness")))
    check("and records a dirty tree rather than silently allowing it",
          cfg["harness"]["analyst_tree_dirty"] is True, str(cfg.get("harness")))
    check("run_config records the pre-registration commit",
          cfg["preregistration_commit"] == "deadbeef")
    check("and the tool list, so the architecture rung is on the record",
          cfg["tools"] == ["describe_corpus", "run_query"], str(cfg["tools"]))
    check("and states that no USD figure from this run is publishable",
          "publish" in cfg["cost_cap_note"], cfg["cost_cap_note"])

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print("%-4s  %-*s  %s" % (status, width, name, detail))
        failed += status == FAIL
    print("\n%d/%d checks passed" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
