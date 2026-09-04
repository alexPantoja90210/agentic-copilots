"""
Selftest for the blind scorer.

Two things decide whether this pilot can report anything, and both are tested
here rather than trusted:

  - the rater is shown the answer and nothing that names the arm;
  - the tool refuses to state a verdict on a corpus smaller than the one the
    pre-registration was written against.

The second is the one worth having. A scorer that will happily produce
"H1 supported" from six incidents is a scorer that will eventually be asked to.
"""

from __future__ import annotations

import score_arms as sa

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def answer(iid, arm, claimed, tokens=(1000, 60)):
    return {"incident_id": iid, "arm": arm, "claimed_cause": claimed,
            "answer": "reasoning here.\nROOT CAUSE: %s" % (claimed or "?"),
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}


def truth(iid, fault, role):
    return {"record": "open", "incident_id": iid, "fault": fault, "node_role": role}


def corpus(n_incidents, arm_b_correct):
    """n incidents, arm A always wrong, arm B correct in `arm_b_correct` of them."""
    answers, labels = [], []
    for index in range(n_incidents):
        iid = "INC-%02d" % index
        labels.append(truth(iid, "F3", "db"))
        answers.append(answer(iid, "arm_a", "web"))
        answers.append(answer(iid, "arm_b",
                              "db" if index < arm_b_correct else "web"))
    return answers, labels


def run() -> int:
    # ---- the outcome is not a boolean, and each category means one thing ----
    check("naming the failed service is correct",
          sa.outcome("db", "F3", "db") == "correct")
    check("case and spacing do not decide correctness",
          sa.outcome("  DB ", "F3", "db") == "correct")
    check("naming a different service is a confident failure",
          sa.outcome("web", "F3", "db") == "wrong_node")
    check("INSUFFICIENT on a real fault is recorded apart from a wrong name",
          sa.outcome("INSUFFICIENT", "F3", "db") == "abstained",
          "an agent that admits it cannot tell has not failed the same way")
    check("INSUFFICIENT is correct for the control",
          sa.outcome("INSUFFICIENT", "F0", "web") == "correct")
    check("naming any service during the control is a false positive",
          sa.outcome("db", "F0", "web") == "false_positive",
          "the control exists to punish exactly this")
    check("an answer with no contract line is its own category",
          sa.outcome(None, "F3", "db") == "no_contract")

    # ---- the blind pass shows nothing that names the arm ----
    answers, labels = corpus(3, 2)
    blind = sa.blind_order(answers, seed=7)
    check("every answer reaches the blind pass", len(blind) == len(answers))
    check("the order is not the order they were produced in",
          [b["token"] for b in blind] != ["R%03d" % i for i in range(len(answers))],
          "an unshuffled blind pass is not blind")
    check("the same seed reproduces the same order",
          [b["token"] for b in sa.blind_order(answers, seed=7)]
          == [b["token"] for b in blind],
          "a shuffle nobody can reproduce cannot be audited")

    shown = []
    sa.rate(blind, ask=lambda _prompt: "4", show=shown.append)
    screen = "\n".join(str(line) for line in shown)
    check("the rater is never shown the arm", "arm_a" not in screen and "arm_b" not in screen,
          screen[:200])
    check("nor the incident id", "INC-" not in screen, screen[:200])
    check("nor the prompt, whose length alone would name the arm",
          "ROOT CAUSE" in screen and "Service dependency graph" not in screen)

    ratings = sa.rate(blind, ask=lambda _p: "4", show=lambda *_a: None)
    check("a rating is captured per answer", len(ratings) == len(answers))
    check("and keyed by the token, not by position",
          {r["token"] for r in ratings} == {b["token"] for b in blind})

    # ---- the join is where the label finally meets the answer ----
    rows = sa.join(answers, ratings, labels)
    check("every answer is scored", len(rows) == len(answers))
    result = sa.wins(rows)
    check("arm B's wins are counted per incident, not per answer",
          len(result["arm_b"]) == 2 and result["incidents"] == 3,
          str(result))

    try:
        sa.join([answer("GHOST", "arm_a", "db")], [], labels)
        check("an answer with no ground-truth record is refused", False,
              "it would have been scored against a label that does not exist")
    except sa.ScoreError as exc:
        check("an answer with no ground-truth record is refused",
              "cannot be scored" in str(exc))

    # ---- THE check: no verdict below the pre-registered corpus size --------
    answers, labels = corpus(6, 6)
    rows = sa.join(answers, [], labels)
    text = sa.verdict(sa.wins(rows))
    check("six incidents with arm B winning all of them yields NO verdict",
          text.startswith("VERDICT: none"),
          "a scorer that will produce 'H1 supported' from six will be asked to")
    check("and it says what the run is instead", "pipeline validation" in text)
    check("and forbids quoting the counts as evidence", "must not be" in text)

    answers, labels = corpus(12, 4)
    text = sa.verdict(sa.wins(sa.join(answers, [], labels)))
    check("at the pre-registered size, meeting the threshold supports H1",
          text.startswith("VERDICT: H1 supported"), text)

    answers, labels = corpus(12, 3)
    text = sa.verdict(sa.wins(sa.join(answers, [], labels)))
    check("one win short and H1 is NOT supported",
          text.startswith("VERDICT: H1 NOT supported"), text)
    check("and the negative result is framed as publishable",
            "published as readily" in text)

    # ---- H2 is measured in tokens, and needs no price file -----------------
    answers, labels = corpus(2, 1)
    for row in answers:
        if row["arm"] == "arm_b":
            row["usage"] = {"input_tokens": 4000, "output_tokens": 60}
    rendered = sa.render(sa.join(answers, [], labels))
    check("the cost table reports the input-token ratio",
          "arm B costs 4.0x the input tokens of arm A." in rendered, rendered)
    check("and no dollar figure appears anywhere",
          "$" not in rendered, "model-pricing.json is still unverified (D11)")

    # ---- accuracy is never pooled across fault classes ---------------------
    mixed_answers, mixed_labels = [], []
    for iid, fault, role in [("A", "F3", "db"), ("B", "F1", "web"), ("C", "F0", "web")]:
        mixed_labels.append(truth(iid, fault, role))
        mixed_answers.append(answer(iid, "arm_a", "web"))
        mixed_answers.append(answer(iid, "arm_b", "db"))
    table = sa.per_class(sa.join(mixed_answers, [], mixed_labels))
    check("the table is broken out by fault class",
          set(table) == {"F3", "F1", "F0"}, str(sorted(table)))
    check("and the control's false positive lands in the control's row",
          table["F0"]["arm_b"]["false_positive"] == 1, str(table["F0"]))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
