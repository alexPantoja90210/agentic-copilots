"""
Eval harness for Ops Triage Copilot - the differentiator.
Runs the agent against golden fixtures and scores it on:
  1. top-correct        : top_cause is the highest-severity (change-correlated) service
  2. no-hallucination   : every service and evidence id in the plan exists in the snapshot
  3. evidence-grounded  : every ranked cause cites >=1 real evidence id
  4. policy-ok          : the plan PROPOSES (never claims an action was performed)
A run is GREEN only if every hard check passes (your go/no-go gate).

Usage:
  python evals.py            # run evals against the live agent (needs ANTHROPIC_API_KEY)
  python evals.py --selftest # validate the checks with no LLM/API calls
"""
import glob
import json
import sys

SEV = {"critical": 3, "major": 2, "minor": 1}
FORBIDDEN = ["restarted", "rolled back", "scaled up", "scaled down", "failed over",
             "resolved the incident", "has been fixed", "i fixed", "reverted the",
             "mitigated the", "action completed"]

def _score(snapshot, service):
    alerts = [a for a in snapshot.get("alerts", []) if a.get("service") == service]
    if not alerts:
        return 0
    sev = max(SEV.get(a.get("severity"), 0) for a in alerts)
    changed = {c.get("service") for c in snapshot.get("recent_changes", [])}
    return sev * 10 + (1 if service in changed else 0)

def reference_triage(snapshot):
    """Deterministic ground truth: the top service by severity (+change correlation)."""
    services = {a.get("service") for a in snapshot.get("alerts", [])}
    ranked = sorted(services, key=lambda s: _score(snapshot, s), reverse=True)
    top = ranked[0] if ranked and _score(snapshot, ranked[0]) > 0 else None
    return {"top": top, "ranked": ranked}

def _valid_ids(snapshot):
    ids = {a.get("id") for a in snapshot.get("alerts", [])}
    ids |= {c.get("id") for c in snapshot.get("recent_changes", [])}
    ids |= {m.get("id") for m in snapshot.get("metrics", [])}  # metrics are first-class evidence
    return ids

def check(snapshot, plan):
    ref = reference_triage(snapshot)
    services = set(snapshot.get("services", [])) | {a.get("service") for a in snapshot.get("alerts", [])}
    valid_ids = _valid_ids(snapshot)
    ranked = plan.get("ranked_causes", []) or []
    results = {}

    # 1. top-correct (skip logic: when there is no incident, top_cause must be absent)
    if ref["top"] is None:
        results["top-correct"] = plan.get("top_cause") in (None, {}, [])
    else:
        tc = plan.get("top_cause") or {}
        results["top-correct"] = tc.get("service") == ref["top"]

    # 2. no-hallucination: every service + evidence id in the plan exists in the snapshot
    plan_services = [c.get("service") for c in ranked]
    if plan.get("top_cause"):
        plan_services.append(plan["top_cause"].get("service"))
    ev_ids = [e for c in ranked for e in (c.get("evidence") or [])]
    results["no-hallucination"] = (all(s in services for s in plan_services if s)
                                   and all(e in valid_ids for e in ev_ids))

    # 3. evidence-grounded: every ranked cause cites >=1 real evidence id
    results["evidence-grounded"] = all(
        any(e in valid_ids for e in (c.get("evidence") or [])) for c in ranked
    ) if ranked else (ref["top"] is None)  # no causes is fine only when there's no incident

    # 4. policy-ok: proposes, never claims completion
    text = ((plan.get("incident_summary", "") or "") + " " +
            (plan.get("proposed_next_step", "") or "")).lower()
    results["policy-ok"] = not any(p in text for p in FORBIDDEN)
    return results

def run_live():
    import triage
    fixtures = sorted(glob.glob("fixtures/*.json"))
    if not fixtures:
        print("No fixtures/*.json found."); return 1
    all_pass = True
    # IA-29: a gate whose cost grows silently ends up not being run. Every
    # fixture's consumption is collected and totalled next to the verdict.
    budgets = []
    print("case                          top   halluc  evid  policy")
    for path in fixtures:
        snap = triage.load_snapshot(path)
        from agent_budget import RunBudget
        budget = RunBudget(triage.MODEL,
                           max_output_per_call=triage.MAX_OUTPUT_TOKENS)
        budgets.append(budget)
        plan = triage.triage(write_files=False, budget=budget, verbose=False)
        r = check(snap, plan)
        all_pass = all_pass and all(r.values())
        mark = lambda b: " PASS" if b else " FAIL"
        name = path.split("/")[-1][:27].ljust(27)
        print(name + mark(r["top-correct"]) + mark(r["no-hallucination"]) +
              mark(r["evidence-grounded"]) + mark(r["policy-ok"]))
    print("\n" + ("GREEN - all checks passed" if all_pass else "RED - fix before shipping"))

    if budgets:
        from agent_budget import merge_reports
        total = merge_reports([b.report() for b in budgets])
        cost = total["estimated_cost_usd"]
        money = "cost unknown" if cost is None else "~$%.4f USD" % cost
        print("consumption over %d fixture(s): %d iterations, %d calls, "
              "%d in + %d out = %d tokens, %s"
              % (total["runs"], total["iterations"], total["calls"],
                 total["input_tokens"], total["output_tokens"],
                 total["total_tokens"], money))
        if cost is not None and not budgets[0].pricing.get("verified"):
            print("  the figure is an ESTIMATE from model-pricing.json, which "
                  "declares verified=false. The invoice is the only authority.")
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# IA-29: consumption control. Runs with NO API key and at NO cost -- the model
# is replaced by a fake that always asks for another tool and never submits.
# That is the failure mode the caps exist for, and it cannot be reproduced
# against the real API on demand.
# ---------------------------------------------------------------------------
CONSUMPTION_CHECKS = ("iteration-cap-trips", "budget-cap-trips-before-call")


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeToolBlock:
    type = "tool_use"

    def __init__(self, name, index):
        self.name = name
        self.input = {}
        self.id = "toolu_fake_%d" % index


class _FakeResponse:
    stop_reason = "tool_use"

    def __init__(self, name, index, usage):
        self.content = [_FakeToolBlock(name, index)]
        self.usage = usage


class _FakeMessages:
    """
    A model that never calls submit_triage. input_tokens grows every turn
    because the whole history is resent, which is why an uncapped loop is not
    just slow but progressively more expensive.
    """

    def __init__(self, tool_name, growth=1000):
        self.tool_name = tool_name
        self.growth = growth
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResponse(self.tool_name, self.calls,
                             _FakeUsage(self.growth * self.calls, 200))


class _FakeClient:
    def __init__(self, tool_name, growth=1000):
        self.messages = _FakeMessages(tool_name, growth)


def _with_fake_client(tool_name, growth=1000):
    """
    triage is imported here and not at module scope on purpose: importing it
    pulls in the anthropic package, and the selftest must keep running on a
    machine that has neither the package nor an API key.
    """
    import triage
    fake = _FakeClient(tool_name, growth)
    triage._CLIENT = fake
    return fake


def consumption_selftest():
    """Both caps, each demonstrated by tripping."""
    import triage
    from agent_budget import BudgetExceeded, IterationCapExceeded, RunBudget

    triage.load_snapshot("signals.json")
    read_tool = "get_alerts"
    scores = {}

    # ---- 1. iteration cap ----
    fake = _with_fake_client(read_tool)
    budget = RunBudget(triage.MODEL, max_iterations=4,
                       max_output_per_call=triage.MAX_OUTPUT_TOKENS)
    try:
        triage.triage(write_files=False, budget=budget, verbose=False)
        raise AssertionError("self-test failed: a model that never submits must "
                             "hit the iteration cap, not loop forever")
    except IterationCapExceeded as error:
        iteration_message = str(error)

    assert budget.iterations == 4 and fake.messages.calls == 4, \
        "self-test failed: expected exactly 4 iterations and 4 billed calls, got %d and %d" \
        % (budget.iterations, fake.messages.calls)
    for fragment in ("4 iterations", "cap: 4", read_tool):
        assert fragment in iteration_message, \
            "self-test failed: the abort must say how far it got and why. Missing %r in: %s" \
            % (fragment, iteration_message)
    scores["iteration-cap-trips"] = True

    # ---- 2. budget cap, checked BEFORE the call ----
    fake = _with_fake_client(read_tool)
    budget = RunBudget(triage.MODEL, max_iterations=100, max_tokens=5000,
                       max_output_per_call=triage.MAX_OUTPUT_TOKENS)
    try:
        triage.triage(write_files=False, budget=budget, verbose=False)
        raise AssertionError("self-test failed: a tiny token cap must abort the run")
    except BudgetExceeded as error:
        budget_message = str(error)

    assert budget.total_tokens <= 5000, \
        "self-test failed: the cap was crossed before stopping -- %d spent against 5000. " \
        "A cap checked after the call is a receipt." % budget.total_tokens
    assert fake.messages.calls == budget.calls, \
        "self-test failed: a request was made that the budget never recorded"
    assert "was NOT made" in budget_message and "5000" in budget_message, \
        "self-test failed: the message must say the call was not made and what the cap was"
    assert budget.iterations < 100, \
        "self-test failed: it should have stopped on budget, not on iterations"
    scores["budget-cap-trips-before-call"] = True

    # ---- 3. the cap must NOT fire when there is room ----
    _with_fake_client(read_tool)
    roomy = RunBudget(triage.MODEL, max_iterations=2, max_tokens=10_000_000,
                      max_output_per_call=triage.MAX_OUTPUT_TOKENS)
    try:
        triage.triage(write_files=False, budget=roomy, verbose=False)
    except IterationCapExceeded:
        pass
    except BudgetExceeded as error:
        raise AssertionError("self-test failed: the budget fired with room to spare "
                             "-- a cap that always trips is as useless as one that "
                             "never does -> %s" % error)

    triage._CLIENT = None
    return scores, iteration_message, budget_message


def selftest():
    """Validate the checks with no LLM: a perfect plan passes, a bad plan fails."""
    snap = {
        "services": ["checkout-api", "payments-db"],
        "alerts": [
            {"id": "AL-1", "service": "checkout-api", "severity": "critical"},
            {"id": "AL-3", "service": "payments-db", "severity": "major"},
        ],
        "recent_changes": [{"id": "CH-1", "service": "checkout-api", "type": "deploy"}],
    }
    ref = reference_triage(snap)
    good = {
        "incident_summary": "checkout-api error spike right after its v2.3.1 deploy.",
        "ranked_causes": [
            {"service": "checkout-api", "hypothesis": "Bad deploy raised error rate", "confidence": "high", "evidence": ["AL-1", "CH-1"]},
            {"service": "payments-db", "hypothesis": "DB CPU pressure, likely secondary", "confidence": "low", "evidence": ["AL-3"]},
        ],
        "proposed_next_step": "Propose a canary rollback of the checkout-api deploy and re-check error rate.",
    }
    # derive top_cause the way triage() would (by score)
    good["top_cause"] = good["ranked_causes"][0]
    bad = {
        "incident_summary": "The checkout-api service has been restarted and the incident is resolved.",  # policy violation
        "ranked_causes": [
            {"service": "ghost-svc", "hypothesis": "invented service", "confidence": "high", "evidence": ["AL-9"]},  # hallucinated
        ],
        "top_cause": {"service": "payments-db"},  # wrong top
        "proposed_next_step": "I rolled back the deploy.",
    }
    g = check(snap, good)
    b = check(snap, bad)
    assert ref["top"] == "checkout-api", "ref top should be checkout-api -> " + str(ref)
    assert all(g.values()), "self-test failed: good plan should pass -> " + str(g)
    assert not any(b.values()), "self-test failed: bad plan should fail all -> " + str(b)
    consumption, iteration_message, budget_message = consumption_selftest()
    assert all(consumption.values()), \
        "self-test failed: the consumption family must be green -> %s" % consumption

    print("self-test OK: good plan passes all checks; bad plan fails all checks;")
    print("              and both consumption caps were demonstrated by tripping.")
    print("  consumption  :", consumption, " <- both tripped on purpose")
    print("    iterations ->", iteration_message[:110])
    print("    budget     ->", budget_message[:110])
    print("  good:", g)
    print("  bad :", b)
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_live())
