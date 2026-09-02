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
import datetime
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



# ---------------------------------------------------------------------------
# IA-4: the contract and the collector. Runs with NO AWS credentials, NO network
# and NO cost -- CloudWatch is replaced by a client that returns recorded shapes.
#
# The instance ids below are invented. Real ones belong to an account and this
# repository is public, which is also why the collector writes signals.live.json
# and not signals.json.
# ---------------------------------------------------------------------------
CONTRACT_CHECKS = ("contract-ok",)
COLLECTOR_CHECKS = ("collected-valid", "every-metric-accounted", "no-invented-alert")

FAKE_IDS = ("i-0aaa111bbb222ccc3", "i-0ddd444eee555fff6")


class _FakePaginator:
    def __init__(self, instance_ids):
        self.instance_ids = instance_ids

    def paginate(self, **kwargs):
        yield {"Metrics": [{"Dimensions": [{"Name": "InstanceId", "Value": i}]}
                           for i in self.instance_ids]}


class _FakeCloudWatch:
    """
    A recorded CloudWatch. `missing` is the interesting part: a real account has
    metrics that exist in the catalogue and have no datapoints in the window,
    which is exactly the case the collector must record instead of dropping.
    """

    def __init__(self, instance_ids, values, missing=(), empty_baseline=False):
        self.instance_ids = list(instance_ids)
        self.values = dict(values)
        self.missing = set(missing)
        self.empty_baseline = empty_baseline
        self.calls = 0

    def get_paginator(self, _name):
        return _FakePaginator(self.instance_ids)

    def get_metric_statistics(self, **kwargs):
        self.calls += 1
        if kwargs["MetricName"] in self.missing:
            return {"Datapoints": []}
        if self.empty_baseline and kwargs["Period"] >= 3600:
            return {"Datapoints": []}
        # Real datapoints always carry a Timestamp, and IA-51 made the collector
        # read it. A fake that omits a field the API always sends is a lie the
        # tests would have to work around, so it sends one too: a flat series,
        # which is the honest reading of "this account sat at one value".
        value = self.values.get(kwargs["MetricName"], 1.0)
        base = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc)
        return {"Datapoints": [
            {"Average": value, "Timestamp": base + datetime.timedelta(minutes=5 * i)}
            for i in range(4)
        ]}


class _ReplayCloudWatch:
    """
    CloudWatch as this account actually answered, replayed from a capture.

    `_FakeCloudWatch` returns whatever the test author decided the account would
    say, which is why it never caught IA-51 or IA-53: the fixtures encoded the
    same assumptions as the code they were testing. This one replays responses
    recorded from the live account by `capture_fixture.py`, so the test's idea
    of reality comes from reality.

    Window and baseline calls are told apart by the SPAN they ask for -- hours
    against days -- rather than by the period, which is derived and could drift.
    """

    def __init__(self, path):
        self.data = _read_json(path)
        self.targets = list(self.data["targets"])
        self.window_seconds = self.data["window_hours"] * 3600
        self.responses = self.data["responses"]
        self.calls = 0

    def get_paginator(self, _name):
        return _FakePaginator(self.targets)

    def get_metric_statistics(self, **kwargs):
        self.calls += 1
        span = (kwargs["EndTime"] - kwargs["StartTime"]).total_seconds()
        kind = "window" if span <= self.window_seconds * 1.5 else "baseline"
        instance = kwargs["Dimensions"][0]["Value"]
        key = "%s|%s|%s" % (instance, kwargs["MetricName"], kind)
        recorded = self.responses.get(key, {"Datapoints": []})
        # Timestamps were serialised as ISO strings when captured; the collector
        # does arithmetic on them, so they go back as datetimes here.
        return {"Datapoints": [
            {"Average": point["Average"],
             "Timestamp": datetime.datetime.fromisoformat(point["Timestamp"])}
            for point in recorded["Datapoints"]
        ]}


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def contract_selftest():
    """The contract, in both directions."""
    from signals_contract import validate_signals

    good = _read_json("signals.json")
    assert not validate_signals(good), \
        "self-test failed: the shipped signals.json must satisfy its own contract -> %s" \
        % validate_signals(good)

    # The check this contract exists for: an alert about a service nobody declared.
    ghost = _read_json("fixtures/broken_service_ref.json")
    problems = validate_signals(ghost)
    assert problems, "self-test failed: an alert on an undeclared service must be RED"
    assert any("ghost-svc" in problem for problem in problems), \
        "self-test failed: the violation must name the offending service -> %s" % problems

    # And the consumer must refuse it, not merely score it.
    import triage
    try:
        triage.load_snapshot("fixtures/broken_service_ref.json")
        raise AssertionError("self-test failed: load_snapshot accepted a snapshot "
                             "that violates the contract")
    except triage.SignalsContractError:
        pass

    return {"contract-ok": True}, problems[0]


def collector_selftest():
    """The collector, driven by recorded responses."""
    import collector
    from signals_contract import validate_signals

    scores = {}

    # ---- 1. a busy account: metrics emitted, one threshold crossed ----
    busy = _FakeCloudWatch(FAKE_IDS,
                           {"CPUUtilization": 93.5, "StatusCheckFailed": 0.0,
                            "CPUCreditBalance": 120.0},
                           missing=("EBSWriteOps",))
    snapshot = collector.collect(client=busy, region="us-east-1")
    problems = validate_signals(snapshot)
    scores["collected-valid"] = not problems
    assert not problems, \
        "self-test failed: the collected snapshot must satisfy the contract -> %s" % problems

    # The invariant borrowed from IA-38: nothing disappears quietly.
    emitted = {(m["service"], m["name"]) for m in snapshot["metrics"]}
    skipped = {(s["target"], s["metric"]) for s in snapshot["source"]["skipped_metrics"]}
    considered = {(i, n) for i in FAKE_IDS
                  for n in (list(collector.THRESHOLDS) + list(collector.TREND_METRICS)
                            + list(collector.CONTEXT_METRICS))}
    scores["every-metric-accounted"] = (emitted | skipped) == considered and not (emitted & skipped)
    assert scores["every-metric-accounted"], \
        "self-test failed: a metric was neither emitted nor recorded as skipped"

    assert len(snapshot["alerts"]) == 2, \
        "self-test failed: CPU at 93.5 over a threshold of 80 must alert on both targets"

    # ---- 2. a quiet account: NO alert may be invented ----
    quiet = _FakeCloudWatch(FAKE_IDS, {"CPUUtilization": 4.0, "StatusCheckFailed": 0.0,
                                       "CPUCreditBalance": 200.0})
    calm = collector.collect(client=quiet, region="us-east-1")
    scores["no-invented-alert"] = calm["alerts"] == []
    assert scores["no-invented-alert"], \
        "self-test failed: no declared threshold was crossed, so alerts must be empty -> %s" \
        % calm["alerts"]
    assert not validate_signals(calm), "self-test failed: the quiet snapshot must still be valid"

    # ---- 3. NEGATIVE: an account with targets but no usable data ----
    # This is the defect the first live run exposed. An empty snapshot passes the
    # contract and reads as a healthy quiet system. It must fail loudly instead.
    barren = _FakeCloudWatch(FAKE_IDS, {},
                             missing=(tuple(collector.THRESHOLDS)
                                      + tuple(collector.TREND_METRICS)
                                      + collector.CONTEXT_METRICS))
    try:
        collector.collect(client=barren, region="us-east-1")
        raise AssertionError("self-test failed: targets with no usable datapoints must "
                             "abort, not produce an empty snapshot that looks healthy")
    except collector.CollectorError as error:
        barren_message = str(error)
    assert "not one metric had usable datapoints" in barren_message

    # ---- 4. NEGATIVE: no targets at all ----
    try:
        collector.collect(client=_FakeCloudWatch((), {}), region="us-east-1")
        raise AssertionError("self-test failed: an account with no metrics must abort")
    except collector.CollectorError:
        pass

    # ---- 5. the period is computed, not assumed ----
    # A 290-hour window at 300s would ask for 3480 datapoints against a cap of
    # 1440. The second defect the live run exposed.
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    for hours in (3, 24, 290, 400):
        start = now - datetime.timedelta(hours=hours)
        period = collector.choose_period(start, now, now)
        assert (hours * 3600) / period <= collector.MAX_DATAPOINTS, \
            "self-test failed: a %dh window at %ds exceeds the datapoint cap" % (hours, period)

    # ---- 6. IA-53: a young instance, replayed from the real account ----
    # Recorded 1 Sep 2026 from an instance a few hours old: values present,
    # every baseline window empty. Before the fix this input produced the
    # abort in check 3 -- the collector reporting that it had found nothing on
    # an account that was emitting normally, with an injected outage sitting
    # unread in the data.
    young = _ReplayCloudWatch("fixtures/young_instance_no_baseline.json")
    snap = collector.collect(client=young, region="us-east-1")

    problems = validate_signals(snap)
    scores["young-instance-valid"] = not problems
    assert not problems, \
        "self-test failed: a snapshot with null baselines must satisfy the contract -> %s" % problems

    assert snap["metrics"], \
        "self-test failed: the young-instance fixture must produce metrics, not an abort"

    without_baseline = [m for m in snap["metrics"] if m["baseline"] is None]
    scores["baseline-absence-is-stated"] = len(without_baseline) == len(snap["metrics"])
    assert scores["baseline-absence-is-stated"], \
        "self-test failed: every metric in this fixture lacks history, so every " \
        "baseline must be an explicit null -> %s" % [m["baseline"] for m in snap["metrics"]]

    # Nothing disappears quietly: the absence is recorded, not merely implied.
    reasons = {e["reason"] for e in snap["source"]["metrics_without_baseline"]}
    assert any("without a baseline" in reason for reason in reasons), \
        "self-test failed: a missing baseline must be recorded"

    # The IA-38 invariant, extended. "Emitted with no baseline" and "emitted but
    # not judged" are states OF AN EMITTED METRIC. Recording them as skipped
    # would say the metric was never collected, which is false, and would break
    # the exclusivity the invariant rests on.
    emitted_here = {(m["service"], m["name"]) for m in snap["metrics"]}
    skipped_here = {(e["target"], e["metric"]) for e in snap["source"]["skipped_metrics"]}
    for field in ("metrics_without_baseline", "metrics_not_judged"):
        annotated = {(e["target"], e["metric"]) for e in snap["source"][field]}
        assert annotated <= emitted_here, \
            "self-test failed: %s must only describe metrics that were emitted" % field
        assert not (annotated & skipped_here), \
            "self-test failed: a metric cannot be both skipped and %s" % field

    # The value is still judged -- and IA-51 changed the verdict.
    #
    # This assertion used to demand the OPPOSITE. Under the old rule the young
    # instance's balance of 5.6 was below the constant 30, so a healthy machine
    # that had simply not accrued yet was reported MAJOR. The check was written
    # to be flipped by this fix rather than to be quietly stepped over, and this
    # is the flip.
    credit_alerts = [a for a in snap["alerts"] if a["signal"] == "cpu_credit_balance"]
    scores["young-instance-not-alerted"] = not credit_alerts
    assert not credit_alerts, \
        "self-test failed: an instance whose credit balance is RISING is accruing, " \
        "not depleting, and must not alert -> %s" % credit_alerts

    # ---- 8. NEGATIVE: a balance that really is draining must alert ----
    # Built by reversing the recorded series in time. The values, the spacing
    # and the magnitudes are the account's own; only the direction changes. That
    # matters: an invented depletion curve would be a curve I chose, and the
    # last three defects all came from fixtures their author had chosen.
    #
    # Stated plainly: this account has never actually depleted its credits, so
    # the falling case is derived from real data rather than observed. That is a
    # weaker claim than the rising case and the write-up says so.
    draining = _ReplayCloudWatch("fixtures/young_instance_no_baseline.json")
    for key, recorded in list(draining.responses.items()):
        if "CPUCreditBalance" in key and recorded["Datapoints"]:
            values = [p["Average"] for p in recorded["Datapoints"]]
            stamps = [p["Timestamp"] for p in recorded["Datapoints"]]
            recorded["Datapoints"] = [
                {"Average": v, "Timestamp": t}
                for v, t in zip(reversed(values), stamps)
            ]
    drained = collector.collect(client=draining, region="us-east-1")
    falling = [a for a in drained["alerts"] if a["signal"] == "cpu_credit_balance"]
    scores["draining-balance-alerts"] = bool(falling)
    assert falling, \
        "self-test failed: a falling credit balance must alert. If this passes " \
        "while check 6 also passes, the rule is judging direction, which is the " \
        "whole point of IA-51."
    assert falling[0]["value"] < 0, \
        "self-test failed: a depletion alert must report a NEGATIVE rate, since " \
        "the rate is what was judged -> %s" % falling[0]

    # ---- 9. NEGATIVE: exhausted and flat must not hide behind the trend ----
    # A balance pinned at zero is not falling. Under a trend-only rule it would
    # stay silent while the instance is throttled.
    import datetime as _dt
    flat_zero = [{"Average": 0.0,
                  "Timestamp": _dt.datetime(2026, 9, 1, tzinfo=_dt.timezone.utc)
                               + _dt.timedelta(minutes=5 * i)}
                 for i in range(6)]
    fires, slope, reason = collector.credit_verdict(flat_zero)
    scores["exhausted-flat-alerts"] = fires
    assert fires and "floor" in reason, \
        "self-test failed: a balance flat at zero is exhausted and must alert -> %s" % reason

    # ---- 10. too short to judge: say so, do not guess ----
    two_points = flat_zero[:2]
    fires, slope, reason = collector.credit_verdict(two_points)
    assert not fires and slope is None and "no verdict is claimed" in reason, \
        "self-test failed: with too few datapoints the collector must decline to " \
        "judge rather than guess -> %s" % reason

    # ---- 7. NEGATIVE: null is allowed, absent is not ----
    # The distinction the fix rests on. "We looked and there was no history" is
    # a claim; a missing key is nobody having filled it in. If both were
    # accepted the reader could no longer tell them apart.
    base = json.loads(json.dumps(snap))
    base["metrics"][0]["baseline"] = None
    assert not validate_signals(base), \
        "self-test failed: an explicit null baseline must be accepted"

    missing = json.loads(json.dumps(snap))
    del missing["metrics"][0]["baseline"]
    absent_problems = validate_signals(missing)
    scores["missing-baseline-is-red"] = bool(absent_problems)
    assert absent_problems, \
        "self-test failed: a metric with NO baseline key must be RED"
    assert any("baseline" in problem for problem in absent_problems), \
        "self-test failed: the violation must name the missing field -> %s" % absent_problems

    return scores, barren_message


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
    contract, ghost_problem = contract_selftest()
    collected, barren_message = collector_selftest()
    assert all(contract.values()) and all(collected.values()), \
        "self-test failed: contract/collector families must be green -> %s %s" % (contract, collected)

    consumption, iteration_message, budget_message = consumption_selftest()
    assert all(consumption.values()), \
        "self-test failed: the consumption family must be green -> %s" % consumption

    print("self-test OK: good plan passes all checks; bad plan fails all checks;")
    print("              and both consumption caps were demonstrated by tripping.")
    print("  contract     :", contract)
    print("               ->", ghost_problem[:104])
    print("  collector    :", collected)
    print("    no data    ->", barren_message.splitlines()[0][:104])
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
