"""
Selftest for the injection sequencer.

The property under test is not "it runs four commands". It is that the gaps it
leaves are the gaps the BUILDER needs, that a control is charged less than a
fault, and that a failure stops the sequence instead of marching on.

The last one matters most: continuing past a failed injection is how a corpus
acquires an incident nobody can explain, and the schedule would look complete
while one of its windows never happened.

No AWS, no waiting: the clock and the injector are both injected.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ground_truth as gt
import incident_builder as ib
import run_corpus as rc

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def run() -> int:
    T0 = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    PRE, POST, D = ib.PRE_MINUTES, ib.POST_MINUTES, 25

    # ---- the gaps are the builder's, not a second set of numbers ----
    plan = [("F3", "db"), ("F1", "web")]
    rows = rc.schedule(plan, D, start=T0)
    gap = (rows[1][2] - rows[0][3]).total_seconds() / 60
    check("after a fault the next window waits the full lead-in",
          gap == PRE, "waited %s, PRE is %s" % (gap, PRE))

    plan = [("F0", "web"), ("F3", "db")]
    rows = rc.schedule(plan, D, start=T0)
    gap = (rows[1][2] - rows[0][3]).total_seconds() / 60
    check("after a control it waits only the tail", gap == POST,
          "waited %s, POST is %s" % (gap, POST))
    check("and the tail really is the cheaper one", POST < PRE)

    # ---- the schedule must satisfy the guard that will judge it -------------
    # The real check: feed the planned windows to open_incident itself. If the
    # sequencer and the guard ever disagree, the sequencer is the one that is
    # wrong, and this is where it shows.
    import tempfile
    from pathlib import Path

    log = Path(tempfile.mkdtemp()) / "gt.jsonl"
    rows = rc.schedule(rc.DEFAULT_PLAN, D, start=T0)
    accepted, refusal = 0, ""
    for index, (fault, node, opens, closes, _wait) in enumerate(rows):
        try:
            gt.open_incident(
                incident_id="PLAN-%d" % index, fault=fault, instance_id="i-0x",
                window_start=opens, window_end_planned=closes,
                operator="selftest", path=log,
                context_pre_minutes=PRE, context_post_minutes=POST)
            accepted += 1
        except gt.GroundTruthError as exc:
            refusal = str(exc)
            break
    check("every window the sequencer plans is accepted by the guard",
          accepted == len(rows), "accepted %d of %d: %s" % (accepted, len(rows), refusal))

    # NEGATIVE: shrink one gap by a minute and the guard must refuse it. Without
    # this the check above could pass because the guard is asleep.
    log = Path(tempfile.mkdtemp()) / "gt.jsonl"
    tight = rc.schedule([("F3", "db"), ("F1", "web")], D, start=T0)
    gt.open_incident(incident_id="T0", fault="F3", instance_id="i-0x",
                     window_start=tight[0][2], window_end_planned=tight[0][3],
                     operator="selftest", path=log,
                     context_pre_minutes=PRE, context_post_minutes=POST)
    try:
        gt.open_incident(
            incident_id="T1", fault="F1", instance_id="i-0x",
            window_start=tight[1][2] - timedelta(minutes=1),
            window_end_planned=tight[1][3] - timedelta(minutes=1),
            operator="selftest", path=log,
            context_pre_minutes=PRE, context_post_minutes=POST)
        check("RED: one minute tighter and the guard refuses", False,
              "the guard accepted a gap the sequencer would not have planned")
    except gt.GroundTruthError:
        check("RED: one minute tighter and the guard refuses", True)

    # ---- a failure stops the sequence --------------------------------------
    calls = []

    def failing_runner(fault, node, duration, dry_run):
        calls.append((fault, node))
        return 0 if len(calls) < 2 else 1        # the second one fails

    completed = rc.run([("F3", "db"), ("F1", "web"), ("F0", "web")], D,
                       sleeper=lambda _s: None, runner=failing_runner)
    check("a failed injection stops the sequence", completed == 1,
          "completed=%d" % completed)
    check("and nothing after it is attempted", len(calls) == 2,
          "it attempted %d injections: %s" % (len(calls), calls))

    ok_calls = []

    def good_runner(fault, node, duration, dry_run):
        ok_calls.append((fault, node))
        return 0

    completed = rc.run(rc.DEFAULT_PLAN, D, sleeper=lambda _s: None, runner=good_runner)
    check("a clean run completes every injection in order",
          completed == len(rc.DEFAULT_PLAN) and ok_calls == rc.DEFAULT_PLAN,
          str(ok_calls))

    # ---- the pre-flight has to know before it starts, not at step three ----
    class FakeEC2:
        def __init__(self, states):
            self.states = states

        def describe_instances(self, Filters=None, InstanceIds=None, **kw):
            if InstanceIds:
                return {"Reservations": [{"Instances": [
                    {"InstanceId": InstanceIds[0], "InstanceType": "t3.micro"}]}]}
            out = []
            for role, state in self.states.items():
                tags = [{"Key": "Pilot", "Value": "IA-45"},
                        {"Key": "ChainRole", "Value": role}]
                upstream = {"app": "db", "web": "app"}.get(role)
                if upstream:
                    tags.append({"Key": "DependsOn", "Value": upstream})
                out.append({"InstanceId": "i-0" + role, "State": {"Name": state},
                            "PrivateIpAddress": "10.0.0.1", "Tags": tags})
            return {"Reservations": [{"Instances": out}]}

    class FakeCW:
        """
        CPUCreditBalance at a fixed value, and NetworkIn as a run of datapoints
        going back `history_minutes` from now. `history_gap_at` punches a hole
        that many minutes back, so the unbroken-run logic can be exercised.
        """

        def __init__(self, balance, history_minutes=600, history_gap_at=None):
            self.balance = balance
            self.history_minutes = history_minutes
            self.history_gap_at = history_gap_at

        def get_metric_statistics(self, **kw):
            if kw["MetricName"] == "CPUCreditBalance":
                if self.balance is None:
                    return {"Datapoints": []}
                return {"Datapoints": [{"Timestamp": T0, "Average": self.balance}]}
            now = datetime.now(timezone.utc)
            points = []
            step = 5
            for minutes_ago in range(0, self.history_minutes + 1, step):
                if self.history_gap_at and minutes_ago == self.history_gap_at:
                    continue
                points.append({"Timestamp": now - timedelta(minutes=minutes_ago),
                               "Average": 3100.0})
            return {"Datapoints": points}

    all_up = {"db": "running", "app": "running", "web": "running"}

    reasons = rc.preflight(rc.DEFAULT_PLAN, D,
                           ec2=FakeEC2({**all_up, "web": "stopped"}),
                           cloudwatch=FakeCW(200))
    check("a chain that is not running is caught before the first injection",
          len(reasons) == 1 and "chain is not running" in reasons[0], str(reasons))

    reasons = rc.preflight(rc.DEFAULT_PLAN, D,
                           ec2=FakeEC2(all_up), cloudwatch=FakeCW(200))
    check("a healthy chain with credits to spare starts", reasons == [], str(reasons))

    # The case that justifies the whole function: enough credits for the CPU
    # fault, checked BEFORE four hours of waiting rather than when it is reached.
    reasons = rc.preflight(rc.DEFAULT_PLAN, D,
                           ec2=FakeEC2(all_up), cloudwatch=FakeCW(5))
    check("a CPU fault that cannot afford its window is refused up front",
          len(reasons) == 1 and "needs" in reasons[0], str(reasons))
    check("and the refusal names the projected balance, not just the current one",
          "projected" in reasons[0], reasons[0] if reasons else "")

    reasons = rc.preflight(rc.DEFAULT_PLAN, D,
                           ec2=FakeEC2(all_up), cloudwatch=FakeCW(None))
    check("no credit datapoint at all is refused, not assumed sufficient",
          len(reasons) == 1 and "no CPUCreditBalance" in reasons[0], str(reasons))

    # ---- the first window's lead-in has to contain data --------------------
    # The check that was missed by hand: a plan about to open its CPU fault
    # fifteen minutes after boot would have spent four hours producing one
    # incident whose baseline was a 60-minute absence.
    reasons = rc.preflight(rc.DEFAULT_PLAN, D, ec2=FakeEC2(all_up),
                           cloudwatch=FakeCW(200, history_minutes=15))
    check("a chain with only minutes of history is refused",
          any("lead-in" in r for r in reasons), str(reasons))
    check("and the refusal says when the lead-in becomes available",
          any("is not available until" in r for r in reasons), str(reasons))

    reasons = rc.preflight(rc.DEFAULT_PLAN, D, ec2=FakeEC2(all_up),
                           cloudwatch=FakeCW(200, history_minutes=600))
    check("hours of continuous history is accepted", reasons == [], str(reasons))

    # A hole 20 minutes back means the unbroken run is only 20 minutes long,
    # however much older data sits behind it. An instance that ran yesterday,
    # was stopped, and started again has exactly this shape.
    reasons = rc.preflight(rc.DEFAULT_PLAN, D, ec2=FakeEC2(all_up),
                           cloudwatch=FakeCW(200, history_minutes=600,
                                             history_gap_at=20))
    check("old data behind a gap does not count as history",
          any("lead-in" in r for r in reasons),
          "a stopped-and-restarted instance has old datapoints and no baseline")

    # A plan with no CPU fault needs no balance at all.
    reasons = rc.preflight([("F3", "db"), ("F0", "web")], D,
                           ec2=FakeEC2(all_up), cloudwatch=FakeCW(None))
    check("a plan with no CPU fault does not ask about credits", reasons == [],
          str(reasons))

    # ---- one clock, in a module that prints times next to each other -------
    # The first live refusal read "continuous data only since 13:28 ... the
    # first window opens at 19:38" -- correct arithmetic, six hours of
    # incoherence, because boto3 returns the machine's zone and the schedule is
    # built in UTC. Same defect as IA-59, new file.
    from datetime import timezone as _tz, timedelta as _td
    minus_six = _tz(_td(hours=-6))
    check("a timestamp in another zone renders as UTC",
          rc._utc(T0.astimezone(minus_six)) == rc._utc(T0) == "12:00",
          "%s vs %s" % (rc._utc(T0.astimezone(minus_six)), rc._utc(T0)))

    class LocalCW(FakeCW):
        """CloudWatch as boto3 actually returns it here: not in UTC."""

        def get_metric_statistics(self, **kw):
            got = FakeCW.get_metric_statistics(self, **kw)
            for point in got["Datapoints"]:
                point["Timestamp"] = point["Timestamp"].astimezone(minus_six)
            return got

    reasons = rc.preflight(rc.DEFAULT_PLAN, D, ec2=FakeEC2(all_up),
                           cloudwatch=LocalCW(200, history_minutes=15))
    check("a refusal built from non-UTC timestamps still speaks one clock",
          reasons and all(":" in r for r in reasons)
          and not any("13:" in r and "19:" in r for r in reasons),
          str(reasons))
    check("and the zone does not change the verdict",
          bool(reasons) == bool(rc.preflight(rc.DEFAULT_PLAN, D,
                                             ec2=FakeEC2(all_up),
                                             cloudwatch=FakeCW(200, history_minutes=15))))

    # ---- the plan itself has to be worth running ---------------------------
    # Two clean F3s survive from the first corpus (IA-61). The plan must add
    # what they lack, or 4.2 hours buys another set of stop faults.
    planned_classes = {fault for fault, _node in rc.DEFAULT_PLAN}
    check("the plan introduces fault classes the surviving corpus lacks",
          {"F1", "F0"} <= planned_classes, str(planned_classes))
    planned_nodes = {node for _fault, node in rc.DEFAULT_PLAN}
    check("and covers more than one cause node", len(planned_nodes) >= 2,
          str(planned_nodes))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
