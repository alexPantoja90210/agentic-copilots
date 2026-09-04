"""
Selftest for the ground-truth -> prompt wiring (IA-48 criterion 3).

The property under test is not "it produces text". It is that the text carries
the incident and NOT its answer -- including through the back door of which
metrics were chosen. A builder that shows CPU for CPU faults and network for
stops has encoded the label in the shape of its input, and arm B would then
score well by reading our minds.

No AWS, no credentials, no cost: the clients are fakes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import build_incidents as bi
import chain_topology as ct

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


CHAIN = {
    "db":  ("i-0db",  None,  "running"),
    "app": ("i-0app", "db",  "running"),
    "web": ("i-0web", "app", "running"),
}


class FakeEC2:
    def describe_instances(self, Filters=None, **kwargs):  # noqa: N803
        instances = []
        for role, (iid, upstream, state) in CHAIN.items():
            tags = [{"Key": "Pilot", "Value": "IA-45"},
                    {"Key": "ChainRole", "Value": role}]
            if upstream:
                tags.append({"Key": "DependsOn", "Value": upstream})
            instances.append({"InstanceId": iid, "State": {"Name": state},
                              "PrivateIpAddress": "10.0.0.1", "Tags": tags})
        return {"Reservations": [{"Instances": instances}]}


class FakeCloudWatch:
    """
    Datapoints for every instance except `silent`, which returns none.

    A stopped node publishes nothing, and the empty series is the signal. It has
    to survive the whole path as an empty list rather than being dropped or
    turned into a zero, so the fake reproduces exactly that.
    """

    def __init__(self, silent=None, start=None):
        self.silent = silent
        self.start = start or datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        self.asked = []

    def get_metric_statistics(self, **kwargs):
        self.asked.append((kwargs["MetricName"],
                           kwargs["Dimensions"][0]["Value"],
                           kwargs["StartTime"], kwargs["EndTime"],
                           kwargs["Period"]))
        if kwargs["Dimensions"][0]["Value"] == self.silent:
            return {"Datapoints": []}
        points = []
        for index in range(6):
            points.append({"Timestamp": self.start + timedelta(minutes=5 * index),
                           "Average": 3100.0 + index})
        return {"Datapoints": points}


def opened(incident_id, fault, role, start="2026-09-03T03:00:00+00:00",
           end="2026-09-03T03:10:00+00:00", **extra):
    entry = {"record": "open", "incident_id": incident_id, "fault": fault,
             "fault_class": "x", "instance_id": "i-0" + role,
             "window_start": start, "window_end_planned": end,
             "operator": "test", "dry_run": False, "node_role": role,
             "topology": {"db": None, "app": "db", "web": "app"}}
    entry.update(extra)
    return entry


def closed(incident_id, outcome="completed"):
    return {"record": "close", "incident_id": incident_id, "outcome": outcome}


def run() -> int:
    nodes = ct.discover(FakeEC2())

    # ---- the leak that would hide in the metric LIST ----------------------
    # Two different fault classes, two different cause nodes. If the built
    # prompts do not carry the same metric names, the choice of metric is the
    # answer and the experiment is over before a model sees it.
    entries = [opened("F3-db-1", "F3", "db"), closed("F3-db-1"),
               opened("F1-web-1", "F1", "web",
                      start="2026-09-03T05:00:00+00:00",
                      end="2026-09-03T05:10:00+00:00"), closed("F1-web-1")]
    built, skipped = bi.build_all(entries, nodes, FakeCloudWatch())
    check("both incidents build", len(built) == 2, "skipped=%s" % skipped)

    def metric_names(text):
        return {name for name in bi.METRICS if name in text}

    names = [metric_names(incident["arm_b"]) for incident in built]
    check("the metric list does not depend on the fault class",
          names[0] == names[1] and names[0] == set(bi.METRICS),
          "F3 carried %s, F1 carried %s" % (sorted(names[0]), sorted(names[1])))

    # ---- arm A must survive verbatim inside arm B -------------------------
    check("arm A appears verbatim inside arm B",
          all(i["arm_a"] in i["arm_b"] for i in built))
    check("the two arms are not the same text",
          all(i["arm_a"] != i["arm_b"] for i in built))

    # ---- the paged node is the symptom, not the cause ---------------------
    by_id = {incident["incident_id"]: incident for incident in built}
    check("a fault at the bottom of the chain pages the top",
          by_id["F3-db-1"]["symptomatic_node"] == "web",
          by_id["F3-db-1"]["symptomatic_node"])
    check("a fault at the top pages itself",
          by_id["F1-web-1"]["symptomatic_node"] == "web")
    check("and the cause is kept separately from the symptom",
          by_id["F3-db-1"]["fault_role"] == "db")

    # ---- the stopped node: absent, not zero -------------------------------
    silent = FakeCloudWatch(silent="i-0db")
    built_silent, _ = bi.build_all([opened("F3-db-2", "F3", "db"),
                                    closed("F3-db-2")], nodes, silent)
    text = built_silent[0]["arm_b"]
    check("a node with no datapoints is named as not reporting",
          "not reporting is not the same as a service reporting zero" in text
          or "no datapoints" in text, text[-400:])
    check("and no zero is invented for it", " 0\n" not in text and " 0.0" not in text)

    # ---- padding, applied identically to every incident --------------------
    probe = FakeCloudWatch()
    bi.build_all([opened("F3-db-3", "F3", "db"), closed("F3-db-3")], nodes, probe)
    starts = {asked[2] for asked in probe.asked}
    ends = {asked[3] for asked in probe.asked}
    periods = {asked[4] for asked in probe.asked}
    fault_start = datetime.fromisoformat("2026-09-03T03:00:00+00:00")
    check("the window is padded before the fault, so a baseline exists",
          starts == {fault_start - timedelta(minutes=bi.PRE_MINUTES)}, str(starts))
    check("and after it, so the recovery is visible",
          ends == {datetime.fromisoformat("2026-09-03T03:10:00+00:00")
                   + timedelta(minutes=bi.POST_MINUTES)}, str(ends))
    check("every series is asked at the 5-minute period EC2 actually publishes",
          periods == {300}, str(periods))
    check("every node is queried, including the one that failed",
          {asked[1] for asked in probe.asked} == {"i-0db", "i-0app", "i-0web"})

    # ---- NEGATIVE: the exemption must not swallow a real leak -------------
    # Exempting the metric names is only sound if a genuine label word is still
    # caught. Without this check the exemption is a hole nobody would notice
    # until the pilot's results were already meaningless.
    import incident_builder as ib

    original_summary = ib.SUMMARY
    ib.SUMMARY = original_summary + " Investigate the cpu saturation on {node}."
    try:
        _, skipped_leak = bi.build_all(
            [opened("F1-web-leak", "F1", "web"), closed("F1-web-leak")],
            nodes, FakeCloudWatch())
        check("a real label word in the prose is still refused, exemption or not",
              len(skipped_leak) == 1 and "gives away the answer" in skipped_leak[0]["reason"],
              str(skipped_leak))
    finally:
        ib.SUMMARY = original_summary

    # And the same text with the exemption NOT applied must also be refused --
    # otherwise the check above could be passing for the wrong reason.
    metrics = bi.fetch_metrics(FakeCloudWatch(), opened("F3-db-x", "F3", "db"), nodes)
    try:
        ib.build(opened("F1-web-x", "F1", "web"), nodes, metrics, "web")
    except ib.BuilderError as exc:
        check("without the exemption the metric names alone trip the detector",
              "gives away the answer" in str(exc), str(exc))
    else:
        check("without the exemption the metric names alone trip the detector",
              False, "it built -- so the exemption is not what made the F1 case pass")

    # ---- IA-61: contamination travels with the built incident -------------
    # The injector now refuses to create these, but six already exist and no
    # code can repair them. What can be done is refuse to let them look clean.
    near = [opened("F3-db-N", "F3", "db",
                   start="2026-09-04T02:00:00+00:00", end="2026-09-04T02:25:00+00:00"),
            closed("F3-db-N"),
            opened("F0-web-N", "F0", "web",           # one minute later: the
                   start="2026-09-04T02:26:00+00:00", # exact shape of 4 Sep
                   end="2026-09-04T02:51:00+00:00"),
            closed("F0-web-N")]
    built_near, _ = bi.build_all(near, nodes, FakeCloudWatch())
    near_by_id = {i["incident_id"]: i for i in built_near}

    check("a neighbour inside the padded range is recorded",
          [c["incident_id"] for c in near_by_id["F0-web-N"]["contaminated_by"]] == ["F3-db-N"],
          str(near_by_id["F0-web-N"]["contaminated_by"]))
    check("and the record says which fault and which node it was",
          near_by_id["F0-web-N"]["contaminated_by"][0]["fault"] == "F3"
          and near_by_id["F0-web-N"]["contaminated_by"][0]["node_role"] == "db")
    # NOT mutual, and that asymmetry is the point. A control induces nothing, so
    # its window carries no foreign event into anybody else's prompt. It still
    # needs protecting itself -- more than anything else does, since its whole
    # claim is that its window was empty.
    check("a control does not contaminate its neighbour, having no signal to lend",
          near_by_id["F3-db-N"]["contaminated_by"] == [],
          str(near_by_id["F3-db-N"]["contaminated_by"]))

    # Between two REAL faults it stays mutual: each one's event sits in the
    # other's context.
    both_real = [opened("F3-db-R", "F3", "db",
                        start="2026-09-04T02:00:00+00:00", end="2026-09-04T02:25:00+00:00"),
                 closed("F3-db-R"),
                 opened("F1-web-R", "F1", "web",
                        start="2026-09-04T02:26:00+00:00", end="2026-09-04T02:51:00+00:00"),
                 closed("F1-web-R")]
    built_real, _ = bi.build_all(both_real, nodes, FakeCloudWatch())
    check("between two real faults the contamination is mutual",
          all(len(i["contaminated_by"]) == 1 for i in built_real),
          str([(i["incident_id"], i["contaminated_by"]) for i in built_real]))

    far = [opened("F3-db-F", "F3", "db",
                  start="2026-09-04T02:00:00+00:00", end="2026-09-04T02:25:00+00:00"),
           closed("F3-db-F"),
           opened("F0-web-F", "F0", "web",            # PRE minutes after the end
                  start="2026-09-04T03:25:00+00:00",
                  end="2026-09-04T03:50:00+00:00"),
           closed("F0-web-F")]
    built_far, _ = bi.build_all(far, nodes, FakeCloudWatch())
    check("a properly separated pair is recorded as clean",
          all(i["contaminated_by"] == [] for i in built_far),
          str([(i["incident_id"], i["contaminated_by"]) for i in built_far]))

    # A failed restore ran LONGER than planned, so the real outage is what has
    # to be checked against the neighbour -- not the window we intended.
    overran = [opened("F3-db-O", "F3", "db",
                      start="2026-09-04T02:00:00+00:00", end="2026-09-04T02:25:00+00:00"),
               {"record": "close", "incident_id": "F3-db-O", "outcome": "failed",
                "window_end_actual": "2026-09-04T03:30:00+00:00"},
               opened("F0-web-O", "F0", "web",
                      start="2026-09-04T03:25:00+00:00",
                      end="2026-09-04T03:50:00+00:00"),
               closed("F0-web-O")]
    built_over, _ = bi.build_all(overran, nodes, FakeCloudWatch())
    over_by_id = {i["incident_id"]: i for i in built_over}
    check("an overrunning neighbour is caught by its ACTUAL end, not its planned one",
          len(over_by_id["F0-web-O"]["contaminated_by"]) == 1,
          "planned separation was clean; the outage actually ran into it")

    # ---- what must be skipped, and said out loud --------------------------
    _, skipped = bi.build_all(
        [opened("DRY-1", "F3", "db", dry_run=True), closed("DRY-1")], nodes,
        FakeCloudWatch())
    check("a dry run is excluded with its reason",
          len(skipped) == 1 and "dry run" in skipped[0]["reason"], str(skipped))

    legacy = opened("F3-old-1", "F3", "db")
    del legacy["node_role"]
    _, skipped = bi.build_all([legacy, closed("F3-old-1")], nodes, FakeCloudWatch())
    check("an incident with no recorded cause node is skipped, not guessed",
          len(skipped) == 1 and "node_role" in skipped[0]["reason"], str(skipped))

    # ---- a failed restore ran longer than planned, and says so ------------
    built_failed, _ = bi.build_all(
        [opened("F3-db-4", "F3", "db"), closed("F3-db-4", outcome="failed")],
        nodes, FakeCloudWatch())
    check("an incident whose restore failed is kept",
          len(built_failed) == 1)
    check("and flagged as having overrun its planned window",
          built_failed[0]["window_overran"] is True)
    check("while a clean one is not",
          by_id["F3-db-1"]["window_overran"] is False)

    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
