"""
CloudWatch -> signals.json. Read-only, and it never invents a number.

WHAT IT COLLECTS
    EC2 metrics, per instance, from the live CloudWatch API. Each metric gets a
    `value` (the recent window) and a `baseline` (a longer preceding window), and
    alerts are derived from declared thresholds -- never asserted by anything.

WHAT IT DOES NOT COLLECT, AND SAYS SO IN THE OUTPUT
    `recent_changes` is deploy history. That is CloudTrail or a CI system, not
    CloudWatch. `runbook` is operational knowledge, written by people.

    Both are declared in `source.not_collected` with the reason. The consumer
    can then tell "nothing happened" from "nobody is looking", which are very
    different things to a triage agent and indistinguishable in an empty list.

THE INVARIANT, borrowed from IA-38 because it was the right one
    Every metric considered is either EMITTED or RECORDED IN `skipped_metrics`
    with a reason. Never neither. A collector that quietly drops a metric with
    no datapoints produces a snapshot that looks complete and is not.

WHY THE OUTPUT IS NOT signals.json
    The committed `signals.json` is the hand-written worked example, and it stays
    that way. This writes `signals.live.json`, which is gitignored: it contains
    real instance ids from a real account, and this repository is public.

THRESHOLDS ARE A CHOICE, NOT A TRUTH
    The numbers in THRESHOLDS below are judgement calls, written here so they can
    be argued with. Nothing in AWS says 80% CPU is a problem. Making them visible
    and editable is the difference between a threshold and a magic number.
"""
import datetime
import json

REGION_DEFAULT = "us-east-1"
OUTPUT_DEFAULT = "signals.live.json"

# Metric -> (comparison, threshold, severity, human signal name).
# "gt" fires when value > threshold; "lt" when value < threshold.
THRESHOLDS = {
    "CPUUtilization": ("gt", 80.0, "major", "cpu_utilization"),
    "StatusCheckFailed": ("gt", 0.0, "critical", "status_check"),
}

# IA-51. CPUCreditBalance used to live in THRESHOLDS as ("lt", 30.0, ...), and
# that rule could not tell two opposite situations apart:
#
#   exhausted    - a long-running instance that has burned its balance and is
#                  now throttled. Worth waking someone for.
#   not accrued  - an instance too young to have earned credits yet. Nothing is
#                  wrong; time is the only fix.
#
# Both sit below 30, so a brand-new healthy t3.micro was reported MAJOR for the
# first two and a half hours of its life. That is the alert that teaches an
# on-call engineer to ignore alerts.
#
# What separates them is direction, not level: a balance that is FALLING is
# depletion; one that is RISING is accrual. Direction needs the series, which is
# why this metric is judged by `credit_verdict` below instead of by a constant.
#
# The two numbers here are judgement calls, stated so they can be argued with:
#
#   TREND_TOLERANCE  a t3.micro earns 12 credits/hour. Losing less than 1/hour
#                    is noise or near-equilibrium, not depletion.
#   EXHAUSTED_FLOOR  at or below 1 credit there is nothing left to spend. This
#                    catches the case the trend alone misses: an instance
#                    already pinned at zero is FLAT, not falling, and would
#                    otherwise stay silent while throttled.
#   MIN_TREND_POINTS two points can be a blip. Below this the collector says it
#                    cannot judge, rather than guessing.
TREND_METRICS = ("CPUCreditBalance",)
TREND_TOLERANCE = 1.0
EXHAUSTED_FLOOR = 1.0
MIN_TREND_POINTS = 3

# Metrics collected for context even when no threshold applies to them.
CONTEXT_METRICS = ("NetworkIn", "NetworkOut", "EBSReadOps", "EBSWriteOps")

STATISTIC = "Average"

# get_metric_statistics returns at most 1440 datapoints per call, and CloudWatch
# refuses fine resolutions for old data: 1-minute survives 15 days, 5-minute 63
# days, 1-hour 455. A fixed period works until the window grows or the data ages
# — and then the call fails, or silently returns nothing.
MAX_DATAPOINTS = 1440
VALID_PERIODS = (60, 300, 900, 3600, 10800, 21600, 86400)


def choose_period(start, end, now):
    """
    The smallest valid period that keeps the request under the datapoint cap AND
    is coarse enough for how old the data is.

    Computed rather than assumed: the first live run of this collector asked for
    a 3-hour window and got nothing, and the obvious fix — widen the window —
    would have hit the 1440 cap instead. Two failures, one root cause: a constant
    where a function belonged.
    """
    span = max((end - start).total_seconds(), 60)
    age_days = max((now - start).total_seconds() / 86400.0, 0)

    floor = 60
    if age_days > 63:
        floor = 3600
    elif age_days > 15:
        floor = 300

    for period in VALID_PERIODS:
        if period >= floor and span / period <= MAX_DATAPOINTS:
            return period
    return VALID_PERIODS[-1]


class CollectorError(RuntimeError):
    """Raised when the collector cannot do its job. Never swallowed."""


def _client(region):
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - environment dependent
        raise CollectorError(
            "boto3 is not installed. The collector needs it; the selftest does "
            "not, which is why this import is here and not at module scope."
        ) from error
    return boto3.client("cloudwatch", region_name=region)


def discover_targets(client, namespace="AWS/EC2", probe="CPUUtilization"):
    """
    Which instances have metrics at all. Discovery comes from CloudWatch itself
    rather than from a hardcoded list, so the collector reports what the account
    actually has instead of what someone assumed it had.
    """
    targets = []
    paginator = client.get_paginator("list_metrics")
    for page in paginator.paginate(Namespace=namespace, MetricName=probe):
        for metric in page.get("Metrics", []):
            for dimension in metric.get("Dimensions", []):
                if dimension["Name"] == "InstanceId":
                    targets.append(dimension["Value"])
    return sorted(set(targets))


def _query_points(client, namespace, metric_name, instance_id, start, end, period):
    """
    The datapoints, oldest first. One call, one code path to CloudWatch.

    `_query` used to average inside the request function, which meant the series
    was thrown away before anyone could look at it. IA-51 needs the direction of
    travel, and direction is a property of the series, not of its mean.
    """
    response = client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=[STATISTIC],
    )
    points = response.get("Datapoints", [])
    return sorted(points, key=lambda point: point["Timestamp"])


def series(client, metric_name, instance_id, start, end,
           period=300, namespace="AWS/EC2"):
    """
    One metric's datapoints for one instance as [(datetime, value), ...].

    The public way to ask this module for a SERIES rather than a snapshot. The
    pilot's incident builder needs the shape of a window, not a verdict about
    it, and it must reach CloudWatch through the same code path as everything
    else -- a second, private caller is a second set of assumptions about
    periods, statistics and ordering that nothing keeps in step.

    `period` defaults to 300 because EC2 basic monitoring publishes every five
    minutes and asking for 60 returns an emptier series, not a finer one.
    """
    points = _query_points(client, namespace, metric_name, instance_id,
                           start, end, period)
    return [(point["Timestamp"], point[STATISTIC]) for point in points]


def _query(client, namespace, metric_name, instance_id, start, end, period):
    points = _query_points(client, namespace, metric_name, instance_id,
                           start, end, period)
    if not points:
        return None
    return sum(point[STATISTIC] for point in points) / len(points)


def credit_verdict(points, tolerance=None, floor=None, min_points=None):
    """
    Judge a CPUCreditBalance series by where it is GOING, not only where it is.

    Returns (fires, slope_per_hour, reason). `slope_per_hour` is None when the
    series is too short to have a direction, and the reason says so instead of
    the collector inventing one.
    """
    tolerance = TREND_TOLERANCE if tolerance is None else tolerance
    floor = EXHAUSTED_FLOOR if floor is None else floor
    min_points = MIN_TREND_POINTS if min_points is None else min_points

    if len(points) < min_points:
        return False, None, (
            "only %d datapoint(s) in the window; at least %d are needed to tell "
            "a falling balance from a rising one, so no verdict is claimed"
            % (len(points), min_points))

    first, last = points[0], points[-1]
    hours = (last["Timestamp"] - first["Timestamp"]).total_seconds() / 3600.0
    if hours <= 0:
        return False, None, "the datapoints carry no elapsed time; no direction can be derived"

    slope = (last[STATISTIC] - first[STATISTIC]) / hours

    # Already spent. Flat at zero is not falling, so the trend alone would miss it.
    if last[STATISTIC] <= floor and slope <= 0:
        return True, slope, (
            "balance is %.2f, at or below the exhausted floor of %.2f, and not recovering"
            % (last[STATISTIC], floor))

    if slope < -tolerance:
        return True, slope, (
            "balance is falling at %.2f credits/hour, faster than the tolerance of %.2f"
            % (slope, tolerance))

    if slope > 0:
        return False, slope, (
            "balance is rising at %.2f credits/hour: the instance is accruing, not depleting"
            % slope)

    return False, slope, "balance is stable at %.2f credits/hour" % slope


def collect(region=REGION_DEFAULT, window_hours=3, baseline_days=7,
            namespace="AWS/EC2", client=None, now=None):
    """
    Build a snapshot from CloudWatch. Returns the dict; writing is the caller's.

    `client` is injectable so the eval harness can drive this with recorded
    responses -- no credentials, no network, no cost.
    """
    client = client or _client(region)
    now = now or datetime.datetime.now(datetime.timezone.utc)

    window_start = now - datetime.timedelta(hours=window_hours)
    baseline_start = window_start - datetime.timedelta(days=baseline_days)

    targets = discover_targets(client, namespace)
    if not targets:
        raise CollectorError(
            "no instance in %s reports metrics in namespace %s. There is nothing "
            "to collect, and an empty snapshot would be a claim about a quiet "
            "system rather than about an absent one." % (region, namespace))

    services = list(targets)
    metrics, alerts, skipped = [], [], []
    # Two states that are NOT "skipped". A metric here was emitted; what is
    # missing is context or a verdict, not the measurement. Folding these into
    # skipped_metrics would break the invariant that a metric is either emitted
    # or explained and never both -- and would quietly turn "we have this value"
    # into "we do not have this metric".
    no_baseline, not_judged = [], []
    metric_names = list(THRESHOLDS) + list(TREND_METRICS) + list(CONTEXT_METRICS)

    for instance_id in targets:
        for metric_name in metric_names:
            points = _query_points(client, namespace, metric_name, instance_id,
                                   window_start, now,
                                   choose_period(window_start, now, now))
            value = (sum(p[STATISTIC] for p in points) / len(points)) if points else None
            if value is None:
                skipped.append({"metric": metric_name, "target": instance_id,
                                "reason": "no datapoints in the collection window"})
                continue

            baseline = _query(client, namespace, metric_name, instance_id,
                              baseline_start, window_start,
                              choose_period(baseline_start, window_start, now))

            # IA-53. A missing baseline used to discard the metric entirely.
            # That is the wrong trade: no threshold rule reads the baseline --
            # `fired` compares value against threshold and nothing else -- so a
            # measured value was being thrown away to protect a comparison that
            # never happens. On an instance younger than the baseline window
            # EVERY metric hit this branch, the metrics list came out empty, and
            # the IA-4 abort fired: the collector reported that it had found
            # nothing, on an account that was emitting normally. A real injected
            # outage sat in CloudWatch, visible, and unread.
            #
            # Now the value is kept and the absence is stated, not implied.
            # `baseline: None` reaches the contract as an explicit null, which
            # it accepts only in this field and only when written on purpose.
            if baseline is None:
                no_baseline.append({"metric": metric_name, "target": instance_id,
                                    "reason": "no datapoints in the baseline window, "
                                              "so the metric is emitted without a "
                                              "baseline and no comparison is claimed"})

            metric_id = "M-%s-%s" % (instance_id[-6:], metric_name)
            metrics.append({
                "id": metric_id,
                "service": instance_id,
                "name": metric_name,
                "value": round(value, 4),
                "baseline": None if baseline is None else round(baseline, 4),
            })

            # IA-51. Two kinds of rule now, because two kinds of question.
            # A level rule asks "is this value bad?". A trend rule asks "is this
            # going the wrong way?" — the only question that separates an
            # exhausted balance from one that has simply not accrued yet.
            if metric_name in TREND_METRICS:
                fires, slope, reason = credit_verdict(points)
                if slope is None:
                    not_judged.append({"metric": metric_name, "target": instance_id,
                                       "reason": reason})
                if fires:
                    alerts.append({
                        "id": "AL-%s-%s" % (instance_id[-6:], metric_name),
                        "service": instance_id,
                        "signal": "cpu_credit_balance",
                        "severity": "major",
                        # The alert reports the RATE, because the rate is what
                        # was judged. Reporting the level here would invite the
                        # reader to re-derive the old, wrong comparison.
                        "value": round(slope, 4),
                        "threshold": round(-TREND_TOLERANCE, 4),
                        "first_seen": window_start.isoformat(),
                        "reason": reason,
                    })
                continue

            rule = THRESHOLDS.get(metric_name)
            if rule is None:
                continue
            comparison, threshold, severity, signal = rule
            fired = value > threshold if comparison == "gt" else value < threshold
            if fired:
                alerts.append({
                    "id": "AL-%s-%s" % (instance_id[-6:], metric_name),
                    "service": instance_id,
                    "signal": signal,
                    "severity": severity,
                    "value": round(value, 4),
                    "threshold": threshold,
                    "first_seen": window_start.isoformat(),
                })

    # Targets but no data is the same lie as no targets, wearing a valid shape.
    # Since IA-53 this fires only when no metric had a VALUE, which is the real
    # emptiness. A missing baseline no longer silences a metric.
    # A snapshot whose metrics list is empty reads as "the system is quiet"; what
    # actually happened is that nobody looked where the data lives. Running this
    # against an account whose instances were terminated days ago is exactly how
    # the gap was found, and the empty snapshot passed the contract cleanly.
    if not metrics:
        reasons = sorted({entry["reason"] for entry in skipped})
        raise CollectorError(
            "found %d target(s) in %s but not one metric had usable datapoints, "
            "so nothing was collected. An empty snapshot would read as a healthy "
            "quiet system; what happened is that the window found no data.\n"
            "  window: %s to %s\n"
            "  reasons: %s\n"
            "  If the instances are stopped or terminated, widen the window: "
            "CloudWatch keeps 15 days at 1-minute, 63 at 5-minute and 455 at "
            "1-hour resolution."
            % (len(targets), region, window_start.isoformat(), now.isoformat(),
               "; ".join(reasons)))

    return {
        "window": "last_%dh" % window_hours,
        "services": services,
        "alerts": alerts,
        "metrics": metrics,
        "source": {
            "collector": "cloudwatch",
            "region": region,
            "collected_at": now.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": now.isoformat(),
            "statistic": STATISTIC,
            "window_period_seconds": choose_period(window_start, now, now),
            "baseline_period_seconds": choose_period(baseline_start, window_start, now),
            "thresholds": {name: {"comparison": rule[0], "threshold": rule[1],
                                  "severity": rule[2]}
                           for name, rule in THRESHOLDS.items()},
            # Published for the same reason as the thresholds: a rule the reader
            # cannot see is a magic number, whether it is a level or a slope.
            "trend_rules": {name: {"judged_by": "direction over the window",
                                   "tolerance_per_hour": TREND_TOLERANCE,
                                   "exhausted_floor": EXHAUSTED_FLOOR,
                                   "min_points": MIN_TREND_POINTS,
                                   "severity": "major"}
                            for name in TREND_METRICS},
            "skipped_metrics": skipped,
            "metrics_without_baseline": no_baseline,
            "metrics_not_judged": not_judged,
            "not_collected": {
                "recent_changes": "deploy history is not in CloudWatch. It comes "
                                  "from CloudTrail or a CI system, and neither is "
                                  "wired up. An empty list here would read as "
                                  "'nothing was deployed', which is not what is known.",
                "runbook": "operational knowledge, written by people. There is "
                           "nothing to collect it from.",
            },
        },
    }


def write_snapshot(snapshot, path=OUTPUT_DEFAULT):
    from signals_contract import SignalsContractError, describe_problems, validate_signals

    problems = validate_signals(snapshot)
    if problems:
        raise SignalsContractError(describe_problems("the collected snapshot", problems))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
        handle.write("\n")
    return path


if __name__ == "__main__":
    import sys

    region = sys.argv[1] if len(sys.argv) > 1 else REGION_DEFAULT
    window_hours = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    baseline_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    snapshot = collect(region=region, window_hours=window_hours,
                       baseline_days=baseline_days)
    path = write_snapshot(snapshot)

    print("Collected from CloudWatch in %s (window %dh, baseline %dd)"
          % (region, window_hours, baseline_days))
    print("  targets  : %d  -> %s" % (len(snapshot["services"]),
                                      ", ".join(snapshot["services"])))
    print("  metrics  : %d emitted, %d skipped and recorded"
          % (len(snapshot["metrics"]), len(snapshot["source"]["skipped_metrics"])))
    print("  alerts   : %d" % len(snapshot["alerts"]))
    for alert in snapshot["alerts"]:
        print("     %s %s on %s: %s vs threshold %s"
              % (alert["severity"].upper(), alert["signal"], alert["service"],
                 alert["value"], alert["threshold"]))
    if not snapshot["alerts"]:
        print("     none — no declared threshold was crossed. That is a finding,")
        print("     not an empty result: the thresholds are in source.thresholds.")
    print("  written  : %s  (gitignored — it carries real instance ids)" % path)
    print("")
    print("NOT collected, by design:")
    for field, reason in snapshot["source"]["not_collected"].items():
        print("  %s: %s" % (field, reason.split(".")[0] + "."))
