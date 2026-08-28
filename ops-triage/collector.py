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
    "CPUCreditBalance": ("lt", 30.0, "major", "cpu_credit_balance"),
}

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


def _query(client, namespace, metric_name, instance_id, start, end, period):
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
    if not points:
        return None
    return sum(point[STATISTIC] for point in points) / len(points)


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
    metric_names = list(THRESHOLDS) + list(CONTEXT_METRICS)

    for instance_id in targets:
        for metric_name in metric_names:
            value = _query(client, namespace, metric_name, instance_id,
                           window_start, now,
                           choose_period(window_start, now, now))
            if value is None:
                skipped.append({"metric": metric_name, "target": instance_id,
                                "reason": "no datapoints in the collection window"})
                continue

            baseline = _query(client, namespace, metric_name, instance_id,
                              baseline_start, window_start,
                              choose_period(baseline_start, window_start, now))
            if baseline is None:
                skipped.append({"metric": metric_name, "target": instance_id,
                                "reason": "no datapoints in the baseline window, "
                                          "so no comparison is possible"})
                continue

            metric_id = "M-%s-%s" % (instance_id[-6:], metric_name)
            metrics.append({
                "id": metric_id,
                "service": instance_id,
                "name": metric_name,
                "value": round(value, 4),
                "baseline": round(baseline, 4),
            })

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
            "skipped_metrics": skipped,
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
