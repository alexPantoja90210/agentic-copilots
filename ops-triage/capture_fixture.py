"""
Record the account's CURRENT CloudWatch responses as a replayable fixture.

Why now and not later: on 1 Sep 2026 this account holds a state that is about to
disappear on its own — an instance a few hours old, whose metrics have values and
whose baseline windows are empty. That combination is exactly what IA-53 needs as
a regression fixture, and it is the combination that made the collector discard
every metric and abort. In a week the instance will have history and this state
can only be recreated by destroying and rebuilding the box.

A young instance is a perishable test fixture. Capture it while it exists.

The queries here are not a reimplementation: the windows, the metric list and the
periods come from `collector` itself, so what is recorded is what the collector
would have received, not an approximation of it.

Instance ids are replaced with stable synthetic ones. The mapping is NOT written
anywhere — the repository takes the shape of the data, never the identifiers of
the environment.

    python capture_fixture.py [region] [window_hours] [baseline_days]
"""

from __future__ import annotations

import datetime
import json
import sys

import collector

OUT = "fixtures/young_instance_no_baseline.json"


def _alias(index: int) -> str:
    # Well-formed but obviously synthetic: a repeated hex nibble.
    return "i-0" + (format(index + 10, "x") * 16)


def capture(region, window_hours, baseline_days):
    client = collector._client(region)
    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(hours=window_hours)
    baseline_start = window_start - datetime.timedelta(days=baseline_days)

    targets = collector.discover_targets(client, "AWS/EC2")
    if not targets:
        raise SystemExit("no targets; nothing to capture")

    aliases = {real: _alias(i) for i, real in enumerate(sorted(targets))}

    window_period = collector.choose_period(window_start, now, now)
    baseline_period = collector.choose_period(baseline_start, window_start, now)

    metric_names = list(collector.THRESHOLDS) + list(collector.CONTEXT_METRICS)
    responses = {}
    summary = {"window_with_data": 0, "window_empty": 0,
               "baseline_with_data": 0, "baseline_empty": 0}

    for real in sorted(targets):
        for name in metric_names:
            for kind, start, end, period in (
                ("window", window_start, now, window_period),
                ("baseline", baseline_start, window_start, baseline_period),
            ):
                raw = client.get_metric_statistics(
                    Namespace="AWS/EC2", MetricName=name,
                    Dimensions=[{"Name": "InstanceId", "Value": real}],
                    StartTime=start, EndTime=end, Period=period,
                    Statistics=[collector.STATISTIC],
                )
                points = [
                    {"Average": round(float(p[collector.STATISTIC]), 6),
                     "Timestamp": p["Timestamp"].astimezone(datetime.timezone.utc).isoformat()}
                    for p in raw.get("Datapoints", [])
                ]
                responses["%s|%s|%s" % (aliases[real], name, kind)] = {"Datapoints": points}
                summary["%s_%s" % (kind, "with_data" if points else "empty")] += 1

    return {
        "captured_at": now.isoformat(),
        "region": region,
        "namespace": "AWS/EC2",
        "statistic": collector.STATISTIC,
        "window_hours": window_hours,
        "baseline_days": baseline_days,
        "window_period_seconds": window_period,
        "baseline_period_seconds": baseline_period,
        "targets": [aliases[t] for t in sorted(targets)],
        "why_this_exists": (
            "Recorded from a live account whose instance was hours old. Values are "
            "present and every baseline window is empty. The collector discarded "
            "every metric on this input and aborted (IA-53). Instance ids are "
            "synthetic; the mapping to the real ones was never written down."
        ),
        "summary": summary,
        "responses": responses,
    }


if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else collector.REGION_DEFAULT
    window_hours = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    baseline_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    data = capture(region, window_hours, baseline_days)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")

    s = data["summary"]
    print("captured %d responses from %d target(s)" % (len(data["responses"]), len(data["targets"])))
    print("  window   : %d with data, %d empty" % (s["window_with_data"], s["window_empty"]))
    print("  baseline : %d with data, %d empty" % (s["baseline_with_data"], s["baseline_empty"]))
    print("  written  : %s" % OUT)
    if s["baseline_with_data"]:
        print("")
        print("  NOTE: some baseline windows already have data. The young-instance")
        print("  state is already fading; this capture is partial. Say so in IA-53")
        print("  rather than presenting it as the clean case.")
