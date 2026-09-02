"""
Read the chain's network metrics per node, as a series.

IA-55 criterion 3 asks whether a dependency failure is DISTINGUISHABLE at the
resolution CloudWatch actually offers. That question cannot be answered by an
average -- the collector aggregates a window to one number, and a collapse
halfway through a window shows up as "somewhat less traffic", which is exactly
the reading that would let a wrong conclusion pass. So this prints the series.

Run it twice: once with the chain healthy, once after stopping a node. The
verdict is whether a human can point at the datapoint where the fault landed.

    python chain_check.py --minutes 40 --injector-role-arn <arn>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import chain_topology as ct
from inject import session

METRICS = ("NetworkIn", "NetworkOut")
PERIOD = 300  # what basic monitoring actually publishes


def series(cw, instance_id, metric, start, end):
    raw = cw.get_metric_statistics(
        Namespace="AWS/EC2", MetricName=metric,
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start, EndTime=end, Period=PERIOD, Statistics=["Average"],
    )
    return sorted(((p["Timestamp"], p["Average"]) for p in raw.get("Datapoints", [])),
                  key=lambda pair: pair[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print the chain's traffic, per node, over time.")
    ap.add_argument("--minutes", type=int, default=40)
    ap.add_argument("--metric", default="NetworkIn", choices=METRICS)
    ap.add_argument("--injector-role-arn", default=None)
    args = ap.parse_args(argv)

    sess = session(args.injector_role_arn)
    ec2, cw = sess.client("ec2"), sess.client("cloudwatch")

    nodes = ct.discover(ec2)
    print(ct.describe(nodes))
    print()

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=args.minutes)

    order = sorted(nodes, key=lambda r: len(ct.chain_from(r, nodes)), reverse=True)
    data = {role: dict(series(cw, nodes[role]["instance_id"], args.metric, start, end))
            for role in order}

    stamps = sorted({t for points in data.values() for t in points})
    if not stamps:
        print("no datapoints in the last %d minutes." % args.minutes)
        print("Basic monitoring publishes every 5 minutes; if the chain was just")
        print("installed, wait and run again. An empty read is not a quiet system.")
        return 1

    header = "  time (UTC)   " + "".join("%12s" % r for r in order)
    print("%s  — %s, KB per 5-minute datapoint" % (header, args.metric))
    print("  " + "-" * (len(header) + 2))
    for stamp in stamps:
        row = "  %s  " % stamp.astimezone(timezone.utc).strftime("%H:%M")
        for role in order:
            value = data[role].get(stamp)
            row += "%12s" % ("—" if value is None else "%.0f" % (value / 1024.0))
        print(row)

    print()
    print("  A node with no datapoint at a timestamp was not reporting: stopped,")
    print("  or not yet publishing. That gap is a signal, not a blank.")
    for role in order:
        if nodes[role]["state"] != "running":
            print("  NOTE: %s is currently %s." % (role, nodes[role]["state"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
