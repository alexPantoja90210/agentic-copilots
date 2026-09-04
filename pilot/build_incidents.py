"""
Ground-truth log + CloudWatch + the topology -> the two arm inputs, per incident.

This is the wiring IA-48 criterion 3 asks for. It owns no judgement: the label
comes from `ground_truth`, the graph from `chain_topology`, the numbers from
`collector`, and the prompt assembly from `incident_builder`. Nothing here
decides what an incident means.

    python build_incidents.py --list
    python build_incidents.py --out C:\\dev\\ia-pilot\\incidents.jsonl

The one rule this file exists to enforce
----------------------------------------
**Every incident is built with the SAME metric list, whatever its fault class.**

It would be natural to show CPUUtilization for a CPU fault and NetworkIn for a
stop. It would also destroy the experiment: the choice of metrics would encode
the answer, and arm B would score well by reading our minds instead of the
graph. METRICS is a module-level constant for that reason, and a selftest
asserts the built prompts of two different fault classes carry the same metric
names.

Padding
-------
The window in the ground-truth log is the FAULT. The window an on-call engineer
looks at is wider, and a series that starts at the fault has no baseline to be
abnormal against. PRE_MINUTES of context before and POST_MINUTES after, applied
identically to every incident.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ops-triage"))

import chain_topology as ct
import collector
import ground_truth as gt
import incident_builder as ib

# Identical for every incident, every fault class. See the module docstring.
METRICS = ("CPUUtilization", "NetworkIn", "NetworkOut", "CPUCreditBalance")

# Defined in incident_builder so the injector can read them without importing
# this module. See IA-61: the guard that protects the prompt must use the same
# numbers the prompt is built with.
PRE_MINUTES = ib.PRE_MINUTES
POST_MINUTES = ib.POST_MINUTES
PERIOD_SECONDS = 300          # EC2 basic monitoring. Asking for 60 returns less.

DEFAULT_OUT = Path(r"C:\dev\ia-pilot\incidents.jsonl")


class BuildError(Exception):
    pass


def pair_records(entries: list[dict]) -> list[tuple[dict, dict | None]]:
    """
    (open, close) per incident id, oldest first.

    A missing close is kept rather than dropped: an incident whose harness died
    mid-window is still a real outage with a certain cause, and silently
    discarding it would let the corpus quietly select for runs that went well.
    """
    opens, closes = {}, {}
    for entry in entries:
        if entry.get("record") == "open":
            opens[entry["incident_id"]] = entry
        elif entry.get("record") == "close":
            closes[entry["incident_id"]] = entry
    return [(opens[key], closes.get(key))
            for key in sorted(opens, key=lambda k: opens[k]["window_start"])]


def fault_role(entry: dict) -> str:
    """The node the operator broke. Recorded at injection time, never inferred."""
    role = entry.get("node_role")
    if not role:
        raise BuildError(
            "%s predates the three-node design and carries no node_role. It "
            "cannot be scored on which service failed, because the log does "
            "not say. Excluded rather than guessed."  % entry["incident_id"])
    return role


def fetch_metrics(client, entry: dict, nodes: dict) -> dict:
    """
    {role: {metric: [(datetime, value), ...]}} for the padded window.

    Every node, every metric in METRICS, including the node that was stopped --
    whose absence of datapoints is the signal, and must reach the builder as an
    empty list rather than as a missing key.
    """
    start = datetime.fromisoformat(entry["window_start"]) - timedelta(minutes=PRE_MINUTES)
    end = datetime.fromisoformat(entry["window_end_planned"]) + timedelta(minutes=POST_MINUTES)

    metrics = {}
    for role, node in nodes.items():
        per_metric = {}
        for name in METRICS:
            per_metric[name] = collector.series(
                client, name, node["instance_id"], start, end,
                period=PERIOD_SECONDS)
        metrics[role] = per_metric
    return metrics


def contaminants(entry: dict, entries: list[dict]) -> list[dict]:
    """
    Other incidents' faults that fall inside this one's padded range.

    IA-61: the injector now refuses to create these, but the six incidents built
    on 3 and 4 Sep already exist and no code can repair them. What can be done
    is refuse to let them look clean -- the contamination travels with the built
    incident and the scorer can see it.

    A neighbour's ACTUAL end is used where it is known: a restore that failed
    ran longer than its planned window, and the metrics show the real outage,
    not the intended one.
    """
    mine_lo = datetime.fromisoformat(entry["window_start"]) - timedelta(minutes=PRE_MINUTES)
    mine_hi = datetime.fromisoformat(entry["window_end_planned"]) + timedelta(minutes=POST_MINUTES)

    closes = {e["incident_id"]: e for e in entries if e.get("record") == "close"}
    found = []
    for other in entries:
        if other.get("record") != "open" or other["incident_id"] == entry["incident_id"]:
            continue
        if other.get("dry_run"):
            continue
        # A control induces nothing, so its window holds no foreign event. It
        # occupies time and contributes no signal, and counting it as a
        # contaminant overstates how crowded the corpus really is -- which
        # would then overstate how much of it has to be injected again.
        if other["fault"] in gt.SIGNAL_FREE_FAULTS:
            continue
        start = datetime.fromisoformat(other["window_start"])
        closed = closes.get(other["incident_id"], {})
        end = datetime.fromisoformat(
            closed.get("window_end_actual") or other["window_end_planned"])
        if start < mine_hi and end > mine_lo:
            found.append({"incident_id": other["incident_id"],
                          "fault": other["fault"],
                          "node_role": other.get("node_role"),
                          "window_start": other["window_start"]})
    return found


def build_all(entries, nodes, client) -> tuple[list[dict], list[dict]]:
    """Returns (built, skipped). A skip is always recorded with its reason."""
    built, skipped = [], []
    for opened, closed in pair_records(entries):
        if opened.get("dry_run"):
            skipped.append({"incident_id": opened["incident_id"],
                            "reason": "dry run: no fault was induced"})
            continue
        try:
            role = fault_role(opened)
            metrics = fetch_metrics(client, opened, nodes)
            incident = ib.build(opened, nodes, metrics, role,
                                uniform_terms=METRICS)
        except (BuildError, ib.BuilderError, ct.TopologyError) as exc:
            skipped.append({"incident_id": opened["incident_id"],
                            "reason": "%s: %s" % (type(exc).__name__, exc)})
            continue

        # The label travels with the built incident so the scorer never has to
        # re-derive it, and so a mismatch between the two is impossible rather
        # than merely unlikely.
        incident["fault"] = opened["fault"]
        incident["fault_role"] = role
        incident["window_start"] = opened["window_start"]
        incident["window_end_planned"] = opened["window_end_planned"]
        incident["restore_outcome"] = (closed or {}).get("outcome", "unknown")
        # An incident whose restore failed ran LONGER than its planned window.
        # The metrics will show that, and the write-up must not pretend the
        # outage was ten minutes because the plan said so.
        incident["window_overran"] = bool(
            closed and closed.get("outcome") == "failed")
        incident["contaminated_by"] = contaminants(opened, entries)
        built.append(incident)
    return built, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(gt.DEFAULT_LOG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--list", action="store_true",
                    help="show what would be built and exit; touches no AWS")
    ap.add_argument("--show", metavar="INCIDENT_ID",
                    help="print both arms for one incident, for eyeballing")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args(argv)

    entries = gt.read_all(Path(args.log))
    pairs = pair_records(entries)
    if args.list:
        for opened, closed in pairs:
            print("%-28s %-3s %-5s %-9s %s" % (
                opened["incident_id"], opened["fault"],
                opened.get("node_role") or "-",
                (closed or {}).get("outcome", "open"),
                "DRY RUN" if opened.get("dry_run") else ""))
        print("\n%d incident(s) in %s" % (len(pairs), args.log))
        return 0

    import boto3  # noqa: PLC0415 - only needed on the paths that touch AWS
    session = boto3.session.Session(region_name=args.region)
    nodes = ct.discover(session.client("ec2"))
    built, skipped = build_all(entries, nodes, session.client("cloudwatch"))

    out = Path(args.out)
    gt.check_location(out)          # never inside a repo, never under OneDrive
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for incident in built:
            handle.write(json.dumps(incident, ensure_ascii=False) + "\n")

    for incident in built:
        dirty = incident["contaminated_by"]
        print("built  %-28s %-3s cause=%-4s paged=%-4s usable=%s%s%s" % (
            incident["incident_id"], incident["fault"], incident["fault_role"],
            incident["symptomatic_node"], incident["usable"],
            "  WINDOW OVERRAN" if incident["window_overran"] else "",
            "  CONTAMINATED by %d" % len(dirty) if dirty else ""))
    for entry in skipped:
        print("skip   %-28s %s" % (entry["incident_id"], entry["reason"]),
              file=sys.stderr)

    if args.show:
        for incident in built:
            if incident["incident_id"] == args.show:
                print("\n===== ARM A =====\n%s\n\n===== ARM B =====\n%s"
                      % (incident["arm_a"], incident["arm_b"]))
                break
        else:
            print("no built incident with id %s" % args.show, file=sys.stderr)
            return 1

    print("\n%d built, %d skipped -> %s" % (len(built), len(skipped), out))
    # A corpus of one fault class cannot distinguish reasoning from a rule of
    # thumb. Say so here rather than discovering it while reading the scores.
    dirty = [i for i in built if i["contaminated_by"]]
    if dirty:
        print("\n!! %d of %d built incidents carry another incident's fault inside"
              " the context their prompt shows (IA-61)." % (len(dirty), len(built)),
              file=sys.stderr)
        for incident in dirty:
            print("!!   %s <- %s" % (incident["incident_id"], ", ".join(
                "%s (%s on %s)" % (c["incident_id"], c["fault"], c["node_role"])
                for c in incident["contaminated_by"])), file=sys.stderr)
        print("!! The label names one cause and the prompt shows more than one"
              " event. Scoring these mixes 'the context did not help' with"
              " 'the context contained something else'.", file=sys.stderr)

    classes = {incident["fault"] for incident in built}
    roles = {incident["fault_role"] for incident in built}
    if len(classes) < 2 or len(roles) < 2:
        print("\n!! This corpus has %d fault class(es) and %d cause node(s)."
              % (len(classes), len(roles)), file=sys.stderr)
        print("!! An agent answering with one fixed rule would score as well as"
              " one that reasons.", file=sys.stderr)
        print("!! Inject a different fault class and a different node before"
              " running IA-49.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
