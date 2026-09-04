"""
Run a planned sequence of injections, spaced so no prompt sees a neighbour's fault.

    python run_corpus.py --plan                 # print the schedule, touch nothing
    python run_corpus.py --go                   # run it

Why this exists
---------------
IA-61: the prompt an agent reads is the fault window plus `PRE_MINUTES` of
lead-in and `POST_MINUTES` of tail. Two injections closer than that clearance
put one incident's fault inside another incident's evidence, and the label
names only one of them. The first corpus was built back-to-back and five of its
six incidents were contaminated.

The clearance is asymmetric, and the asymmetry is not a convenience:

  after a real fault   PRE minutes   -- the next prompt needs a clean baseline
  after a control      POST minutes  -- a control induces nothing, so only its
                                        own tail has to stay empty

Both numbers are read from `incident_builder`. Restating them here is how a
scheduler ends up spacing injections for a padding the builder no longer uses.

What it does NOT do
-------------------
It does not decide the plan, retry a failed injection, or skip ahead. If one
injection fails, the sequence stops: the next one's pre-flight would refuse
anyway (the chain is not running), and continuing past a failure is how a
corpus acquires an incident nobody can explain.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ground_truth as gt
import incident_builder as ib

# The default plan for the four incidents IA-61 left to re-inject. Faults and
# controls interleaved, because a control costs POST after it instead of PRE.
DEFAULT_PLAN = [("F1", "web"), ("F0", "web"), ("F3", "app"), ("F3", "db")]

DURATION_MINUTES = 25


def clearance_after(fault: str) -> int:
    """Minutes the next window must wait after this one's window ends."""
    return ib.POST_MINUTES if fault in gt.SIGNAL_FREE_FAULTS else ib.PRE_MINUTES


def schedule(plan, duration=DURATION_MINUTES, start=None):
    """[(fault, node, opens_at, closes_at, wait_before)] for the whole plan."""
    now = start or datetime.now(timezone.utc)
    out, cursor = [], now
    for index, (fault, node) in enumerate(plan):
        wait = 0
        if index:
            previous_fault = plan[index - 1][0]
            wait = clearance_after(previous_fault)
            cursor += timedelta(minutes=wait)
        out.append((fault, node, cursor, cursor + timedelta(minutes=duration), wait))
        cursor += timedelta(minutes=duration)
    return out


def render(rows) -> str:
    lines = ["  #  fault  node   opens (UTC)   closes        waited before"]
    for index, (fault, node, opens, closes, wait) in enumerate(rows, 1):
        lines.append("  %d  %-5s  %-5s  %s         %s      %s"
                     % (index, fault, node, opens.strftime("%H:%M"),
                        closes.strftime("%H:%M"),
                        "-" if not wait else "%d min" % wait))
    total = (rows[-1][3] - rows[0][2]).total_seconds() / 3600
    lines.append("\n  %d injections, %.1f hours end to end." % (len(rows), total))
    return "\n".join(lines)


def preflight(plan, duration, ec2=None, cloudwatch=None) -> list[str]:
    """
    Everything knowable BEFORE the first injection. Returns the reasons to stop.

    The first run of this sequencer started a 4.2-hour plan against a chain that
    was switched off, and found out at step one. That cost nothing. The same
    mistake with the failing step at position three costs two hours of waiting
    to be told something that was true before the first command.

    So: check the chain once, and check the credit balance for every CPU fault
    in the plan against the balance it will need -- not the balance now, but the
    balance at the time that injection actually opens, which is higher because
    the instance accrues while the earlier windows run.
    """
    import inject

    reasons = []
    ec2 = ec2 or inject.session(None).client("ec2")
    nodes, _role = inject.resolve_target(ec2, plan[0][1])

    down = sorted(role for role, node in nodes.items() if node["state"] != "running")
    if down:
        reasons.append(
            "the chain is not running: %s. Start all three, let the services "
            "settle, and plan again." % ", ".join(down))
        return reasons          # nothing else can be checked meaningfully

    rows = schedule(plan, duration)
    cloudwatch = cloudwatch or inject.session(None).client("cloudwatch")
    for (fault, node, opens, _closes, _wait) in rows:
        if fault != "F1":
            continue
        instance_id = nodes[node]["instance_id"]
        instance_type = inject.instance_type_of(ec2, instance_id)
        needed = (inject.credit_cost(instance_type, duration)
                  + inject.F1_END_BALANCE_FLOOR)
        balance, observed_at = inject.latest_credit_balance(cloudwatch, instance_id)
        if balance is None:
            reasons.append("%s on %s: no CPUCreditBalance datapoint yet. A freshly "
                           "started instance publishes every 5 minutes; wait for one."
                           % (fault, node))
            continue
        # It will still be accruing while the earlier windows run.
        earns_per_hour = inject.BURSTABLE[instance_type]["earn_per_hour"]
        hours = max((opens - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0)
        projected = min(balance + earns_per_hour * hours, 288.0)
        if projected < needed:
            reasons.append(
                "%s on %s opens at %s UTC with about %.0f credits projected "
                "(%.0f now, accruing %.0f/hour) and needs %.0f. Start the chain "
                "earlier or shorten the window."
                % (fault, node, opens.strftime("%H:%M"), projected, balance,
                   earns_per_hour, needed))
    return reasons


def run(plan, duration, dry_run=False, sleeper=time.sleep, runner=None) -> int:
    """
    Execute the plan. Returns the number of injections that completed.

    `runner` is injectable so a test can drive the sequencing without AWS.
    """
    runner = runner or _invoke_inject
    completed = 0
    for index, (fault, node) in enumerate(plan):
        if index:
            wait = clearance_after(plan[index - 1][0])
            resume = datetime.now(timezone.utc) + timedelta(minutes=wait)
            print("\n  waiting %d min before %s on %s "
                  "(clearance after %s) -- resumes about %s UTC"
                  % (wait, fault, node, plan[index - 1][0],
                     resume.strftime("%H:%M")), flush=True)
            sleeper(wait * 60)

        print("\n=== %d/%d: %s on %s ===" % (index + 1, len(plan), fault, node),
              flush=True)
        code = runner(fault, node, duration, dry_run)
        if code != 0:
            print("\n!! %s on %s exited %d. Stopping the sequence." % (fault, node, code),
                  file=sys.stderr)
            print("!! The remaining injections would be refused anyway: their "
                  "pre-flight requires the whole chain running, and a failed "
                  "restore leaves a node stopped.", file=sys.stderr)
            print("!! %d of %d completed. Fix the target, then re-run with the "
                  "rest of the plan." % (completed, len(plan)), file=sys.stderr)
            return completed
        completed += 1
    return completed


def _invoke_inject(fault, node, duration, dry_run) -> int:
    cmd = [sys.executable, str(Path(__file__).with_name("inject.py")),
           fault, "--duration", str(duration), "--node", node]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true",
                    help="print the schedule and exit; touches nothing")
    ap.add_argument("--go", action="store_true", help="actually run it")
    ap.add_argument("--duration", type=int, default=DURATION_MINUTES)
    ap.add_argument("--dry-run", action="store_true",
                    help="pass --dry-run to every injection")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="start without checking the chain or the credit "
                         "projections. There is no good reason; it exists so "
                         "that using it is a visible choice.")
    args = ap.parse_args(argv)

    plan = DEFAULT_PLAN
    rows = schedule(plan, args.duration)
    print("Clearance: %d min after a fault, %d min after a control "
          "(from incident_builder)." % (ib.PRE_MINUTES, ib.POST_MINUTES))
    print(render(rows))

    if not args.go:
        print("\n  --plan only. Add --go to run it.")
        return 0

    if not args.dry_run and not args.skip_preflight:
        reasons = preflight(plan, args.duration)
        if reasons:
            print("\nRefusing to start. What a 4.2-hour plan can know now:",
                  file=sys.stderr)
            for reason in reasons:
                print("  - %s" % reason, file=sys.stderr)
            return 1
        print("\n  pre-flight: chain running, credits sufficient for every CPU "
              "fault in the plan.")

    completed = run(plan, args.duration, dry_run=args.dry_run)
    print("\n%d of %d injections completed." % (completed, len(plan)))
    return 0 if completed == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
