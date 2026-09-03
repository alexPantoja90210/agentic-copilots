"""
Fault injection for the IA-45 pilot.

Produces incidents whose root cause is not a matter of opinion: if the operator
stops the instance, the label is certain because the operator caused it.

    python inject.py F1 --duration 12 --dry-run
    python inject.py F3 --duration 8

Refusals, all of them deliberate:
  - it will not touch an instance that does not carry the pilot tag;
  - it will not run a CPU fault against a burstable instance in "unlimited"
    mode, where sustained load is billed as surplus instead of draining the
    credit balance (the defect found in the live account on 1 Sep 2026);
  - it will not open a window that overlaps another;
  - it restores the previous state even when the fault fails.

This module knows nothing about the model, the prompts or the scoring. Keeping
the thing that generates truth away from the thing that judges answers is the
same rule as D3, and it applies to the harness too.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import chain_topology as ct
import ground_truth as gt

REGION = "us-east-1"
PILOT_TAG_KEY = "Pilot"
PILOT_TAG_VALUE = "IA-45"

# A busy loop, one worker per vCPU, bounded by the shell itself so that the
# load cannot outlive the window even if this script dies.
BURN_COMMAND = (
    'END=$(( $(date +%s) + {seconds} )); '
    'for i in $(seq 1 $(nproc)); do '
    '  ( while [ $(date +%s) -lt $END ]; do :; done ) & '
    'done; wait'
)


class InjectionError(Exception):
    pass


def _boto3():
    try:
        import boto3  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise InjectionError(
            "boto3 is not installed. python -m pip install boto3"
        ) from exc
    return boto3


def session(injector_role_arn: str | None):
    boto3 = _boto3()
    if not injector_role_arn:
        return boto3.Session(region_name=REGION)
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=injector_role_arn,
        RoleSessionName=f"ia-pilot-{uuid.uuid4().hex[:8]}",
        DurationSeconds=3600,
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=REGION,
    )


def resolve_target(ec2, role: str | None = None):
    """
    Find the node to break, by its ROLE in the dependency chain.

    Until IA-55 the pilot had one instance and this function refused whenever it
    found more than one, which was correct then and became a wall the moment the
    chain existed: three tagged instances, and the harness could not inject
    anything at all.

    Roles rather than instance ids, because the role is what the experiment is
    about. "Stop the database" survives the machine being replaced; an id does
    not, and IA-46 already replaced this account's instance once.

    Returns (nodes, role) with the topology alongside, since every caller that
    needs a target also needs the graph the target sits in.
    """
    nodes = ct.discover(ec2)

    if role is None:
        if len(nodes) == 1:
            role = next(iter(nodes))
        else:
            raise InjectionError(
                "this account has %d pilot nodes (%s). Name the one to break with "
                "--node: which service fails is the experiment's variable, not "
                "something for the harness to choose."
                % (len(nodes), ", ".join(sorted(nodes))))

    if role not in nodes:
        raise InjectionError(
            f"'{role}' is not a node in this graph. Known roles: "
            + ", ".join(sorted(nodes)))
    return nodes, role


def assert_standard_credits(ec2, instance_id: str) -> str:
    """A CPU fault against an 'unlimited' t-instance is a charge, not a signal."""
    spec = ec2.describe_instance_credit_specifications(InstanceIds=[instance_id])
    creds = spec.get("InstanceCreditSpecifications", [])
    if not creds:
        raise InjectionError(
            f"{instance_id} reports no credit specification. It may not be burstable; "
            "confirm before running a CPU fault."
        )
    mode = creds[0].get("CpuCredits", "").lower()
    if mode != "standard":
        raise InjectionError(
            f"{instance_id} is in '{mode}' credit mode. Under 'unlimited', sustained "
            "CPU is billed as surplus credits instead of draining CPUCreditBalance: "
            "F1 would generate real spend against the zero-spend budget (D4) and F2 "
            "could not happen at all. Set credit_specification { cpu_credits = "
            '"standard" } and apply before injecting a CPU fault.'
        )
    return mode


RESTORE_ATTEMPTS = 3
RESTORE_BACKOFF_S = (15, 45, 90)


def current_state(ec2, instance_id: str) -> str:
    got = ec2.describe_instances(InstanceIds=[instance_id])
    return got["Reservations"][0]["Instances"][0]["State"]["Name"]


def start_with_retries(ec2, instance_id: str, attempts: int = RESTORE_ATTEMPTS,
                       sleeper=time.sleep, timeout_s: int = 300):
    """
    Start the instance, and mean it.

    On 1 Sep 2026 the first real F3 left the pilot target stopped: AWS answered
    the start with `Server.InternalError`, the harness tried exactly once, gave
    up, and recorded the failure in a JSON file nobody was watching. A manual
    retry minutes later succeeded on the first attempt, which is the ordinary
    behaviour of that error — it is transient and usually lands on different
    hardware next time.

    One attempt is not a serious effort to recover from a transient failure.

    Returns (running, attempts_made, restored_at, errors).
    """
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            wait_for_state(ec2, instance_id, "running", timeout_s=timeout_s)
            return True, attempt, datetime.now(timezone.utc), errors
        except Exception as exc:  # noqa: BLE001 - every failure is data
            errors.append("attempt %d: %s: %s" % (attempt, type(exc).__name__, exc))
            print("  restore attempt %d failed: %s" % (attempt, exc), file=sys.stderr)
            if attempt < attempts:
                pause = RESTORE_BACKOFF_S[min(attempt - 1, len(RESTORE_BACKOFF_S) - 1)]
                print("  retrying in %ds" % pause, file=sys.stderr)
                sleeper(pause)
    return False, attempts, None, errors


def wait_for_state(ec2, instance_id: str, state: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        got = ec2.describe_instances(InstanceIds=[instance_id])
        now = got["Reservations"][0]["Instances"][0]["State"]["Name"]
        if now == state:
            return
        time.sleep(10)
    raise InjectionError(f"{instance_id} did not reach state '{state}' within {timeout_s}s.")


# `ensure_running` was removed when the chain landed. It used to start a stopped
# target and wait three minutes for metrics to settle. With three nodes that
# behaviour became wrong, not merely unused: starting a chain and injecting
# immediately would open the window before the traffic had stabilised, and the
# first datapoints would record the bootstrap rather than the system. The harness
# now REFUSES a chain that is not already up, and the operator starts it and
# lets it settle. Recorded here because deleting a function silently invites
# somebody to write it again.


def burn_cpu(ssm, instance_id: str, seconds: int, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would send an SSM RunShellScript busy loop for {seconds}s")
        return
    cmd = BURN_COMMAND.format(seconds=seconds)
    sent = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="IA-45 pilot F1/F2 load",
        Parameters={"commands": [cmd]},
        TimeoutSeconds=max(60, seconds + 120),
    )
    command_id = sent["Command"]["CommandId"]
    print(f"  SSM command {command_id} dispatched; burning for {seconds}s")
    time.sleep(seconds + 15)
    inv = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
    print(f"  SSM status: {inv['Status']}")
    if inv["Status"] not in ("Success", "Delivery Timed Out"):
        raise InjectionError(
            f"SSM command ended as {inv['Status']}: {inv.get('StandardErrorContent', '')[:400]}"
        )


def stop_and_hold(ec2, instance_id: str, seconds: int, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would stop {instance_id}, hold {seconds}s, then start it again")
        return
    ec2.stop_instances(InstanceIds=[instance_id])
    wait_for_state(ec2, instance_id, "stopped")
    print(f"  stopped; holding {seconds}s")
    time.sleep(seconds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inject one labelled fault for the IA-45 pilot.")
    ap.add_argument("fault", choices=sorted(gt.FAULT_CLASSES))
    ap.add_argument("--duration", type=int, default=12, help="fault duration in minutes")
    ap.add_argument("--node", default=None,
                    help="which service to break: a ChainRole such as db, app or web")
    ap.add_argument("--injector-role-arn", default=None, help="role to assume; omit to use ambient credentials")
    ap.add_argument("--operator", default="alejandro", help="who ran it, recorded in the label")
    ap.add_argument("--dry-run", action="store_true", help="every step except the fault itself")
    args = ap.parse_args(argv)

    seconds = args.duration * 60

    sess = session(args.injector_role_arn)
    ec2 = sess.client("ec2")
    ssm = sess.client("ssm")

    nodes, role = resolve_target(ec2, args.node)
    # The id names the service. Two F3s on different nodes are different
    # incidents with different correct answers, and an identifier that cannot
    # tell them apart makes the log ambiguous exactly where it must not be.
    incident_id = f"{args.fault}-{role}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    instance_id = nodes[role]["instance_id"]
    print(ct.describe(nodes))
    print()
    print(f"target: {role} ({nodes[role]['state']})")

    # Every node must be up before the fault. An incident injected while some
    # OTHER node is already down carries two faults under one label, and the
    # second one is invisible: nothing in the metrics distinguishes "app is
    # quiet because I stopped db just now" from "app was quiet all along".
    #
    # This is the case the single-node pre-flight could not see. It checked the
    # target; the chain means the target is not the only thing that can be wrong.
    if not args.dry_run:
        down = sorted(r for r, n in nodes.items() if n["state"] != "running")
        if down:
            raise InjectionError(
                "these services are not running: %s. Refusing to open a window: "
                "the incident would carry more than one fault under a single "
                "label, and the extra one leaves no trace a reader could find. "
                "Start the whole chain, let it settle, and try again."
                % ", ".join(down))

    if args.fault in ("F1", "F2"):
        mode = assert_standard_credits(ec2, instance_id)
        print(f"credit mode: {mode}")

    window_start = datetime.now(timezone.utc)
    window_end_planned = window_start + timedelta(seconds=seconds)

    # A dry run must not write into the real log. Its record would occupy a
    # window, and the overlap guard would then refuse the real injection that
    # the rehearsal was meant to prepare — the safety check turned into an
    # obstacle by the safety check. Rehearsals get their own file.
    log_path = gt.DEFAULT_LOG
    if args.dry_run:
        log_path = log_path.with_suffix(log_path.suffix + ".dryrun")

    # THE LABEL IS WRITTEN HERE, BEFORE THE FAULT. Nothing below may edit it.
    gt.open_incident(
        incident_id=incident_id,
        fault=args.fault,
        instance_id=instance_id,
        window_start=window_start,
        window_end_planned=window_end_planned,
        operator=args.operator,
        dry_run=args.dry_run,
        path=log_path,
        details={
            # The role IS the answer the pilot scores against. An instance id
            # cannot be scored -- and R1 closes this account in February, after
            # which an id resolves to nothing while a role still reads.
            "node_role": role,
            # The graph as it stood when the fault was injected. Recorded rather
            # than re-derived later, because a label that depends on querying a
            # live account is a label with an expiry date.
            "topology": {r: n["depends_on"] for r, n in nodes.items()},
        },
    )
    print(f"label written: {incident_id} — {gt.FAULT_CLASSES[args.fault]} on {role}")

    outcome, notes = "completed", ""
    try:
        if args.fault == "F0":
            print(f"  control window: doing nothing for {seconds}s")
            if not args.dry_run:
                time.sleep(seconds)
        elif args.fault in ("F1", "F2"):
            burn_cpu(ssm, instance_id, seconds, args.dry_run)
        elif args.fault == "F3":
            stop_and_hold(ec2, instance_id, seconds, args.dry_run)
    except Exception as exc:  # noqa: BLE001 - the outcome is data, not a crash
        outcome, notes = "failed", f"{type(exc).__name__}: {exc}"
        print(f"  fault failed: {notes}", file=sys.stderr)
    finally:
        # Restore, always. A harness that leaves an instance stopped spends
        # money for nothing and corrupts the next window.
        details = {}
        if args.fault == "F3" and not args.dry_run:
            print("  restoring: starting the instance again")
            restored, tries, restored_at, errors = start_with_retries(ec2, instance_id)
            details["restore_attempts"] = tries
            details["service_restored_at"] = restored_at.isoformat() if restored_at else None
            if not restored:
                outcome = "failed"
                notes = (notes + " | " if notes else "") + "restore failed: " + "; ".join(errors)

        if not args.dry_run:
            try:
                details["instance_state_left"] = current_state(ec2, instance_id)
            except Exception as exc:  # noqa: BLE001
                details["instance_state_left"] = "unknown: %s" % exc

        gt.close_incident(
            incident_id=incident_id,
            window_end_actual=datetime.now(timezone.utc),
            outcome=outcome,
            notes=notes,
            path=log_path,
            details=details,
        )
        print(f"closed: {incident_id} -> {outcome}")

        # A failure that leaves infrastructure broken deserves more than a line
        # inside a JSON file nobody is watching.
        if details.get("instance_state_left") not in (None, "running"):
            print("", file=sys.stderr)
            print("!! THE PILOT TARGET WAS LEFT %s" % str(details["instance_state_left"]).upper(),
                  file=sys.stderr)
            print("!! %s is not running. The next injection will refuse to open a"
                  % instance_id, file=sys.stderr)
            print("!! window until it is. Start it by hand and confirm the state.",
                  file=sys.stderr)

    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
