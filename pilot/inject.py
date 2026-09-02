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


def resolve_target(ec2, instance_id: str | None) -> dict:
    """Find the pilot target and prove it is the pilot target."""
    filters = [{"Name": f"tag:{PILOT_TAG_KEY}", "Values": [PILOT_TAG_VALUE]}]
    kwargs = {"Filters": filters}
    if instance_id:
        kwargs["InstanceIds"] = [instance_id]

    reservations = ec2.describe_instances(**kwargs).get("Reservations", [])
    found = [i for r in reservations for i in r.get("Instances", [])
             if i["State"]["Name"] != "terminated"]

    if not found:
        raise InjectionError(
            f"No non-terminated instance carries {PILOT_TAG_KEY}={PILOT_TAG_VALUE}. "
            "Set pilot_enabled = true in terraform.tfvars and apply, or pass the "
            "instance id explicitly once it is tagged. This harness will not act on "
            "an untagged instance."
        )
    if len(found) > 1:
        ids = ", ".join(i["InstanceId"] for i in found)
        raise InjectionError(
            f"{len(found)} instances carry the pilot tag ({ids}). The pilot targets "
            "exactly one; refusing to guess."
        )
    return found[0]


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


def ensure_running(ec2, instance_id: str, dry_run: bool) -> None:
    state = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]["State"]["Name"]
    if state == "running":
        return
    if dry_run:
        print(f"[dry-run] would start {instance_id} (currently {state})")
        return
    print(f"  starting {instance_id} (currently {state})")
    ec2.start_instances(InstanceIds=[instance_id])
    wait_for_state(ec2, instance_id, "running")
    # CloudWatch needs a few minutes of a running instance before the window is
    # meaningful. Starting inside the window would put the boot in the metrics.
    print("  waiting 180s for metrics to settle before the window opens")
    time.sleep(180)


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
    ap.add_argument("--instance", default=None, help="instance id; resolved from the pilot tag if omitted")
    ap.add_argument("--injector-role-arn", default=None, help="role to assume; omit to use ambient credentials")
    ap.add_argument("--operator", default="alejandro", help="who ran it, recorded in the label")
    ap.add_argument("--dry-run", action="store_true", help="every step except the fault itself")
    args = ap.parse_args(argv)

    seconds = args.duration * 60
    incident_id = f"{args.fault}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"

    sess = session(args.injector_role_arn)
    ec2 = sess.client("ec2")
    ssm = sess.client("ssm")

    target = resolve_target(ec2, args.instance)
    instance_id = target["InstanceId"]
    print(f"target: {instance_id} ({target['InstanceType']}, {target['State']['Name']})")

    if args.fault in ("F1", "F2"):
        mode = assert_standard_credits(ec2, instance_id)
        print(f"credit mode: {mode}")

    # IA-52, criterion 5. A previous run whose restore failed leaves the target
    # stopped. Opening a window against a stopped instance would produce a
    # labelled incident whose signal is the LAST failure, not this one -- a
    # contaminated data point that looks exactly like a clean one.
    #
    # So the state is confirmed, not assumed, before any label is written.
    ensure_running(ec2, instance_id, args.dry_run)
    if not args.dry_run:
        state = current_state(ec2, instance_id)
        if state != "running":
            raise InjectionError(
                f"{instance_id} is '{state}', not running, and could not be started. "
                "Refusing to open a window: the incident would carry the previous "
                "failure's signal under this run's label. Start it by hand, confirm "
                "it is running, and try again."
            )

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
    )
    print(f"label written: {incident_id} ({gt.FAULT_CLASSES[args.fault]})")

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
