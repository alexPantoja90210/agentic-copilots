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
import incident_builder as ib

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


# Burstable credit economics, for the instance types this pilot actually uses.
# An unknown type is REFUSED rather than guessed: guessing a drain rate is the
# defect this table exists to prevent, and a wrong guess produces a wrong label.
BURSTABLE = {
    "t3.micro": {"vcpus": 2, "earn_per_hour": 12.0},
    "t3.small": {"vcpus": 2, "earn_per_hour": 24.0},
}

# The balance an F1 window must still have left when it ends (IA-58).
#
# NOT a copy of a collector threshold. IA-51 deleted the constant 30 that used
# to live in `collector.THRESHOLDS` as ("lt", 30.0, ...), because a level cannot
# tell an exhausted instance from one too young to have accrued. Reserving 30
# here would be a margin against a rule this project removed.
#
# And "the balance is falling" separates nothing: it falls during EVERY F1, at
# 108 credits/hour under full load on a t3.micro. What separates F1 from F2 is
# THROTTLING -- a balance that reaches the floor forces CPU back down to the
# 10% baseline on its own, and the window stops being a CPU fault while the log
# still says it is one.
#
# 10 is a judgement call, stated so it can be argued with: enough headroom that
# five-minute publishing granularity, or a window that overruns a little,
# cannot land on the floor.
F1_END_BALANCE_FLOOR = 10.0


def credit_cost(instance_type: str, minutes: float) -> float:
    """
    Credits a full-load window of this length consumes, net of what it earns.

    Computed from the window and the vCPU count rather than fixed, because a
    25-minute window and a 10-minute one do not need the same balance and a
    constant would be wrong for one of them.
    """
    spec = BURSTABLE.get(instance_type)
    if spec is None:
        raise InjectionError(
            "%s is not in the burstable table, so the credit drain of a CPU "
            "fault against it is unknown. Add it with its vCPU count and earn "
            "rate, or run the fault elsewhere. Guessing the rate is how an F1 "
            "becomes an F2 wearing an F1 label." % instance_type)
    # 100%% of one vCPU costs one credit per minute, by definition.
    drain_per_minute = spec["vcpus"] - spec["earn_per_hour"] / 60.0
    return drain_per_minute * minutes


def latest_credit_balance(cloudwatch, instance_id: str, now=None):
    """
    The most recent CPUCreditBalance datapoint, as (value, observed_at).

    Published every five minutes, so this can be up to five minutes stale. On an
    idle instance the balance only RISES in that gap, which makes a stale
    reading conservative in the safe direction: it can refuse a window that
    would have been fine, never permit one that would not.
    """
    now = now or datetime.now(timezone.utc)
    got = cloudwatch.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName="CPUCreditBalance",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=now - timedelta(hours=1),
        EndTime=now,
        Period=300,
        Statistics=["Average"],
    )
    points = sorted(got.get("Datapoints", []), key=lambda point: point["Timestamp"])
    if not points:
        return None, None
    return points[-1]["Average"], points[-1]["Timestamp"]


def assert_credit_headroom(cloudwatch, instance_id: str, instance_type: str,
                           minutes: float, now=None) -> dict:
    """
    Refuse an F1 whose window would run the balance into throttling.

    F2 is deliberately exempt and never reaches here: exhaustion is what F2 is
    FOR, and applying this guard to it would refuse the fault for doing its job.
    """
    required = credit_cost(instance_type, minutes) + F1_END_BALANCE_FLOOR
    balance, observed_at = latest_credit_balance(cloudwatch, instance_id, now=now)

    if balance is None:
        raise InjectionError(
            "no CPUCreditBalance datapoint in the last hour for this instance. "
            "The balance is unknown, and unknown is not the same as sufficient. "
            "Wait for the instance to publish (5-minute granularity) and retry.")

    if balance < required:
        deficit = required - balance
        earn_per_hour = BURSTABLE[instance_type]["earn_per_hour"]
        wait_hours = deficit / earn_per_hour
        raise InjectionError(
            "credit balance %.0f is below the %.0f this %.0f-minute CPU fault "
            "needs (%.0f consumed + %.0f left at the end, so the instance is "
            "never throttled). Short by %.0f credits: about %.1f hour(s) of "
            "idle accrual at %.0f/hour. Reading taken at %s.\n"
            "Refusing, because a window that exhausts its credits stops being a "
            "CPU fault part-way through -- the instance is throttled to its "
            "baseline and CPU comes down on its own -- while the log would still "
            "call it F1. A false label is worse than a missing incident."
            % (balance, required, minutes, credit_cost(instance_type, minutes),
               F1_END_BALANCE_FLOOR, deficit, wait_hours, earn_per_hour,
               observed_at.isoformat() if observed_at else "unknown"))

    return {"balance": balance, "required": required,
            "observed_at": observed_at.isoformat() if observed_at else None,
            "instance_type": instance_type}


def instance_type_of(ec2, instance_id: str) -> str:
    got = ec2.describe_instances(InstanceIds=[instance_id])
    return got["Reservations"][0]["Instances"][0]["InstanceType"]


RESTORE_ATTEMPTS = 3
RESTORE_BACKOFF_S = (15, 45, 90)


POLL_INTERVAL_S = 10


class StateWaitTimeout(InjectionError):
    """A wait that ran out of time, carrying what was actually observed.

    On 3 Sep 2026 three restore attempts printed the same sentence:

        i-0460bd1bda7ca477f did not reach state 'running' within 300s.

    That sentence names the state that did NOT happen. Absence is not an
    observation, and three very different faults render identically under it:
    AWS refused the start and the instance never left `stopped`; the start was
    accepted and the instance hung in `pending`; the instance was still
    `stopping` when the start was issued. Nothing in the record said which.

    This exception carries the observation instead: the states actually seen,
    in order, and — when the instance is sitting in `stopped` — the reason AWS
    gives for it being there.
    """

    def __init__(self, instance_id: str, wanted: str, observation: dict):
        self.instance_id = instance_id
        self.wanted = wanted
        self.observation = observation
        super().__init__(observation["message"])


def observe_state(ec2, instance_id: str) -> tuple[str, str]:
    """Return (state, StateTransitionReason) exactly as the API reports them.

    `StateTransitionReason` is the field that distinguishes "stopped because
    somebody stopped it" from "stopped because Server.InternalError", and it is
    the single most useful string in a failed restore. It is absent from a
    running instance and from most fakes, hence the `.get`.
    """
    got = ec2.describe_instances(InstanceIds=[instance_id])
    inst = got["Reservations"][0]["Instances"][0]
    return inst["State"]["Name"], (inst.get("StateTransitionReason") or "").strip()


def current_state(ec2, instance_id: str) -> str:
    return observe_state(ec2, instance_id)[0]


def start_with_retries(ec2, instance_id: str, attempts: int = RESTORE_ATTEMPTS,
                       sleeper=time.sleep, timeout_s: int = 300, clock=time.time):
    """
    Start the instance, and mean it.

    On 1 Sep 2026 the first real F3 left the pilot target stopped: AWS answered
    the start with `Server.InternalError`, the harness tried exactly once, gave
    up, and recorded the failure in a JSON file nobody was watching. A manual
    retry minutes later succeeded on the first attempt, which is the ordinary
    behaviour of that error — it is transient and usually lands on different
    hardware next time.

    One attempt is not a serious effort to recover from a transient failure.

    The retry POLICY is deliberately untouched by IA-56. Retrying is not what
    failed on 3 Sep; understanding is. What changed is that every attempt now
    returns an observation — the states seen, in order — instead of a string
    saying which state did not arrive. Three identical sentences told us
    nothing three times.

    Returns (running, attempts_made, restored_at, observations), where each
    observation is a JSON-serialisable dict destined for the incident's close
    record. stderr is a place to notice a failure, not a place to keep it.
    """
    observations: list[dict] = []
    for attempt in range(1, attempts + 1):
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            obs = wait_for_state(ec2, instance_id, "running",
                                 timeout_s=timeout_s, sleeper=sleeper, clock=clock)
            obs["attempt"] = attempt
            observations.append(obs)
            return True, attempt, datetime.now(timezone.utc), observations
        except StateWaitTimeout as exc:
            obs = dict(exc.observation)
            obs["attempt"] = attempt
            obs["error_type"] = type(exc).__name__
            observations.append(obs)
        except Exception as exc:  # noqa: BLE001 - every failure is data
            # StartInstances itself refused. The state is still worth having:
            # a refusal against a `stopping` instance and a refusal against a
            # `stopped` one are different problems.
            try:
                last, reason = observe_state(ec2, instance_id)
            except Exception:  # noqa: BLE001 - an unobservable instance is data too
                last, reason = "unobservable", ""
            observations.append({
                "outcome": "api_error",
                "waiting_for": "running",
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "last_state": last,
                "states_seen": [last],
                "progressed": False,
                "state_transition_reason": reason,
                "polls": 0,
                "waited_s": 0.0,
                "message": "%s: %s (instance is '%s'%s)" % (
                    type(exc).__name__, exc, last,
                    "; StateTransitionReason: " + reason if last == "stopped" and reason else "",
                ),
            })
        print("  restore attempt %d failed: %s" % (attempt, observations[-1]["message"]),
              file=sys.stderr)
        if attempt < attempts:
            pause = RESTORE_BACKOFF_S[min(attempt - 1, len(RESTORE_BACKOFF_S) - 1)]
            print("  retrying in %ds" % pause, file=sys.stderr)
            sleeper(pause)
    return False, attempts, None, observations


def wait_for_state(ec2, instance_id: str, state: str, timeout_s: int = 300,
                   sleeper=time.sleep, clock=time.time) -> dict:
    """Wait for `state`, and report what was seen rather than what was missed.

    Returns an observation dict on success and raises StateWaitTimeout — which
    carries the same shape — on timeout. Either way the caller can answer the
    question the old harness could not: what was the instance actually doing?

    The deadline is checked AFTER the first poll, so an observation always
    exists. A wait that reports nothing at all is the defect this fixes.
    """
    started = clock()
    deadline = started + timeout_s
    seen: list[str] = []
    reason = ""
    polls = 0
    while True:
        now, reason = observe_state(ec2, instance_id)
        polls += 1
        if not seen or seen[-1] != now:
            seen.append(now)
        if now == state:
            return _observation(instance_id, state, seen, reason, polls,
                                round(clock() - started, 1), reached=True)
        if clock() >= deadline:
            break
        sleeper(POLL_INTERVAL_S)

    raise StateWaitTimeout(
        instance_id, state,
        _observation(instance_id, state, seen, reason, polls,
                     round(clock() - started, 1), reached=False),
    )


def _observation(instance_id: str, wanted: str, seen: list[str], reason: str,
                 polls: int, waited_s: float, reached: bool) -> dict:
    """Render one wait as structured data plus one sentence a human can act on.

    Three outcomes, three different sentences — which is the whole point:

      reached    the path is named, so a slow boot reads as a slow boot;
      no motion  the instance never left its starting state, so the start was
                 refused or never took effect — retrying is plausible;
      stalled    the instance began changing state and did not finish, so it is
                 AWS that is stuck, and a fourth attempt would not help.
    """
    path = " -> ".join(seen)
    last = seen[-1]
    progressed = len(seen) > 1
    if reached:
        outcome = "reached"
        message = (f"{instance_id} reached '{wanted}' after {waited_s}s "
                   f"({polls} polls) via {path}.")
    elif not progressed:
        outcome = "no_motion"
        message = (f"{instance_id} never left '{last}' while waiting for "
                   f"'{wanted}': {waited_s}s, {polls} polls, no state change at all.")
    else:
        outcome = "stalled"
        message = (f"{instance_id} went {path} while waiting for '{wanted}': it "
                   f"began changing state and did not finish within {waited_s}s "
                   f"({polls} polls).")
    # The reason AWS gives for an instance being stopped is the difference
    # between "the operator stopped it" and "Server.InternalError". Report it
    # whenever the instance is sitting in `stopped`, which is the only state
    # for which it is both populated and meaningful.
    if last == "stopped" and reason:
        message += f" StateTransitionReason: {reason}"
    return {
        "outcome": outcome,
        "waiting_for": wanted,
        "last_state": last,
        "states_seen": seen,
        "progressed": progressed,
        "state_transition_reason": reason,
        "polls": polls,
        "waited_s": waited_s,
        "message": message,
    }


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

    credit_check = None
    if args.fault in ("F1", "F2"):
        mode = assert_standard_credits(ec2, instance_id)
        print(f"credit mode: {mode}")

    # F1 only. F2 wants exhaustion; guarding it against exhaustion would refuse
    # the fault for doing exactly what it exists to do.
    if args.fault == "F1" and not args.dry_run:
        credit_check = assert_credit_headroom(
            sess.client("cloudwatch"), instance_id,
            instance_type_of(ec2, instance_id), args.duration)
        print("credit balance: %.0f (needs %.0f for %d minutes), read at %s"
              % (credit_check["balance"], credit_check["required"],
                 args.duration, credit_check["observed_at"]))

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
            # What the balance was when the window opened. Evidence that this
            # F1 had the headroom to stay an F1 -- checkable later against the
            # metrics rather than taken on trust.
            "credit_check": credit_check,
        },
        # IA-61. The injector must refuse a window whose CONTEXT would contain
        # a neighbour's fault, not only one whose window overlaps. The numbers
        # come from incident_builder, which is what actually pads the query --
        # restating them here would create a second source that drifts.
        context_pre_minutes=ib.PRE_MINUTES,
        context_post_minutes=ib.POST_MINUTES,
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
            restored, tries, restored_at, observations = start_with_retries(ec2, instance_id)
            details["restore_attempts"] = tries
            details["service_restored_at"] = restored_at.isoformat() if restored_at else None
            # Criterion 2 of IA-56: the observations go into the record as
            # structured data. A failure that exists only in a terminal
            # scrollback is a failure that cannot be scored later.
            details["restore_observations"] = observations
            if not restored:
                outcome = "failed"
                notes = (notes + " | " if notes else "") + "restore failed: " + "; ".join(
                    "attempt %s (%s): %s" % (o.get("attempt"), o.get("outcome"), o["message"])
                    for o in observations
                )

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
