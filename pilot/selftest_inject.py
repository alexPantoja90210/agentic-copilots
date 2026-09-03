"""
Selftest for the injection harness's restore path.

The first real F3, on 1 Sep 2026, left the pilot target stopped: AWS answered
the start with Server.InternalError, the harness tried once, gave up, and wrote
the failure into a JSON file nobody was watching. A manual retry succeeded
immediately.

So the property under test is not "starting works". It is "a transient failure
is retried, and a permanent one is reported as failure rather than papered
over". Both directions, because a retry that has never been observed retrying
is not a retry.

No AWS, no credentials, no cost: the client is a fake whose failures are
scripted.
"""

from __future__ import annotations

import json

import inject

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


class FakeEC2:
    """A start that fails `start_failures` times before it works."""

    def __init__(self, start_failures=0, state="stopped", error="Server.InternalError"):
        self.remaining_failures = start_failures
        self.state = state
        self.error = error
        self.start_calls = 0

    def start_instances(self, InstanceIds):  # noqa: N803 - boto3's spelling
        self.start_calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("An error occurred (%s) when calling StartInstances" % self.error)
        self.state = "running"
        return {"StartingInstances": [{"InstanceId": InstanceIds[0]}]}

    def stop_instances(self, InstanceIds):  # noqa: N803
        self.state = "stopped"
        return {"StoppingInstances": [{"InstanceId": InstanceIds[0]}]}

    def describe_instances(self, InstanceIds):  # noqa: N803
        return {"Reservations": [{"Instances": [
            {"InstanceId": InstanceIds[0], "State": {"Name": self.state}}
        ]}]}


class _FakeChain:
    """Tagged instances as describe_instances returns them."""

    def __init__(self, spec):
        self.spec = spec

    def describe_instances(self, Filters=None, **kwargs):  # noqa: N803
        instances = []
        for role, (iid, upstream, state) in self.spec.items():
            tags = [{"Key": "Pilot", "Value": "IA-45"},
                    {"Key": "ChainRole", "Value": role}]
            if upstream:
                tags.append({"Key": "DependsOn", "Value": upstream})
            instances.append({"InstanceId": iid, "State": {"Name": state},
                              "PrivateIpAddress": "10.0.0.1", "Tags": tags})
        return {"Reservations": [{"Instances": instances}]}


class ScriptedEC2:
    """describe_instances replays a scripted sequence of states, one per poll.

    The last entry repeats forever, so "never leaves stopped", "hangs in
    pending" and "boots on the fifth poll" are the same fake with different
    scripts. StartInstances always succeeds here: the point of these three is
    what happens AFTER the API says yes, which is precisely the region the old
    message could not describe.
    """

    def __init__(self, states, reason=""):
        self.states = list(states)
        self.reason = reason
        self.polls = 0
        self.start_calls = 0

    def start_instances(self, InstanceIds):  # noqa: N803 - boto3's spelling
        self.start_calls += 1
        return {"StartingInstances": [{"InstanceId": InstanceIds[0]}]}

    def describe_instances(self, InstanceIds):  # noqa: N803
        i = min(self.polls, len(self.states) - 1)
        self.polls += 1
        inst = {"InstanceId": InstanceIds[0], "State": {"Name": self.states[i]}}
        if self.reason:
            inst["StateTransitionReason"] = self.reason
        return {"Reservations": [{"Instances": [inst]}]}


class FakeClock:
    """Time that only moves when something sleeps.

    Lets a 300-second wait be tested in microseconds, and — more importantly —
    makes the number of polls deterministic, so the test can assert on it.
    """

    def __init__(self, start=1_000_000.0):
        self.t = start

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def pre_ia56_restore(client) -> str:
    """Reproduce the 3 Sep pipeline and return the line it wrote to stderr.

    Not a hardcoded string: the old `wait_for_state` body AND the old rendering
    in `start_with_retries`, both verbatim, driven by the same fake clients the
    new code is driven by. The only edit is an injectable clock so 300 seconds
    pass in microseconds.

    The pair matters. Mutating the old wait alone does NOT go red, because the
    new caller re-observes the instance in its exception handler and would
    report the state anyway. That masking was found by running the mutation and
    watching it come back green — the defect being reproduced here lives in the
    two functions together, so the mutation has to restore both.
    """
    clock = FakeClock()

    def old_wait(ec2, instance_id, state, timeout_s=300):
        deadline = clock.time() + timeout_s
        while clock.time() < deadline:
            got = ec2.describe_instances(InstanceIds=[instance_id])
            now = got["Reservations"][0]["Instances"][0]["State"]["Name"]
            if now == state:
                return
            clock.sleep(10)
        raise inject.InjectionError(
            "%s did not reach state '%s' within %ds." % (instance_id, state, timeout_s))

    try:
        client.start_instances(InstanceIds=["i-0test"])
        old_wait(client, "i-0test", "running")
    except Exception as exc:  # the old handler, spelled as it was
        return "attempt 1: %s: %s" % (type(exc).__name__, exc)
    return "attempt 1: restored"


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def no_sleep(_seconds):
    return None


def run() -> int:
    # ---- the defect's own scenario: two transient failures, then success ----
    ec2 = FakeEC2(start_failures=2)
    ok, tries, restored_at, observations = inject.start_with_retries(
        ec2, "i-0test", sleeper=no_sleep)
    failures = [o for o in observations if o["outcome"] != "reached"]
    check("a transient failure is retried until it works", ok,
          "gave up after %d attempt(s): %s" % (tries, observations))
    check("it took more than one attempt", tries == 3,
          "attempts=%d — if this is 1 the fixture is not exercising the retry" % tries)
    check("the recovery is timestamped", restored_at is not None)
    check("the failed attempts are kept, not swallowed", len(failures) == 2,
          "failures=%s" % failures)
    # IA-56 widened this list from "the errors" to "every attempt". The attempt
    # that WORKED is evidence too: it is what says the fault was transient, and
    # on 3 Sep its absence is precisely what left us guessing.
    check("the attempt that succeeded is recorded as well",
          len(observations) == 3 and observations[-1]["outcome"] == "reached",
          str([o["outcome"] for o in observations]))
    check("the instance really is running afterwards", ec2.state == "running")

    # ---- NEGATIVE: a permanent failure must stay a failure ----
    dead = FakeEC2(start_failures=99)
    ok, tries, restored_at, errors = inject.start_with_retries(
        dead, "i-0test", sleeper=no_sleep)
    check("a permanent failure is reported as failure", not ok)
    check("it stopped at the configured number of attempts",
          tries == inject.RESTORE_ATTEMPTS,
          "attempts=%d, expected %d" % (tries, inject.RESTORE_ATTEMPTS))
    check("no recovery time is invented when there was no recovery",
          restored_at is None)
    check("every attempt is recorded", len(errors) == inject.RESTORE_ATTEMPTS,
          "attempts recorded=%d" % len(errors))
    check("the instance is left stopped, and says so", dead.state == "stopped")

    # ---- IA-56: three failures that used to render as one sentence ----
    # Criterion 4. Each client is driven through the real start_with_retries
    # with attempts=1, so what is under test is the message the operator sees.

    # 1. AWS accepted the start and nothing happened: the instance never left
    #    `stopped`. This is the 3 Sep case, and the reason is the evidence.
    clock = FakeClock()
    inert = ScriptedEC2(["stopped"], reason="Server.InternalError")
    ok, _, _, obs = inject.start_with_retries(
        inert, "i-0test", attempts=1, sleeper=clock.sleep, clock=clock.time)
    stopped_obs = obs[0]
    check("an instance that never moves is reported as not moving",
          not ok and stopped_obs["outcome"] == "no_motion", stopped_obs["message"])
    check("the message names the state it WAS in",
          "never left 'stopped'" in stopped_obs["message"], stopped_obs["message"])
    check("and the reason AWS gives for it being there",
          "Server.InternalError" in stopped_obs["message"], stopped_obs["message"])
    check("no progress is claimed", stopped_obs["progressed"] is False)

    # 2. The start took effect and then stalled: `pending` forever. Same old
    #    sentence, completely different fault — a fourth retry cannot help.
    clock = FakeClock()
    hung = ScriptedEC2(["stopped", "pending"])
    ok, _, _, obs = inject.start_with_retries(
        hung, "i-0test", attempts=1, sleeper=clock.sleep, clock=clock.time)
    pending_obs = obs[0]
    check("an instance stuck mid-transition is reported as stalled, not inert",
          not ok and pending_obs["outcome"] == "stalled", pending_obs["message"])
    check("the message shows the path it took",
          "stopped -> pending" in pending_obs["message"], pending_obs["message"])
    check("progress is recorded, because there was some",
          pending_obs["progressed"] is True and pending_obs["last_state"] == "pending")

    # 3. A slow boot that arrives before the deadline: success, and the record
    #    still says how long and by which path.
    clock = FakeClock()
    slow = ScriptedEC2(["stopped", "pending", "pending", "pending", "running"])
    ok, tries, restored_at, obs = inject.start_with_retries(
        slow, "i-0test", attempts=1, sleeper=clock.sleep, clock=clock.time)
    late_obs = obs[0]
    check("a late transition is a success, not a timeout",
          ok and late_obs["outcome"] == "reached", late_obs["message"])
    check("the successful path is recorded too",
          late_obs["states_seen"] == ["stopped", "pending", "running"],
          str(late_obs["states_seen"]))
    check("and the time it took", late_obs["waited_s"] == 40.0, str(late_obs["waited_s"]))
    check("a success is timestamped", restored_at is not None and tries == 1)

    # ---- the mutation: the old message against the same three clients ----
    # This is the RED. Under old_message all three observations collapse to one
    # sentence; under the new one they are three. If a future change makes the
    # first check pass by making the new messages equal too, it has reverted
    # IA-56.
    messages = {stopped_obs["message"], pending_obs["message"], late_obs["message"]}
    old = {pre_ia56_restore(ScriptedEC2(["stopped"], reason="Server.InternalError")),
           pre_ia56_restore(ScriptedEC2(["stopped", "pending"]))}
    check("RED: the pre-IA-56 pipeline rendered both failures as one sentence",
          len(old) == 1, "old=%s" % old)
    check("and that sentence never named the state the instance was in",
          not any("stopped" in m or "pending" in m for m in old), str(old))
    check("the new messages are three distinct sentences", len(messages) == 3,
          "\n".join(sorted(messages)))
    check("the stuck-in-pending case is no longer readable as 'stopped'",
          "stopped" not in pending_obs["last_state"])
    check("no message says only what did not happen",
          not any("did not reach state" in m for m in messages))

    # ---- criterion 2: the observations must survive into the close record ----
    clock = FakeClock()
    dead_hung = ScriptedEC2(["stopped", "pending"])
    _, _, _, record = inject.start_with_retries(
        dead_hung, "i-0test", sleeper=clock.sleep, clock=clock.time)
    check("every attempt is carried, not just the last",
          len(record) == inject.RESTORE_ATTEMPTS, "entries=%d" % len(record))
    required = {"outcome", "last_state", "states_seen", "progressed",
                "state_transition_reason", "polls", "waited_s", "message", "attempt"}
    check("each entry is structured, not a sentence",
          all(required <= set(e) for e in record),
          str([sorted(required - set(e)) for e in record]))
    try:
        json.dumps(record)
        serialisable = True
    except TypeError as exc:
        serialisable = False
        detail = str(exc)
    check("and JSON-serialisable, because ground_truth writes JSONL",
          serialisable, locals().get("detail", ""))

    # A StartInstances refusal still yields an observation, so the api_error
    # path is not a hole where the state goes missing.
    refuser = FakeEC2(start_failures=99, state="stopped")
    _, _, _, api_obs = inject.start_with_retries(refuser, "i-0test", sleeper=no_sleep)
    check("a refused start also records the observed state",
          all(e["outcome"] == "api_error" and e["last_state"] == "stopped"
              for e in api_obs), str(api_obs[:1]))

    # ---- the pre-flight state check ----
    stopped = FakeEC2(start_failures=99, state="stopped")
    check("a target that cannot be started reports a non-running state",
          inject.current_state(stopped, "i-0test") == "stopped")

    running = FakeEC2(state="running")
    check("a healthy target reports running",
          inject.current_state(running, "i-0test") == "running")

    # ---- IA-55 aftermath: the target is a ROLE, not the only tagged instance ----
    # Before this, resolve_target refused whenever it found more than one tagged
    # instance. Correct with one node; a wall with three, and the harness could
    # not inject anything at all.
    chain = _FakeChain({
        "db":  ("i-0db",  None,  "running"),
        "app": ("i-0app", "db",  "running"),
        "web": ("i-0web", "app", "running"),
    })
    nodes, role = inject.resolve_target(chain, "app")
    check("a node is resolved by its role", role == "app" and nodes[role]["instance_id"] == "i-0app")
    check("and the graph comes back with it", set(nodes) == {"db", "app", "web"})

    try:
        inject.resolve_target(chain, None)
    except inject.InjectionError as exc:
        check("with several nodes, omitting the role is refused",
              "--node" in str(exc), str(exc))
    else:
        check("with several nodes, omitting the role is refused", False,
              "it guessed — which service fails is the experiment's variable")

    try:
        inject.resolve_target(chain, "cache")
    except inject.InjectionError as exc:
        check("an unknown role is refused, and the known ones are listed",
              "db" in str(exc) and "web" in str(exc), str(exc))
    else:
        check("an unknown role is refused", False, "no error")

    single = _FakeChain({"db": ("i-0db", None, "running")})
    _, only = inject.resolve_target(single, None)
    check("with exactly one node, the role may be omitted", only == "db")

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
