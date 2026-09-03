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


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def no_sleep(_seconds):
    return None


def run() -> int:
    # ---- the defect's own scenario: two transient failures, then success ----
    ec2 = FakeEC2(start_failures=2)
    ok, tries, restored_at, errors = inject.start_with_retries(
        ec2, "i-0test", sleeper=no_sleep)
    check("a transient failure is retried until it works", ok,
          "gave up after %d attempt(s): %s" % (tries, errors))
    check("it took more than one attempt", tries == 3,
          "attempts=%d — if this is 1 the fixture is not exercising the retry" % tries)
    check("the recovery is timestamped", restored_at is not None)
    check("the failed attempts are kept, not swallowed", len(errors) == 2,
          "errors=%s" % errors)
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
          "errors=%d" % len(errors))
    check("the instance is left stopped, and says so", dead.state == "stopped")

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
