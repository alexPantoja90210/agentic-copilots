"""
Selftest for the topology reader. No AWS.

The graph is the pilot's independent variable: if it is wrong, arm B is wrong
and every result built on it is wrong in a way no later check would notice. So
the checks here are mostly NEGATIVE — the shapes that must be refused.
"""

from __future__ import annotations

import chain_topology as ct

PASS, FAIL = "PASS", "FAIL"
results = []


class FakeEC2:
    def __init__(self, spec):
        """spec: role -> (instance_id, depends_on, state)"""
        self.spec = spec

    def describe_instances(self, Filters=None):  # noqa: N803
        instances = []
        for role, (iid, upstream, state) in self.spec.items():
            tags = [{"Key": "Pilot", "Value": "IA-45"}]
            if role is not None:
                tags.append({"Key": "ChainRole", "Value": role})
            if upstream is not None:
                tags.append({"Key": "DependsOn", "Value": upstream})
            instances.append({
                "InstanceId": iid,
                "State": {"Name": state},
                "PrivateIpAddress": "10.0.0.%d" % (len(instances) + 1),
                "Tags": tags,
            })
        return {"Reservations": [{"Instances": instances}]}


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def expect_error(name, fn, needle):
    try:
        fn()
    except ct.TopologyError as exc:
        check(name, needle.lower() in str(exc).lower(), f"message lacks {needle!r}: {exc}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
    else:
        check(name, False, "no error raised — the guard did not fire")


HEALTHY = {
    "db":  ("i-0db", None,  "running"),
    "app": ("i-0app", "db",  "running"),
    "web": ("i-0web", "app", "running"),
}


def run() -> int:
    nodes = ct.discover(FakeEC2(HEALTHY))
    check("the declared chain is read back whole", set(nodes) == {"db", "app", "web"})
    check("the path from web reaches the tail",
          ct.chain_from("web", nodes) == ["web", "app", "db"],
          str(ct.chain_from("web", nodes)))
    check("stopping db would show symptoms on app and web",
          ct.dependents_of("db", nodes) == ["app", "web"],
          str(ct.dependents_of("db", nodes)))
    check("stopping app would show a symptom on web only — NOT on db",
          ct.dependents_of("app", nodes) == ["web"],
          str(ct.dependents_of("app", nodes)))
    check("stopping web affects nothing downstream",
          ct.dependents_of("web", nodes) == [])

    # A stopped node stays in the graph.
    stopped = dict(HEALTHY, db=("i-0db", None, "stopped"))
    nodes_stopped = ct.discover(FakeEC2(stopped))
    check("a stopped node is still part of the graph",
          set(nodes_stopped) == {"db", "app", "web"}
          and nodes_stopped["db"]["state"] == "stopped",
          "a node vanishing while off would let the fault edit the topology")

    # The agent's view carries no environment identifiers.
    rendered = ct.describe(nodes)
    check("the rendered graph leaks no instance ids",
          not any(node["instance_id"] in rendered for node in nodes.values()),
          rendered)
    check("the rendered graph states every edge",
          "web depends on app" in rendered and "app depends on db" in rendered
          and "db depends on nothing" in rendered, rendered)

    # ---- negatives ----
    expect_error("an edge pointing at a node that does not exist is refused",
                 lambda: ct.discover(FakeEC2({
                     "app": ("i-0app", "ghost", "running"),
                     "db": ("i-0db", None, "running")})),
                 "not a node")

    expect_error("a cycle is refused",
                 lambda: ct.discover(FakeEC2({
                     "app": ("i-0app", "web", "running"),
                     "web": ("i-0web", "app", "running")})),
                 "cycle")

    expect_error("two nodes claiming the same role is refused",
                 lambda: ct.discover(_DuplicateRole()),
                 "two instances claim")

    expect_error("a graph with no tail is refused",
                 lambda: ct.discover(FakeEC2({
                     "app": ("i-0app", "db", "running"),
                     "db": ("i-0db", "app", "running")})),
                 "cycle")

    expect_error("two tails is refused",
                 lambda: ct.discover(FakeEC2({
                     "app": ("i-0app", None, "running"),
                     "db": ("i-0db", None, "running")})),
                 "exactly one node")

    expect_error("no tagged instance at all is refused",
                 lambda: ct.discover(FakeEC2({})),
                 "no instance carries")

    width = max(len(n) for n, _, _ in results)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


class _DuplicateRole:
    def describe_instances(self, Filters=None):  # noqa: N803
        tags = [{"Key": "Pilot", "Value": "IA-45"}, {"Key": "ChainRole", "Value": "app"}]
        return {"Reservations": [{"Instances": [
            {"InstanceId": "i-0one", "State": {"Name": "running"}, "Tags": tags},
            {"InstanceId": "i-0two", "State": {"Name": "running"}, "Tags": tags},
        ]}]}


if __name__ == "__main__":
    raise SystemExit(run())
