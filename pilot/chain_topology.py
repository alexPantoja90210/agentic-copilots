"""
The dependency graph, read back from the account.

IA-55, criterion 4: the topology handed to the agent must be GENERATED from the
declared graph, never typed into a prompt. So the edge lives on the instance as
a `DependsOn` tag, Terraform writes it, and this module reads it back. The graph
the agent reasons over and the graph that exists are the same object; there is
no second copy to drift.

That matters more than it sounds. If the graph were maintained separately, the
pilot could report that the agent traced a dependency correctly while the
dependency it traced had been edited out of the infrastructure weeks earlier --
a correct answer to a question about a world that no longer exists.
"""

from __future__ import annotations

PILOT_TAG_KEY = "Pilot"
PILOT_TAG_VALUE = "IA-45"


class TopologyError(Exception):
    pass


def _tags(instance) -> dict:
    return {t["Key"]: t["Value"] for t in instance.get("Tags", [])}


def discover(ec2, pilot_tag_value: str = PILOT_TAG_VALUE) -> dict:
    """
    Every pilot node, its role, its state and what it depends on.

    Instances that are stopped are INCLUDED. A node missing from the graph
    because it happens to be off would quietly turn a two-hop chain into a
    one-hop one, and the agent would be judged against a topology that was
    edited by the very fault under investigation.
    """
    reservations = ec2.describe_instances(Filters=[
        {"Name": f"tag:{PILOT_TAG_KEY}", "Values": [pilot_tag_value]},
    ]).get("Reservations", [])

    nodes = {}
    for reservation in reservations:
        for instance in reservation.get("Instances", []):
            if instance["State"]["Name"] == "terminated":
                continue
            tags = _tags(instance)
            role = tags.get("ChainRole")
            if not role:
                continue
            if role in nodes:
                raise TopologyError(
                    f"two instances claim the role '{role}': {nodes[role]['instance_id']} "
                    f"and {instance['InstanceId']}. The graph would be ambiguous."
                )
            nodes[role] = {
                "role": role,
                "instance_id": instance["InstanceId"],
                "private_ip": instance.get("PrivateIpAddress"),
                "state": instance["State"]["Name"],
                "depends_on": tags.get("DependsOn"),
            }

    if not nodes:
        raise TopologyError(
            f"no instance carries {PILOT_TAG_KEY}={pilot_tag_value} together with a "
            "ChainRole tag. Enable the chain in Terraform and apply before "
            "expecting a graph."
        )

    _validate(nodes)
    return nodes


def _validate(nodes: dict) -> None:
    """A graph that does not hold together is not context; it is noise."""
    for role, node in nodes.items():
        upstream = node["depends_on"]
        if upstream is None:
            continue
        if upstream not in nodes:
            raise TopologyError(
                f"'{role}' declares a dependency on '{upstream}', which is not a "
                "node in this graph. An edge pointing at nothing is the same "
                "defect as an alert about a service nobody declared."
            )

    # A cycle would make "the root cause is upstream" meaningless.
    for start in nodes:
        seen, cursor = [], start
        while cursor is not None:
            if cursor in seen:
                raise TopologyError(
                    "dependency cycle: " + " -> ".join(seen + [cursor]))
            seen.append(cursor)
            cursor = nodes[cursor]["depends_on"]

    tails = [r for r, n in nodes.items() if n["depends_on"] is None]
    if len(tails) != 1:
        raise TopologyError(
            "expected exactly one node that depends on nothing, found %d: %s"
            % (len(tails), ", ".join(sorted(tails)) or "none"))


def chain_from(role: str, nodes: dict) -> list[str]:
    """The path from a node to the tail: who this node ultimately rests on."""
    if role not in nodes:
        raise TopologyError(f"unknown role '{role}'")
    path, cursor = [], role
    while cursor is not None:
        path.append(cursor)
        cursor = nodes[cursor]["depends_on"]
    return path


def dependents_of(role: str, nodes: dict) -> list[str]:
    """Everything that would show a symptom if this node failed."""
    affected = []
    for candidate in nodes:
        if candidate != role and role in chain_from(candidate, nodes)[1:]:
            affected.append(candidate)
    return sorted(affected)


def describe(nodes: dict) -> str:
    """
    The graph as the agent will see it. Roles and edges only.

    Instance ids are deliberately left out: they are environment identifiers,
    they add nothing a reader can reason with, and the pilot's write-up has to
    be publishable.
    """
    lines = ["Service dependency graph:"]
    for role in sorted(nodes):
        upstream = nodes[role]["depends_on"]
        if upstream:
            lines.append(f"  {role} depends on {upstream}")
        else:
            lines.append(f"  {role} depends on nothing")
    return "\n".join(lines)
