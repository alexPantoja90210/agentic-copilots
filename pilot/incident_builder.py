"""
One injected event -> the two inputs the experiment compares.

Both arms come out of one function so they cannot drift apart and produce a
difference that is an artefact of assembly rather than of context. Arm A's text
appears verbatim inside arm B; everything else in B is added, never rewritten.

Three rules this file exists to enforce
---------------------------------------

1. The title is GENERATED, never written per incident. A human writing "the web
   server cannot reach the database" has put the answer in arm A and there is no
   experiment left.

2. The title names the node a human would be paged about -- the most
   user-facing node affected -- which for a propagated fault is NOT the node
   that failed. That asymmetry is the whole point: symptom on `web`, cause on
   `db`, two hops apart. It is derived from the DECLARED graph plus the fault's
   location, so it is deterministic and involves no analysis of the metrics.

3. Arm B carries metrics for EVERY node, not only the symptomatic one. Showing
   just the node that hurts would remove the one thing a dependency graph is
   for: looking upstream and finding a machine that is quiet when it should not
   be. That would be a different and much easier question than the one the paper
   left open.

A limitation, stated here because it belongs in the write-up
------------------------------------------------------------
The template is uniform and deliberately neutral: it opens an incident on a node
at a time and asks for a root cause. It does not assert that an anomaly was
detected, because in the F0 control there was none, and a control that announces
itself by its wording is not a control.

The cost is that these titles carry less than the real tickets the paper used.
Arm A here has less to work with than a real on-call engineer would, so its
ABSOLUTE accuracy is not comparable to the paper's. The comparison BETWEEN arms
is unaffected -- both arms get the identical title -- and that is the comparison
this pilot makes.
"""

from __future__ import annotations

import chain_topology as ct

# Words that would hand over the answer. Checked against both arms' text.
FAULT_SYNONYMS = {
    "F0": ("no fault", "control", "nothing happened", "healthy"),
    "F1": ("cpu", "saturation", "saturated", "load", "busy", "utilisation", "utilization"),
    "F2": ("credit", "throttl", "burst"),
    "F3": ("stopped", "stop", "shut down", "shutdown", "unavailable", "outage", "down"),
}

TITLE = "Incident {incident_id}: service '{node}' under investigation"

SUMMARY = (
    "An incident was opened for service '{node}' covering {start} to {end} UTC. "
    "Determine the root cause: which service is responsible, and why. "
    "If the available information does not support attributing a cause, say so."
)


class BuilderError(Exception):
    pass


def symptomatic_node(fault_role: str, nodes: dict) -> str:
    """
    Who gets paged.

    The most user-facing node affected: the one nothing else depends on, walking
    down from the fault. For a local fault that is the failing node itself; for a
    propagated one it is somebody else entirely, which is exactly the case the
    graph is supposed to help with.

    Derived from the declared topology and the fault's location. No metric is
    consulted, so this cannot smuggle our own analysis into arm A.
    """
    if fault_role not in nodes:
        raise BuilderError(f"'{fault_role}' is not a node in this graph")

    affected = [fault_role] + ct.dependents_of(fault_role, nodes)
    # The user-facing end: the affected node that nothing else affected depends on.
    exposed = [role for role in affected
               if not any(role in ct.chain_from(other, nodes)[1:] for other in affected)]
    if len(exposed) != 1:
        raise BuilderError(
            "expected exactly one most-exposed node among %s, found %s. A graph "
            "that branches needs a rule for which branch gets paged, and "
            "inventing one silently would put a choice of mine inside the data."
            % (affected, exposed))
    return exposed[0]


BUCKET_SECONDS = 300


def _spans(stamp, moment):
    """Does the 5-minute bucket starting at `stamp` contain `moment`?"""
    from datetime import timedelta
    return stamp <= moment < stamp + timedelta(seconds=BUCKET_SECONDS)


def _format_series(points, window_start, window_end):
    """
    Datapoints as text, with the transitions marked and gaps named.

    A fault is only resolvable to its five-minute bucket, so the datapoint whose
    bucket CONTAINS the injection -- and the one containing its end -- are
    neither the healthy value nor the failed one. They are shown, because hiding
    them would be editing the evidence, and labelled, so nothing downstream reads
    them as a state.

    Only those two. The first version marked every datapoint inside the window,
    on every node, including nodes whose series never moved. A marker that
    appears on eight rows out of eight points at nothing.

    A series that ENDS before the window does is its own signal, and it gets a
    sentence rather than a silence -- absent data is not a value of zero, and
    that difference is the clue in half the fault catalogue.

    The wording is deliberately observational. An earlier version said the
    service "stopped reporting", and the leak detector refused the build: "stop"
    is F3's own vocabulary. It was right for a better reason than the one it
    checks. This function's job is to PRESENT data, not to narrate it -- saying a
    service stopped is already a conclusion, and the conclusion is what the agent
    is being asked for.
    """
    lines = []
    for stamp, value in points:
        mark = ""
        if _spans(stamp, window_start):
            mark = "   <- the fault begins inside this datapoint: a transition, not a state"
        elif _spans(stamp, window_end):
            mark = "   <- the window ends inside this datapoint: a transition, not a state"
        lines.append("    %s  %10.0f%s" % (stamp.strftime("%H:%M"), value, mark))

    if points:
        last = points[-1][0]
        if last < window_end:
            lines.append("    (no datapoints after %s. Absent data is not a value "
                         "of zero.)" % last.strftime("%H:%M"))
    return "\n".join(lines)


def build(entry: dict, nodes: dict, metrics: dict, fault_role: str) -> dict:
    """
    entry    one 'open' record from the ground-truth log
    nodes    chain_topology.discover() output
    metrics  {role: {metric_name: [(datetime, value), ...]}}
    """
    from datetime import datetime

    window_start = datetime.fromisoformat(entry["window_start"])
    window_end = datetime.fromisoformat(entry["window_end_planned"])
    node = symptomatic_node(fault_role, nodes)

    title = TITLE.format(incident_id=entry["incident_id"], node=node)
    summary = SUMMARY.format(node=node,
                             start=window_start.strftime("%H:%M"),
                             end=window_end.strftime("%H:%M"))
    arm_a = "%s\n\n%s" % (title, summary)

    # --- arm B: the same text, then context -------------------------------
    parts = [arm_a, "", ct.describe(nodes), "", "Service metrics, 5-minute datapoints:"]
    usable, notes = False, []

    for role in sorted(nodes):
        parts.append("")
        parts.append("  %s:" % role)
        node_metrics = metrics.get(role) or {}
        if not node_metrics:
            parts.append("    no datapoints in this window. A service that is not "
                         "reporting is not the same as a service reporting zero.")
            notes.append("%s: no metrics" % role)
            continue
        for metric_name in sorted(node_metrics):
            points = node_metrics[metric_name]
            if not points:
                parts.append("    %s: no datapoints" % metric_name)
                notes.append("%s/%s: no datapoints" % (role, metric_name))
                continue
            usable = True
            parts.append("    %s:" % metric_name)
            parts.append(_format_series(points, window_start, window_end))

    arm_b = "\n".join(parts)

    if arm_a not in arm_b:
        raise BuilderError(
            "arm A is not contained verbatim in arm B. The arms would differ in "
            "more than the added context, and any measured difference could no "
            "longer be attributed to context alone.")

    leaked = (leaks(entry["fault"], fault_role, node, arm_a)
              | leaks(entry["fault"], fault_role, node, arm_b))
    if leaked:
        raise BuilderError(
            "the prompt gives away the answer: %s. Refusing to build an incident "
            "whose input contains its own label." % ", ".join(sorted(leaked)))

    return {
        "incident_id": entry["incident_id"],
        "symptomatic_node": node,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "usable": usable,
        "notes": notes,
    }


def leaks(fault: str, fault_role: str, symptom_role: str, text: str) -> set:
    """
    Anything in the text that would hand over the answer.

    Both halves matter: the fault CLASS ("stopped", "cpu") and the root-cause
    NODE. Naming the culprit is as fatal as naming the mechanism, and for a
    propagated fault it is the easier mistake to make, because the culprit is a
    node the prompt has every innocent reason to mention.
    """
    found = set()
    lowered = text.lower()
    for word in FAULT_SYNONYMS.get(fault, ()):
        if word in lowered:
            found.add("fault word '%s'" % word)

    # The graph names every node, so the culprit's ROLE appears legitimately in
    # arm B's topology block. What must never appear is the culprit named in the
    # incident TEXT -- the title and summary, which is what arm A consists of.
    head = lowered.split("service dependency graph")[0]
    if fault_role != symptom_role and ("'%s'" % fault_role) in head:
        # Only when the culprit is NOT the node being paged about. When they are
        # the same node the title names it legitimately -- a local fault pages
        # the machine that failed, and hiding that would be stranger than
        # showing it.
        found.add("root-cause node '%s' named in the incident text" % fault_role)
    return found
