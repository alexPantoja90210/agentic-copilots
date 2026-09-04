"""
Selftest for the incident builder. No AWS, no model.

What is being protected here is the experiment's validity, not the code's
behaviour. If the prompt contains its own answer, every number the pilot
produces afterwards is meaningless in a way no later check would notice -- the
runs would succeed, the transcripts would look fine, and the accuracy would be
high for the wrong reason.

So most of these are negatives.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import chain_topology as ct
import incident_builder as ib

PASS, FAIL = "PASS", "FAIL"
results = []

NODES = {
    "db":  {"role": "db",  "instance_id": "i-0db",  "private_ip": "10.0.0.1",
            "state": "running", "depends_on": None},
    "app": {"role": "app", "instance_id": "i-0app", "private_ip": "10.0.0.2",
            "state": "running", "depends_on": "db"},
    "web": {"role": "web", "instance_id": "i-0web", "private_ip": "10.0.0.3",
            "state": "running", "depends_on": "app"},
}

T0 = datetime(2026, 9, 2, 18, 9, tzinfo=timezone.utc)


def entry(fault="F3", incident="F3-20260902T180900"):
    return {
        "record": "open", "incident_id": incident, "fault": fault,
        "fault_class": "instance_unavailable", "instance_id": "i-0db",
        "window_start": T0.isoformat(),
        "window_end_planned": (T0 + timedelta(minutes=8)).isoformat(),
        "operator": "selftest", "dry_run": False,
        "written_at": T0.isoformat(),
    }


def metrics(collapse_from=(), stopped=()):
    """
    A window shaped like the one IA-55 actually measured on 2 Sep:

      - nodes downstream of the fault collapse from ~3105 to ~13 KB;
      - the node that was STOPPED produces no datapoints at all after the fault,
        because a stopped instance does not report. The gap is the clue, and an
        earlier version of this fixture had the culprit sitting at a healthy
        3105 -- which would have made the correct answer unfindable and let the
        suite bless a prompt nobody could solve.
    """
    out = {}
    for role in NODES:
        points = []
        for i in range(8):
            stamp = T0 - timedelta(minutes=20) + timedelta(minutes=5 * i)
            if role in stopped and stamp > T0:
                continue  # stopped instances do not publish
            failing = role in collapse_from and stamp > T0
            points.append((stamp, 13.0 if failing else 3105.0))
        out[role] = {"NetworkIn": points}
    return out


def check(name, condition, detail=""):
    results.append((name, PASS if condition else FAIL, "" if condition else detail))


def expect_error(name, fn, needle):
    try:
        fn()
    except ib.BuilderError as exc:
        check(name, needle.lower() in str(exc).lower(), f"message lacks {needle!r}: {exc}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
    else:
        check(name, False, "no error raised — the guard did not fire")


def run() -> int:
    # ---- who gets paged ----
    check("a fault at the tail pages the most user-facing node",
          ib.symptomatic_node("db", NODES) == "web",
          ib.symptomatic_node("db", NODES))
    check("a fault in the middle also pages the user-facing node",
          ib.symptomatic_node("app", NODES) == "web")
    check("a local fault pages the node that failed",
          ib.symptomatic_node("web", NODES) == "web")

    # ---- the two arms ----
    FIXTURE = metrics(collapse_from={"web", "app"}, stopped={"db"})
    built = ib.build(entry(), NODES, FIXTURE, "db")

    check("arm A is contained verbatim in arm B", built["arm_a"] in built["arm_b"])
    check("arm A carries no metrics", "NetworkIn" not in built["arm_a"])
    check("arm B carries the dependency graph", "depends on" in built["arm_b"])

    # The heart of it: for a two-hop fault the title names web and the answer is db.
    check("the title names the symptom, not the cause",
          "'web'" in built["arm_a"] and "'db'" not in built["arm_a"],
          built["arm_a"])

    # ---- arm B must show EVERY node ----
    for role in NODES:
        check(f"arm B includes metrics for {role}",
              ("  %s:" % role) in built["arm_b"],
              "a graph is useless if only the hurting node's numbers are shown")

    # ---- the transition datapoint is labelled, and ONLY it ----
    check("the datapoint spanning the injection is marked as a transition",
          "the incident window begins inside this datapoint" in built["arm_b"])
    # The marker appears in EVERY node's series at the same wall-clock bucket,
    # including nodes that never failed. Calling it "the fault begins" there
    # invited reading it as an accusation of the node whose column it sat in.
    # It marks the window, so it says window.
    check("the marker describes the window, not a fault in the node it sits on",
          "the fault begins inside" not in built["arm_b"])
    marks = built["arm_b"].count("transition, not a state")
    check("the marker is not sprayed across the whole window",
          marks <= 2 * len(NODES),
          "%d markers — a marker on every row points at nothing" % marks)

    # ---- the two clocks must be one clock ----
    # The first live build declared the window in UTC and printed the series in
    # the operator's local time, six hours away. Every timestamp was tz-aware,
    # so every comparison was correct and every marker landed in the right
    # bucket -- the arithmetic was never wrong. Only the rendering was, and no
    # test looked at the rendering, because the tests built their fixtures in
    # UTC and could not tell the two apart.
    #
    # So: the same instants, expressed in a zone that is not UTC.
    from datetime import timezone as _tz, timedelta as _td
    minus_six = _tz(_td(hours=-6))
    shifted = {role: {name: [(stamp.astimezone(minus_six), value)
                             for stamp, value in points]
                      for name, points in per_metric.items()}
               for role, per_metric in FIXTURE.items()}
    elsewhere = ib.build(entry(), NODES, shifted, "db")

    check("a series in another zone renders in UTC, not in the zone it arrived in",
          elsewhere["arm_b"] == built["arm_b"],
          "the same instants produced different text depending on their tzinfo")

    declared = T0.strftime("%H:%M")
    marked = [line.strip().split()[0] for line in elsewhere["arm_b"].splitlines()
              if "the incident window begins inside" in line]
    check("the marked datapoint and the declared window agree on the clock",
          marked and all(mark <= declared for mark in marked) and len(set(marked)) == 1,
          "summary says %s, series marks %s" % (declared, marked))

    # NEGATIVE: a timestamp with no zone must be refused, not assumed to be UTC.
    naive = {role: {name: [(stamp.replace(tzinfo=None), value)
                           for stamp, value in points]
                    for name, points in per_metric.items()}
             for role, per_metric in FIXTURE.items()}
    try:
        ib.build(entry(), NODES, naive, "db")
    except ib.BuilderError as exc:
        check("a timestamp with no timezone is refused rather than guessed",
              "no timezone" in str(exc), str(exc))
    else:
        check("a timestamp with no timezone is refused rather than guessed",
              False, "it rendered -- which is exactly how the six hours got in")

    # ---- a stopped node's gap is stated, not left as a silence ----
    check("a series that ends early says so, rather than trailing off",
          "no datapoint from" in built["arm_b"],
          "the gap IS the clue in a stop fault; leaving it as absent rows hides it")

    # ---- IA-59: an absence in the MIDDLE gets the same sentence ----
    # The function used to narrate a series that ended early and stay silent
    # about a hole inside one. Same signal, two treatments -- and in the pilot's
    # real data the unnarrated hole was the answer to the incident.
    holed = metrics(collapse_from={"web", "app"}, stopped=set())
    for name, points in holed["app"].items():
        holed["app"][name] = points[:2] + points[5:]     # buckets 2,3,4 removed
    with_hole = ib.build(entry(), NODES, holed, "db")

    check("an absence in the middle of a series is named",
          "no datapoint from" in with_hole["arm_b"],
          "the hole is silent -- a reader has to spot missing timestamps by eye")
    check("a run of missing datapoints states the span it covers",
          with_hole["arm_b"].count("no datapoint from") >= 1
          and " to " in [line for line in with_hole["arm_b"].splitlines()
                         if "no datapoint from" in line][0],
          [line for line in with_hole["arm_b"].splitlines() if "no datapoint" in line][:2])

    single = metrics(collapse_from={"web", "app"}, stopped=set())
    for name, points in single["app"].items():
        single["app"][name] = points[:3] + points[4:]    # exactly one removed
    with_one = ib.build(entry(), NODES, single, "db")
    check("a single missing datapoint is named as one, not as a span",
          "no datapoint at " in with_one["arm_b"],
          [line for line in with_one["arm_b"].splitlines() if "no datapoint" in line][:2])

    # RED: the pre-IA-59 rendering had no interior detection at all. Its only
    # absence sentence was the trailing one, so on this fixture -- whose series
    # runs to the end of the window -- it produced no absence sentence whatever.
    def pre_ia59(points, window_start, window_end):
        rows = []
        for stamp, value in points:
            rows.append("    %s  %10.0f" % (stamp.strftime("%H:%M"), value))
        if points and points[-1][0] < window_end:
            rows.append("    (no datapoints after %s.)" % points[-1][0].strftime("%H:%M"))
        return "\n".join(rows)

    hole_points = holed["app"]["NetworkIn"]
    old_render = pre_ia59(hole_points, T0, T0 + timedelta(minutes=8))
    new_render = ib._format_series(hole_points, T0, T0 + timedelta(minutes=8))
    check("RED: the old rendering left this same hole entirely unmentioned",
          "no datapoint" not in old_render and "no datapoint" in new_render,
          "old=%r" % old_render)

    check("a series with no gaps gains no absence sentences",
          "no datapoint" not in ib._format_series(
              metrics()["db"]["NetworkIn"], T0, T0 + timedelta(minutes=8)),
          "a sentence on every row points at nothing")

    check("the interior absence sentence survives the leak detector",
          not ib.leaks("F3", "app", "web", with_hole["arm_b"]),
          "naming an absence must stay an observation, not a conclusion")
    check("and the distinction from zero is spelled out",
          "not a value of zero" in built["arm_b"])
    check("the gap is described without using the fault's own vocabulary",
          not ib.leaks("F3", "db", "web", built["arm_b"]),
          "presenting data must not become narrating a conclusion")

    # ---- no environment identifiers ----
    check("no instance id reaches either arm",
          not any(n["instance_id"] in built["arm_b"] for n in NODES.values()))

    # ---- usability is reported ----
    empty = ib.build(entry(), NODES, {role: {"NetworkIn": []} for role in NODES}, "db")
    check("a window with no datapoints is marked unusable", not empty["usable"])
    check("and says which node produced nothing", len(empty["notes"]) == 3, str(empty["notes"]))
    check("a window with datapoints is marked usable", built["usable"])

    # ---- NEGATIVES: the leak detector must be able to fire ----
    expect_error(
        "a fault word in the template is refused",
        lambda: _with_summary(
            "Service '{node}' stopped responding between {start} and {end} UTC.",
            entry(), "db"),
        "fault word")

    expect_error(
        "naming the culprit in the incident text is refused",
        lambda: _with_summary(
            "An incident was opened for '{node}'; also check 'db' between {start} and {end}.",
            entry(), "db"),
        "root-cause node")

    check("the same node named in a LOCAL fault is not a leak",
          "'web'" in ib.build(entry(fault="F1"), NODES, metrics(), "web")["arm_a"],
          "a local fault legitimately pages the machine that failed")

    expect_error("an unknown fault node is refused",
                 lambda: ib.symptomatic_node("cache", NODES), "not a node")

    # ---- the leak detector, exercised directly in both directions ----
    check("leaks() finds a fault word",
          ib.leaks("F3", "db", "web", "the service was stopped") != set())
    check("leaks() finds the culprit named in the text",
          ib.leaks("F3", "db", "web", "check 'db' please") != set())
    check("leaks() passes clean text",
          ib.leaks("F3", "db", "web", "Incident opened for 'web'.") == set())

    width = max(len(n) for n, _, _ in results)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


def _with_summary(template, ent, fault_role):
    original = ib.SUMMARY
    ib.SUMMARY = template
    try:
        return ib.build(ent, NODES, metrics(), fault_role)
    finally:
        ib.SUMMARY = original


if __name__ == "__main__":
    raise SystemExit(run())
