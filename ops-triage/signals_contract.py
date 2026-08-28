"""
The contract for signals.json — what Ops Triage requires of its input.

Until IA-4 this file was written by hand and `load_snapshot` read it with a bare
`json.load`. That was survivable only because producer and consumer were the
same person. IA-4 introduces a collector, and the moment something PRODUCES this
file, producer and consumer can drift apart.

That drift is not hypothetical. It is IA-26, in this repository, in the sibling
agent: the tools were reading a schema `guardian.py` had stopped producing, and
the model papered over the gap so convincingly that nothing looked wrong.

The rule this file exists to enforce, and the one worth stating on its own:

    NO ALERT AND NO METRIC MAY REFER TO A SERVICE THAT IS NOT IN `services`.

An alert about a service nobody declared is the same defect as a plan ranking a
resource that does not exist. The agent cannot catch it — from inside the
conversation an invented service is indistinguishable from a real one.

Note what this contract deliberately does NOT check: whether the VALUES are
right. Nothing here can know whether an error rate of 0.42 is true. It checks
that the shape is usable and the references resolve. Correctness of the numbers
is the collector's problem, and provenance is how it is defended.
"""

KNOWN_SEVERITIES = ("critical", "major", "minor")

REQUIRED_ROOT = ("window", "services", "alerts", "metrics")
OPTIONAL_ROOT = ("recent_changes", "runbook", "source")

REQUIRED_ALERT_FIELDS = ("id", "service", "signal", "severity", "value",
                         "threshold", "first_seen")
REQUIRED_METRIC_FIELDS = ("id", "service", "name", "value", "baseline")
REQUIRED_CHANGE_FIELDS = ("id", "service", "type", "at", "summary")
REQUIRED_RUNBOOK_FIELDS = ("cause_type", "proposed_step")

# A collected file must say where it came from. A hand-written one may not have
# provenance, but a file that claims to be collected and cannot say from where
# is worse than one that never claimed anything.
REQUIRED_SOURCE_FIELDS = ("collector", "region", "collected_at",
                          "window_start", "window_end", "not_collected")


class SignalsContractError(ValueError):
    """Raised when the input does not meet the contract. Never swallowed."""


def _is_number(value):
    # bool is a subclass of int in Python; a boolean threshold is not a threshold.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_items(items, label, required_fields, services, problems,
                 service_key="service"):
    seen_ids = set()
    for index, item in enumerate(items):
        where = "%s[%d]" % (label, index)
        if not isinstance(item, dict):
            problems.append("%s is not an object" % where)
            continue

        for field in required_fields:
            if field not in item:
                problems.append("%s: missing '%s'" % (where, field))

        item_id = item.get("id")
        if item_id is not None:
            if item_id in seen_ids:
                problems.append("%s: duplicate id '%s' — ids must be unique"
                                % (where, item_id))
            seen_ids.add(item_id)

        service = item.get(service_key)
        if service is not None and services is not None and service not in services:
            problems.append(
                "%s: refers to service '%s', which is not in 'services'. "
                "An alert about a service nobody declared cannot be triaged."
                % (where, service))


def validate_signals(snapshot):
    """
    Return a list of contract violations. An empty list means the snapshot is
    usable. Never raises: the caller decides what to do with the problems.
    """
    problems = []

    if not isinstance(snapshot, dict):
        return ["the snapshot is not an object"]

    for key in REQUIRED_ROOT:
        if key not in snapshot:
            problems.append("missing top-level key '%s'" % key)

    if not isinstance(snapshot.get("window", ""), str):
        problems.append("'window' must be a string describing the period observed")

    services = snapshot.get("services")
    if not isinstance(services, list):
        problems.append("'services' must be a list")
        services = None
    else:
        for index, name in enumerate(services):
            if not isinstance(name, str) or not name:
                problems.append("services[%d] is not a non-empty string" % index)
        if len(set(services)) != len(services):
            problems.append("'services' contains duplicates")

    alerts = snapshot.get("alerts")
    if not isinstance(alerts, list):
        problems.append("'alerts' must be a list (empty is valid and meaningful)")
    else:
        _check_items(alerts, "alerts", REQUIRED_ALERT_FIELDS, services, problems)
        for index, alert in enumerate(alerts):
            if not isinstance(alert, dict):
                continue
            severity = alert.get("severity")
            if severity is not None and severity not in KNOWN_SEVERITIES:
                problems.append("alerts[%d]: unknown severity '%s'. Known: %s"
                                % (index, severity, ", ".join(KNOWN_SEVERITIES)))
            for field in ("value", "threshold"):
                if field in alert and not _is_number(alert[field]):
                    problems.append("alerts[%d]: '%s' must be a number, got %r"
                                    % (index, field, alert[field]))

    metrics = snapshot.get("metrics")
    if not isinstance(metrics, list):
        problems.append("'metrics' must be a list")
    else:
        _check_items(metrics, "metrics", REQUIRED_METRIC_FIELDS, services, problems)
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            for field in ("value", "baseline"):
                if field in metric and not _is_number(metric[field]):
                    problems.append("metrics[%d]: '%s' must be a number, got %r"
                                    % (index, field, metric[field]))

    changes = snapshot.get("recent_changes")
    if changes is not None:
        if not isinstance(changes, list):
            problems.append("'recent_changes' must be a list when present")
        else:
            _check_items(changes, "recent_changes", REQUIRED_CHANGE_FIELDS,
                         services, problems)

    runbook = snapshot.get("runbook")
    if runbook is not None:
        if not isinstance(runbook, list):
            problems.append("'runbook' must be a list when present")
        else:
            for index, entry in enumerate(runbook):
                if not isinstance(entry, dict):
                    problems.append("runbook[%d] is not an object" % index)
                    continue
                for field in REQUIRED_RUNBOOK_FIELDS:
                    if field not in entry:
                        problems.append("runbook[%d]: missing '%s'" % (index, field))

    problems.extend(_validate_source(snapshot.get("source")))

    return problems


def _validate_source(source):
    """
    Provenance is optional — a hand-written snapshot has none. But a snapshot
    that HAS a source block must fill it completely: half a provenance record is
    a claim without backing.
    """
    if source is None:
        return []
    if not isinstance(source, dict):
        return ["'source' must be an object when present"]

    problems = []
    for field in REQUIRED_SOURCE_FIELDS:
        if field not in source:
            problems.append("source: missing '%s'. A file that claims to be "
                            "collected must say where it came from." % field)

    skipped = source.get("skipped_metrics")
    if skipped is not None:
        if not isinstance(skipped, list):
            problems.append("source.skipped_metrics must be a list")
        else:
            for index, entry in enumerate(skipped):
                if not isinstance(entry, dict):
                    problems.append("source.skipped_metrics[%d] is not an object" % index)
                    continue
                for field in ("metric", "target", "reason"):
                    if not entry.get(field):
                        problems.append(
                            "source.skipped_metrics[%d]: missing '%s'. A metric "
                            "that was considered and left out must say which one "
                            "and why — dropping it silently is the defect."
                            % (index, field))

    not_collected = source.get("not_collected")
    if not_collected is not None:
        if not isinstance(not_collected, dict):
            problems.append("source.not_collected must be an object mapping "
                            "each uncollected field to the reason")
        else:
            for field, reason in not_collected.items():
                if not isinstance(reason, str) or not reason.strip():
                    problems.append("source.not_collected['%s'] must give a "
                                    "reason, not an empty value" % field)
    return problems


def describe_problems(path, problems):
    lines = ["%s does not meet the signals contract:" % path]
    lines.extend("  - " + problem for problem in problems)
    lines.append("")
    lines.append("Triaging an input that does not meet its contract produces a "
                 "confident answer about something that is not there.")
    return "\n".join(lines)
