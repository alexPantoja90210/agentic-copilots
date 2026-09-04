"""
Ground-truth log for the IA-45 pilot.

The whole pilot's validity rests on one property: the label for an incident is
written BEFORE the fault is induced, and never edited afterwards. A label
written after seeing a model's answer is not a label, it is a rationalisation.

This module enforces that property mechanically rather than trusting the
operator to remember it:

  - the log is append-only JSONL; an entry is closed by appending a second
    record, never by rewriting the first;
  - entries must arrive in non-decreasing time order;
  - a new incident whose window overlaps an open or recorded one is refused,
    because two faults inside one window produce a label that is a guess about
    which one the model saw;
  - the log refuses to live inside a git repository, and refuses to live under
    OneDrive. It carries instance ids. D7 exists because files that must never
    leave the machine do not belong in a folder that replicates them by design.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_LOG = Path(os.environ.get("IA_PILOT_LOG", r"C:\dev\ia-pilot\ground_truth.jsonl"))

FAULT_CLASSES = {
    "F0": "no_fault",
    "F1": "cpu_saturation",
    "F2": "cpu_credit_exhaustion",
    "F3": "instance_unavailable",
}


class GroundTruthError(Exception):
    """The log refused a write. Every message names what would have been violated."""


def _utc(ts=None) -> str:
    return (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def check_location(path: Path) -> None:
    """Refuse a log path that would be committed or synced to the cloud."""
    resolved = path.expanduser().resolve()
    parts_lower = [p.lower() for p in resolved.parts]

    if "onedrive" in parts_lower:
        raise GroundTruthError(
            f"{resolved} is under OneDrive. The log carries instance ids and OneDrive "
            "replicates regardless of .gitignore (D7). Set IA_PILOT_LOG to a path "
            "outside OneDrive."
        )

    for parent in [resolved, *resolved.parents]:
        if (parent / ".git").exists():
            raise GroundTruthError(
                f"{resolved} is inside the git repository at {parent}. The log carries "
                "instance ids and must not be committable. Set IA_PILOT_LOG to a path "
                "outside any repository."
            )


def read_all(path: Path = DEFAULT_LOG) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GroundTruthError(f"{path}:{n} is not valid JSON: {exc}") from exc
    return out


def _append(record: dict, path: Path) -> dict:
    check_location(path)
    existing = read_all(path)

    if existing:
        last = _parse(existing[-1]["written_at"])
        if _parse(record["written_at"]) < last:
            raise GroundTruthError(
                "Refusing an out-of-order write: this entry is timestamped "
                f"{record['written_at']}, earlier than the last entry at "
                f"{existing[-1]['written_at']}. An append-only log that accepts "
                "backdated entries is not append-only."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# Faults that induce nothing. Their windows occupy time and produce no signal,
# so they cannot contaminate anybody else's context -- there is no foreign event
# inside them to be seen. They still need protecting themselves, and more than
# anything else does: the control's whole purpose is that its window is empty,
# and a neighbour's outage bleeding into its lead-in is what destroys it.
SIGNAL_FREE_FAULTS = frozenset({"F0"})


def _windows(entries: list[dict]) -> list[tuple[datetime, datetime, str, str]]:
    """(start, end, incident_id, fault) for every OPEN record. End is the planned one."""
    out = []
    for e in entries:
        if e.get("record") != "open":
            continue
        out.append((_parse(e["window_start"]), _parse(e["window_end_planned"]),
                    e["incident_id"], e.get("fault", "")))
    return out


def open_incident(
    incident_id: str,
    fault: str,
    instance_id: str,
    window_start: datetime,
    window_end_planned: datetime,
    operator: str,
    path: Path = DEFAULT_LOG,
    dry_run: bool = False,
    details: dict | None = None,
    # IA-61: how much context the built prompt will carry either side of
    # this window. Zero means the caller is not building prompts from it
    # and only the plain overlap rule applies.
    context_pre_minutes: int = 0,
    context_post_minutes: int = 0,
) -> dict:
    """
    Write the label. Called BEFORE the fault is induced — never after.

    `details` carries what the label MEANS rather than how it was produced: the
    node's role, the graph it sat in. An instance id is not something a scorer
    can compare an answer against, and this account closes in February 2027 —
    after which an id resolves to nothing while a role still reads.
    """
    if fault not in FAULT_CLASSES:
        raise GroundTruthError(f"Unknown fault {fault!r}. Known: {sorted(FAULT_CLASSES)}")
    if window_end_planned <= window_start:
        raise GroundTruthError("window_end_planned must be after window_start.")

    existing = read_all(path)

    if any(e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"Incident {incident_id} already exists in the log.")

    pre = timedelta(minutes=context_pre_minutes)
    post = timedelta(minutes=context_post_minutes)
    conflicts = []
    i_am_silent = fault in SIGNAL_FREE_FAULTS
    for start, end, other, other_fault in _windows(existing):
        if window_start < end and start < window_end_planned:
            raise GroundTruthError(
                f"Window [{_utc(window_start)} .. {_utc(window_end_planned)}] overlaps "
                f"incident {other} [{_utc(start)} .. {_utc(end)}]. Two faults in one "
                "window produce a label that is a guess about which one the model saw."
            )

        # IA-61. The check above protects the interval the OPERATOR causes. The
        # prompt is built from a wider interval -- the window plus the context
        # either side -- and a neighbour's fault landing in that padding is
        # visible to the agent while the label says nothing about it.
        #
        # Both directions matter. A neighbour inside my padding contaminates my
        # prompt; my window inside a neighbour's padding contaminates theirs,
        # and theirs was written first and cannot be fixed afterwards.
        # Asymmetric on purpose. Two questions, and only one of them is about me.
        they_are_silent = other_fault in SIGNAL_FREE_FAULTS
        if (not they_are_silent
                and start < window_end_planned + post and window_start - pre < end):
            conflicts.append((other, start, end, "their fault sits in my context window"))
        elif (not i_am_silent
                and window_start < end + post and start - pre < window_end_planned):
            conflicts.append((other, start, end, "my fault sits in their context window"))

    if conflicts:
        clearance = max(pre, post)
        earliest = max(end for _o, _s, end, _w in conflicts) + clearance
        detail = "; ".join(
            "%s [%s .. %s] -- %s" % (o, _utc(s_), _utc(e), why)
            for o, s_, e, why in conflicts)
        raise GroundTruthError(
            f"Window [{_utc(window_start)} .. {_utc(window_end_planned)}] does not "
            f"overlap another incident, but its {context_pre_minutes}-minute lead-in "
            f"and {context_post_minutes}-minute tail do: {detail}.\n"
            f"The prompt an agent reads is the window PLUS that context, so a "
            f"neighbouring fault inside it is evidence the label does not account "
            f"for -- and the control, whose correct answer is that nothing "
            f"happened, is the case it ruins first.\n"
            f"Earliest window this may open: {_utc(earliest)} "
            f"({clearance.total_seconds() / 60:.0f} minutes after the last "
            f"conflicting window ends)."
        )

    record = {
        "record": "open",
        "incident_id": incident_id,
        "fault": fault,
        "fault_class": FAULT_CLASSES[fault],
        "instance_id": instance_id,
        "window_start": _utc(window_start),
        "window_end_planned": _utc(window_end_planned),
        "operator": operator,
        "dry_run": dry_run,
        "written_at": _utc(),
    }
    for key, value in (details or {}).items():
        if key in record:
            raise GroundTruthError(
                "details may not overwrite the field %r of an open record" % key)
        record[key] = value
    return _append(record, path)


def close_incident(
    incident_id: str,
    window_end_actual: datetime,
    outcome: str,
    notes: str = "",
    path: Path = DEFAULT_LOG,
    details: dict | None = None,
) -> dict:
    """
    Close by APPENDING. The open record is never rewritten.

    `details` carries structured facts about how the run ended — the state the
    instance was left in, whether service came back and when. IA-52: those were
    buried in a prose note, where nothing can query them and a reader has to
    parse a sentence to learn that infrastructure was left broken.
    """
    existing = read_all(path)
    if not any(e.get("record") == "open" and e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"No open record for incident {incident_id}.")
    if any(e.get("record") == "close" and e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"Incident {incident_id} is already closed.")

    record = {
        "record": "close",
        "incident_id": incident_id,
        # When the HARNESS stopped acting. Not necessarily when service
        # returned: those coincide only if the restore succeeded, and on
        # 1 Sep 2026 they did not. `service_restored_at` in details says which.
        "window_end_actual": _utc(window_end_actual),
        "outcome": outcome,
        "notes": notes,
        "written_at": _utc(),
    }
    for key, value in (details or {}).items():
        if key in record:
            raise GroundTruthError(
                "details may not overwrite the field %r of a close record" % key)
        record[key] = value
    return _append(record, path)
