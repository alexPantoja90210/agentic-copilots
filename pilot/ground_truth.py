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
from datetime import datetime, timezone
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


def _windows(entries: list[dict]) -> list[tuple[datetime, datetime, str]]:
    """(start, end, incident_id) for every OPEN record. End is the planned one."""
    out = []
    for e in entries:
        if e.get("record") != "open":
            continue
        out.append((_parse(e["window_start"]), _parse(e["window_end_planned"]), e["incident_id"]))
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
) -> dict:
    """Write the label. Called BEFORE the fault is induced — never after."""
    if fault not in FAULT_CLASSES:
        raise GroundTruthError(f"Unknown fault {fault!r}. Known: {sorted(FAULT_CLASSES)}")
    if window_end_planned <= window_start:
        raise GroundTruthError("window_end_planned must be after window_start.")

    existing = read_all(path)

    if any(e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"Incident {incident_id} already exists in the log.")

    for start, end, other in _windows(existing):
        if window_start < end and start < window_end_planned:
            raise GroundTruthError(
                f"Window [{_utc(window_start)} .. {_utc(window_end_planned)}] overlaps "
                f"incident {other} [{_utc(start)} .. {_utc(end)}]. Two faults in one "
                "window produce a label that is a guess about which one the model saw."
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
    return _append(record, path)


def close_incident(
    incident_id: str,
    window_end_actual: datetime,
    outcome: str,
    notes: str = "",
    path: Path = DEFAULT_LOG,
) -> dict:
    """Close by APPENDING. The open record is never rewritten."""
    existing = read_all(path)
    if not any(e.get("record") == "open" and e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"No open record for incident {incident_id}.")
    if any(e.get("record") == "close" and e.get("incident_id") == incident_id for e in existing):
        raise GroundTruthError(f"Incident {incident_id} is already closed.")

    record = {
        "record": "close",
        "incident_id": incident_id,
        "window_end_actual": _utc(window_end_actual),
        "outcome": outcome,
        "notes": notes,
        "written_at": _utc(),
    }
    return _append(record, path)
