"""
Selftest for the ground-truth log.

Every check here is written so that it CAN fail: each one first proves the
guard rejects the bad case, then proves the good case still passes. A check
that can only pass is not a check — the rule this project keeps re-learning.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ground_truth as gt

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def expect_raises(name: str, fn, needle: str) -> None:
    try:
        fn()
    except gt.GroundTruthError as exc:
        ok = needle.lower() in str(exc).lower()
        results.append((name, PASS if ok else FAIL,
                        "" if ok else f"raised, but message lacks {needle!r}: {exc}"))
    except Exception as exc:  # noqa: BLE001
        results.append((name, FAIL, f"wrong exception type {type(exc).__name__}: {exc}"))
    else:
        results.append((name, FAIL, "no exception raised — the guard did not fire"))


def expect_ok(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        results.append((name, FAIL, f"{type(exc).__name__}: {exc}"))
    else:
        results.append((name, PASS, ""))


def run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "gt.jsonl"

        # --- location guards -------------------------------------------------
        expect_raises(
            "log under OneDrive is refused",
            lambda: gt.check_location(Path(tmp) / "OneDrive" / "Docs" / "gt.jsonl"),
            "onedrive",
        )

        repo = Path(tmp) / "repo"
        (repo / ".git").mkdir(parents=True)
        expect_raises(
            "log inside a git repository is refused",
            lambda: gt.check_location(repo / "pilot" / "gt.jsonl"),
            "repository",
        )
        expect_ok("log outside repo and OneDrive is accepted", lambda: gt.check_location(log))

        # --- the happy path --------------------------------------------------
        expect_ok(
            "first incident opens",
            lambda: gt.open_incident("F1-a", "F1", "i-0test", T0, T0 + timedelta(minutes=10),
                                     "selftest", path=log),
        )

        # --- overlap ---------------------------------------------------------
        expect_raises(
            "overlapping window is refused",
            lambda: gt.open_incident("F3-b", "F3", "i-0test", T0 + timedelta(minutes=5),
                                     T0 + timedelta(minutes=15), "selftest", path=log),
            "overlaps",
        )
        expect_ok(
            "a window that starts after the previous one ends is accepted",
            lambda: gt.open_incident("F3-b", "F3", "i-0test", T0 + timedelta(minutes=10),
                                     T0 + timedelta(minutes=20), "selftest", path=log),
        )

        # --- identity --------------------------------------------------------
        expect_raises(
            "duplicate incident id is refused",
            lambda: gt.open_incident("F1-a", "F1", "i-0test", T0 + timedelta(hours=2),
                                     T0 + timedelta(hours=3), "selftest", path=log),
            "already exists",
        )
        expect_raises(
            "unknown fault class is refused",
            lambda: gt.open_incident("F9-x", "F9", "i-0test", T0 + timedelta(hours=4),
                                     T0 + timedelta(hours=5), "selftest", path=log),
            "unknown fault",
        )
        expect_raises(
            "a window that ends before it starts is refused",
            lambda: gt.open_incident("F1-c", "F1", "i-0test", T0 + timedelta(hours=6),
                                     T0 + timedelta(hours=5), "selftest", path=log),
            "after",
        )

        # --- closing is appending, never rewriting ---------------------------
        before = log.read_text(encoding="utf-8")
        expect_ok(
            "closing appends a record",
            lambda: gt.close_incident("F1-a", T0 + timedelta(minutes=10), "completed", path=log),
        )
        after = log.read_text(encoding="utf-8")
        results.append((
            "the open record is left byte-identical after closing",
            PASS if after.startswith(before) else FAIL,
            "" if after.startswith(before) else "the existing content changed",
        ))

        expect_raises(
            "closing twice is refused",
            lambda: gt.close_incident("F1-a", T0 + timedelta(minutes=11), "completed", path=log),
            "already closed",
        )
        expect_raises(
            "closing an unknown incident is refused",
            lambda: gt.close_incident("nope", T0, "completed", path=log),
            "no open record",
        )

        # --- append-only ordering -------------------------------------------
        # Forge a backdated entry the way a careless edit would.
        expect_raises(
            "a backdated entry is refused",
            lambda: gt._append(
                {"record": "open", "incident_id": "forged", "written_at": "2000-01-01T00:00:00+00:00"},
                log,
            ),
            "out-of-order",
        )

        # --- the negative control -------------------------------------------
        # If the overlap guard were removed, the check above would pass anyway
        # unless it is genuinely exercised. Prove the fixture really overlaps.
        entries = gt.read_all(log)
        opens = [e for e in entries if e["record"] == "open"]
        results.append((
            "the fixture used for the overlap test genuinely overlaps",
            PASS if len(opens) == 2 else FAIL,
            "" if len(opens) == 2 else f"expected 2 open records, found {len(opens)}",
        ))

    # ---- IA-61: the guard must protect the PROMPT, not only the window ----
    # Five of the first six incidents this project built contained a
    # neighbour's fault, because the plain overlap rule watches the interval the
    # operator causes while the prompt is built from a 90-minute-wider one. The
    # control was the worst case: its correct answer is "nothing happened" and
    # its prompt held a real 20-minute outage.
    import incident_builder as _ib
    PRE, POST = _ib.PRE_MINUTES, _ib.POST_MINUTES
    base = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)

    def fresh_with_a(context=True):
        """A log holding one 25-minute incident starting at `base`."""
        log = Path(tempfile.mkdtemp()) / "gt.jsonl"
        extra = dict(context_pre_minutes=PRE, context_post_minutes=POST) if context else {}
        gt.open_incident(incident_id="A", fault="F3", instance_id="i-0x",
                         window_start=base,
                         window_end_planned=base + timedelta(minutes=25),
                         operator="selftest", path=log, **extra)
        return log

    def place(log, start_at, context=True, name="B"):
        extra = dict(context_pre_minutes=PRE, context_post_minutes=POST) if context else {}
        return lambda: gt.open_incident(
            incident_id=name, fault="F3", instance_id="i-0x",
            window_start=start_at,
            window_end_planned=start_at + timedelta(minutes=25),
            operator="selftest", path=log, **extra)

    expect_ok("a window separated by the full lead-in is allowed",
              place(fresh_with_a(), base + timedelta(minutes=25 + PRE)))

    expect_raises("one minute short of the lead-in is refused",
                  place(fresh_with_a(), base + timedelta(minutes=25 + PRE - 1)),
                  "Earliest window this may open")

    # RED: overlapping ONLY in the padding -- the exact shape of the six
    # incidents built on 3 and 4 Sep. The windows do not touch, so the plain
    # overlap rule allows it. That is the defect, reproduced.
    padding_only = base + timedelta(minutes=26)      # one minute after A ends
    expect_raises("RED: a padding-only overlap is refused",
                  place(fresh_with_a(), padding_only),
                  "does not overlap another incident")

    # ...and the proof it is the new rule doing the work: the same two windows,
    # with no context declared, are still accepted. If this ever starts failing,
    # the check above has stopped being specific to the padding.
    expect_ok("RED confirmed: with no context declared, the same pair passes",
              place(fresh_with_a(context=False), padding_only, context=False))

    # The other direction: a window landing inside an EARLIER incident's tail.
    # It matters on its own, because that prompt was written first and cannot be
    # repaired afterwards.
    late = Path(tempfile.mkdtemp()) / "gt.jsonl"
    gt.open_incident(incident_id="A", fault="F3", instance_id="i-0x",
                     window_start=base + timedelta(minutes=200),
                     window_end_planned=base + timedelta(minutes=225),
                     operator="selftest", path=late,
                     context_pre_minutes=PRE, context_post_minutes=POST)
    # The fixture has to be honest about what it is testing: B must sit inside
    # A's lead-in WITHOUT touching A's window, or the plain overlap rule fires
    # and this test passes while proving nothing. (It did exactly that on the
    # first attempt.)
    b_start = base + timedelta(minutes=170)
    b_end = b_start + timedelta(minutes=25)
    a_start = base + timedelta(minutes=200)
    results.append((
        "the fixture for the context test does NOT plainly overlap",
        PASS if b_end <= a_start else FAIL,
        "" if b_end <= a_start else "B ends %s, A starts %s" % (b_end, a_start)))
    results.append((
        "and it does sit inside the earlier incident's lead-in",
        PASS if b_end > a_start - timedelta(minutes=PRE) else FAIL, ""))

    expect_raises("a window inside an earlier incident's context is refused",
                  place(late, b_start),
                  "context window")

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        print(f"{status:4}  {name:<{width}}  {detail}")
        failed += status == FAIL
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
