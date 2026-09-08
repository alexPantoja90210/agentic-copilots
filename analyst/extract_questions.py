"""
Pull the 23 task questions out of the reference workbook's task document.

    python extract_questions.py --tasks "<...>/Tasks.docx" --out C:\\dev\\ia-analyst\\reference

Why this writes OUTSIDE the repository
--------------------------------------
The reference project (github.com/febx1/IT-Ticket-Management-Excel-Project)
carries **no LICENSE file and no copyright notice**, which means all rights are
reserved by default. Using it locally and citing it is ordinary practice.
Committing its workbook, its task document, or its text verbatim into a public
repository is redistribution, and it is not ours to do.

So the repository carries the METHOD -- this file -- and the extracted content
lands in the quarantine directory outside it. Anyone can reproduce the
extraction from their own copy of the source.

The second reason is IA-65: Tasks.docx contains the ANSWERS as well as the
questions. It is quarantined material, and the agent's corpus directory is a
sibling of the quarantine, never a parent of it.

Why the indices are recorded rather than detected
-------------------------------------------------
A heuristic would get this wrong, and wrong quietly. The subjective section of
the document contains the analyst's own rhetorical headings -- "What is
Cost-Benefit Analysis?", "Do We Need to Fire Any Agents?" -- which end in a
question mark and are not task questions. Nine of the ten real subjective
questions are followed by a line beginning "Analysis:"; the tenth is not. Any
rule tight enough to exclude the headings also drops the tenth question, and any
rule loose enough to keep it admits headings.

That is the project's recurring defect in miniature: a check that cannot
distinguish two situations it treats as one. So the questions were curated by
reading the document, their positions recorded, and a checksum added. If the
document changes, this refuses rather than silently returning a different set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import docx

# Curated on 8 Sep 2026 by reading Tasks.docx end to end.
OBJECTIVE = (3, 5, 10, 12, 15, 18, 20, 23, 25, 29, 31, 36, 40)
SUBJECTIVE = (47, 85, 106, 115, 131, 142, 170, 199, 233, 258)

# First 16 hex characters of the SHA-256 of the Tasks.docx these indices were
# read from. A different document invalidates them.
TASKS_SHA16 = "1647482fc37488d4"


class ExtractError(Exception):
    pass


def paragraphs(path: Path) -> list[str]:
    return [p.text.strip() for p in docx.Document(str(path)).paragraphs if p.text.strip()]


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def extract(tasks: Path) -> dict:
    got = checksum(tasks)
    if got != TASKS_SHA16:
        raise ExtractError(
            "Tasks.docx has checksum %s, the recorded indices were read from %s. "
            "The paragraph positions may no longer point at the questions. Re-read "
            "the document and update OBJECTIVE/SUBJECTIVE/TASKS_SHA16 together, "
            "rather than trusting stale positions." % (got, TASKS_SHA16))

    ps = paragraphs(tasks)
    out = {"source": tasks.name, "tasks_sha16": got, "objective": [], "subjective": []}
    for label, indices, bucket in (("O", OBJECTIVE, "objective"),
                                   ("S", SUBJECTIVE, "subjective")):
        for n, i in enumerate(indices, 1):
            if i >= len(ps):
                raise ExtractError("paragraph %d is past the end of the document" % i)
            text = ps[i]
            # A question ends with '?' -- except O13, which the author closed
            # with a bracketed instruction. Both are accepted; anything else
            # means the indices have drifted and the run stops.
            if not (text.endswith("?") or text.endswith("]")):
                raise ExtractError(
                    "paragraph %d does not look like a question: %r" % (i, text[:80]))
            out[bucket].append({"id": "%s%d" % (label, n), "para": i, "question": text})
    if len(out["objective"]) != 13 or len(out["subjective"]) != 10:
        raise ExtractError("expected 13 objective and 10 subjective questions")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, help="path to the reference Tasks.docx")
    ap.add_argument("--out", required=True,
                    help="quarantine directory, OUTSIDE the repository and outside "
                         "the agent's corpus directory")
    args = ap.parse_args(argv)

    try:
        data = extract(Path(args.tasks))
    except ExtractError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("%d objective + %d subjective -> %s"
          % (len(data["objective"]), len(data["subjective"]),
             out_dir / "questions.json"))
    print("This directory is quarantined. The agent's corpus is a SIBLING of it, "
          "never a parent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
