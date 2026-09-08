# analyst — can an agent reproduce a validated human analyst deliverable?

IA-64. A human analyst answered 23 questions about an IT ticket dataset in
Excel, documented the method for each, and published the workbook. This asks
whether an LLM agent, given the same raw data and the same questions, reproduces
the computed answers, abstains instead of inventing when it cannot compute, and
produces a defensible recommendation — and what that costs against the analyst's
effort.

It is a **tool-use** experiment, not a classification one: the agent has to query
the data and compute. An answer that arrives without a tool call is recorded as
unsupported, not as correct.

## The reference project is not ours, and is not redistributed here

The source is **github.com/febx1/IT-Ticket-Management-Excel-Project** — the
workbook, the task document and the dashboards are that author's work.

**That repository carries no LICENSE file and no copyright notice, so all rights
are reserved by default.** Using it locally and citing it is ordinary practice.
Committing its workbook, its task document, or verbatim reproductions of its
text into this public repository is redistribution, and it is not ours to do.

So this directory holds the **method**. The reference content stays on the
operator's disk, outside the repository, and anyone reproducing this work
supplies their own copy of the source. The underlying dataset appears to be a
synthetic FP20 Analytics challenge set; that is stated wherever a figure derived
from it is published, because it bounds what any finding here can claim.

## Where things live, and why they live apart

    C:\Users\...\GitHub\IT-Ticket-Management-Excel-Project\   READ ONLY
        IT Tickets Analysis.xlsb      the source workbook
        Tasks.docx                    the questions AND the answers

    C:\Users\...\GitHub\agentic-copilots\analyst\             THIS DIRECTORY
        code, tests and the pre-registration. Committed.

    C:\dev\ia-analyst\                                        NEVER COMMITTED
        reference\   questions.json and the human answers      QUARANTINED
        corpus\      tickets.csv, agents.csv, manifest.json    THE AGENT SEES THIS
        runs\        transcripts, run_config.json, scores

`reference\` and `corpus\` are **siblings**. Neither contains the other. The
agent is given `corpus\` as its root, so the answers are not one path traversal
away — and `prepare_corpus.py` refuses to run if the workbook is inside the
corpus directory at all.

This mirrors `pilot\` and `C:\dev\ia-pilot\`: code in the repository, everything
the run touches outside it.

## Order of work, and it is not negotiable

1. **`extract_questions.py`** — pulls the 23 questions into the quarantine.
   Verifies the source document's checksum first and refuses on a mismatch.
2. **`prepare_corpus.py`** — writes the two permitted sheets to `corpus\`,
   scans them for leaked answer values, and records what it dropped. IA-65.
3. **`baseline.py`** — recomputes every objective answer in code and asserts it
   against the human's stated value, including where they disagree. IA-67.
4. **`PREREGISTRATION.md`** — committed **before the first API call**. IA-66.
5. Only then does anything spend money.

Steps 1–4 cost nothing. Running them in a different order is how a pilot ends up
with rules that were chosen after seeing the answers.

## Reproducing

    python extract_questions.py --tasks "<reference>\Tasks.docx" --out C:\dev\ia-analyst\reference
    python prepare_corpus.py --workbook "<reference>\IT Tickets Analysis.xlsb" --out C:\dev\ia-analyst\corpus
    python selftest_prepare.py "<reference>\IT Tickets Analysis.xlsb"

Requires `pyxlsb`, `python-docx` and `pandas`.

## Corpus facts, confirmed on the prepared output

97,498 tickets · 50 agents · 2016-01-01 to 2020-12-31 · 1,827 of 1,827 calendar
days present with no gap · no free-text column anywhere · `Column1` null in
every row and dropped · `Severity Imputated` and `Priority Imputated` fill zero
nulls and carry counts identical to their source columns, which is what proves
they impute nothing.
