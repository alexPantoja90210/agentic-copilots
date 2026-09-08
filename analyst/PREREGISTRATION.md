# PREREGISTRATION — IA-64

**Committed before the first API call. Nothing below is changed after it.**

Status: awaiting commit. The commit hash of this file is written into
`run_config.json` by the runner, so any run can prove which version of these
rules it ran under. A run whose `run_config.json` carries a hash that is not an
ancestor of the reported results is not scored.

---

## 1. What is being tested

**The claim.** Given the same raw data and the same questions a human analyst
answered in Excel, an LLM agent with tool use reproduces the computed answers,
abstains rather than inventing when it cannot compute, and produces a
recommendation.

**H3, as amended 8 Sep 2026 and scoped to this run.** Contextual synthesis is
not a capability the model has or lacks; it is a function of what the
architecture is given. This run tests **rung 1 only** — an agent that can query
the data and compute — and only inside this workbook, where the architecture is
held constant across all 23 questions and only the question type varies.

**Prediction:** the objective questions meet the success criterion in §7; the
subjective questions yield recommendations that are plausible and **not
falsifiable from the data the agent was given**.

**How H3 is refuted:** a recommendation that is specific, grounded in figures
the agent computed through tool calls, and **that changes when those figures
change** — tested by re-running the subjective questions against a perturbed
slice of the corpus. A recommendation that does not move was never reading the
data. If H3 is refuted, that is the result and it is published with the same
prominence the opposite would have had.

---

## 2. The question set

23 questions: **13 objective, 10 subjective**.

The set is not reproduced here. The reference project carries no LICENCE and no
copyright notice, so its text is not ours to redistribute — see the README. The
set is pinned instead, which is stronger than quoting it:

- Source document `Tasks.docx`, **SHA-256 prefix `1647482fc37488d4`**.
- Extracted by `extract_questions.py`, which verifies that checksum and refuses
  on a mismatch rather than returning a different set.
- Written to `C:\dev\ia-analyst\reference\questions.json` with the paragraph
  index of each question, so every id below resolves to one exact paragraph.

Questions are referred to as **O1–O13** and **S1–S10** throughout.

**No question is added, removed or reworded after the first call.** A question
later found ill-posed is **excluded and reported with its reason**, never
rewritten.

---

## 3. Ground truth

`baseline.py` (IA-67), recomputed from the corpus and asserted against the
analyst's published figures by `selftest_baseline.py`. The scorer reads
`baseline.json`; **no agent answer is ever compared against a hand-typed
number.**

10 of 11 published figures agree under the analyst's own documented method. The
eleventh is a second reading of an ambiguous question, resolved in §5.

---

## 4. Tolerance — six rules, one per answer type

There are six, not the four first estimated: single numeric values split into
counts, means and the correlation, and each needs a different band. Each rule is
applied mechanically by the scorer. **No tolerance is decided per answer.**

| | Applies to | Correct when |
|---|---|---|
| **A. Exact counts** | O1, O2, O10 figures | Matches the baseline integer exactly. A count is right or it is not. |
| **B. Means** | O3, O9, O11 | Within **0.5%** of the baseline value, **or** rounds to the figure the analyst published. |
| **C. Correlation** | O12 | Within **0.005** of the baseline and the same sign. This admits −0.041 and rejects Spearman's −0.017. |
| **D. Tables** | O4, O5, O8 | Every key present **and** every count exact. No partial credit: a distribution with one wrong bucket is a wrong distribution. |
| **E. Methods** | O6, O7 | The method, **executed against the corpus**, reproduces the delivered artefact. Judged by running it, never by comparing strings. |
| **F. Prose with figures** | O10 narrative, O13 | Every hard figure matches under rule A **and** the stated direction matches the data. |

Rule B's 0.5% band on the three means is ±0.27 on 53.365, ±0.023 on 4.5485 and
±0.20 on 40.06. It admits an agent that says "about 53.4" and rejects one that
says 52.

Rule C's 0.005 band survives floating-point differences across platforms — the
same figure computed on two machines differed in the fourteenth decimal — while
staying four orders of magnitude tighter than the gap between Pearson and
Spearman.

---

## 5. The three ambiguous questions, resolved here and now

Three questions have more than one defensible answer, and the question text does
not say which is meant. **The reading that counts is fixed below, before any
answer has been seen.** Deciding afterwards would be deciding in whichever
direction suited the result.

**The governing principle:** this experiment asks whether the agent reproduces
*a validated human deliverable*. The target is therefore **the analyst's own
documented method**, not the platonically best answer.

| | The ambiguity | **The reading that counts** | The rejected reading |
|---|---|---|---|
| **O1** | what "attributes present in the data" counts | **16** — source columns across both sheets, before the analyst's derived ones | 26, the workbook as delivered |
| **O9** | which average "daily average resolution time" means | **4.5485**, the mean of daily means — their documented pivot-by-date method | 4.5532, the global mean over all tickets |
| **O13** | same source-versus-delivered question, for categorical columns | **source columns only** | as delivered |

**O6 is resolved differently, because the analyst's authority does not settle
it.** Their documented formula
`=MID(R5,FIND("@",R5)+1,LEN(R5)-FIND("@",R5)-4)` returns `fp20analytics`, while
the `Email Domain` column they shipped in the same workbook holds
`fp20analytics.com`. Method and artefact contradict each other.

**Resolution, per the PO's decision of 8 Sep:** O6 and O7 ask *how would you do
this*. **Any method that, executed against the corpus, extracts the domain
correctly is scored correct**, whether or not it matches the analyst's string.
Correctness here is whether the method works.

---

## 6. Outcomes — six, counted separately and never pooled

| | Meaning |
|---|---|
| `correct` | Meets the tolerance rule for its answer type, under the reading fixed in §5. |
| `other_reading` | Right under a reading §5 rejected. **Not `wrong`.** Being right about a different question is not the same failure as being wrong, and collapsing the two would be the defect this project keeps finding. |
| `wrong` | A value that is neither. |
| `abstained` | Stated it could not compute. **This is a designed outcome, not a failure** — an agent that says so is worth more than one that invents a number. |
| `unsupported` | The value is right and **no tool call produced it**. A right answer arrived at without touching the data is luck, and luck does not generalise. |
| `no_contract` | No parseable answer. |

A single pooled accuracy figure is never reported. Pooling hides the difference
between an agent that admitted it could not compute and one that invented a
number, which is the whole point of measuring.

---

## 7. Success, and failure, stated the same way

**Success:** **11 of the 13 objective questions** scored `correct`.

Two of thirteen is the allowance, and it is there because three questions are
ambiguous as asked; one reasonable divergent reading should not sink the run.
`other_reading` counts toward neither success nor failure and is reported on its
own line.

**Failure:** **fewer than 11 of 13** scored `correct`.

If the run fails, that is the report. It is published with the same prominence
success would have had, as IA-45's negative result was.

**The ten subjective questions are not scored numerically and do not enter this
criterion.** Two recommendations can both be defensible; no winner is declared
by arithmetic. Each is presented beside the analyst's own written analysis, with
the differences named, and each is put through the perturbation test in §1.

---

## 8. Cost, and the sample

- **Hard cap: 5.00 USD**, enforced in code via `agent_budget`. The cap exists to
  stop a loop, not to budget the run.
- **Reconciliation** between the budget counter and the sum of the transcripts.
  A discrepancy **is** the finding and blocks the report until it is known which
  counter is measuring something else.
- **All 23 questions, one pass each.** No question is re-asked. No sampling.
- Wall-clock time per question is recorded; it is half the business metric and
  cannot be reconstructed afterwards.
- The model is recorded in `run_config.json`. **Sampling is not controlled** —
  the installed SDK does not expose `temperature`, and it is not smuggled
  through `extra_body`: a parameter nobody can verify was applied is worse than
  a declared absence. What this costs is reproducibility, and every record says
  so.

---

## 9. What may not be claimed from this run

- **Nothing about IT operations.** The corpus is a synthetic competition dataset
  — uniform email domain, category split of exactly 40.00 / 29.94 / 20.07 /
  9.98%. Any wording implying real operational insight is rejected in review.
- **Nothing about rung 2 or above**, and nothing about agents in general. This
  run measures one architecture on one workbook.
- **No dollar figure per token.** `model-pricing.json` is still unverified; cost
  is reported as measured spend and tokens, never as a derived price.
- The analyst's effort, used as the comparison, is an **estimate** read off
  their documented method. It is labelled an estimate wherever it appears.

---

## 10. Amendment rule

**Before the first call:** amendments are allowed, and each is recorded with its
date and reason with the original left visible beside it. Two have already
happened — H3's scope (8 Sep) and criterion 3's reversal (8 Sep), both on IA-66.

**After the first call:** none. A question found ill-posed is excluded and
reported with its reason. Nothing is rewritten.

---

## Sign-off

| | |
|---|---|
| Question set | `Tasks.docx` SHA-256 prefix `1647482fc37488d4` · 13 objective + 10 subjective |
| Ground truth | `baseline.py`, 34 checks green, commit `ef2218a` |
| Corpus | 97,498 tickets · 50 agents · 2016-01-01 to 2020-12-31 · commit `c523f0c` |
| Success criterion | 11 of 13 objective `correct` |
| Cost cap | 5.00 USD, enforced in code |
| Approved by | PO, 8 Sep 2026 |
