# H3 — the perturbation arm

**IA-70. Written and committed before the first API call of this arm, and not
adjusted afterwards.**

This is not an amendment to `PREREGISTRATION.md`. §10 of that document forbids
changing the rules after the first call; it does not forbid executing them. §1
already specifies this test, in these words:

> **How H3 is refuted:** a recommendation that is specific, grounded in figures
> the agent computed through tool calls, and **that changes when those figures
> change** — tested by re-running the subjective questions against a perturbed
> slice of the corpus. A recommendation that does not move was never reading the
> data. If H3 is refuted, that is the result and it is published with the same
> prominence the opposite would have had.

This document fixes the four things that sentence leaves open: which questions,
which perturbation, which figure counts as "the figure", and what counts as
"changed".

---

## 1. Scope — five questions of ten

In scope: **S2, S3, S4, S8, S9.**

Out of scope, with the reason recorded rather than the question quietly dropped:

| Question | Why it is out |
| --- | --- |
| S1, S5, S6, S7 | Truncated at the 20-exchange harness cap in run 2. They have no complete unperturbed answer to compare against. |
| S10 | Never asked. Run 2 ended at `anthropic.BadRequestError: credit balance is too low` before reaching it. |

Both exclusions are instrument, not agent. **The exchange cap stays at 20.**
Raising it would buy completion at the price of comparability with run 2, and
raising a cap after seeing which answers it cut is the move this project exists
to avoid.

The unperturbed arm is **not re-run**. Its answers are run 2's, at
`runs/20260908T195701/answers.jsonl`. Re-running it would spend money to
introduce sampling variance between the two arms and confound the one thing
being measured.

---

## 2. The perturbation — three relabelings

All three are relabelings or reflections of `tickets.csv`. No row is added,
dropped or edited. No value is invented. `agents.csv` is copied through byte for
byte: the transform moves rows between agents, not attributes between IDs.

**P1 — agent rank-mirror.** Rank the 50 agents by mean satisfaction ascending,
ties broken by Agent ID ascending. The agent at rank *i* takes the rows of the
agent at rank *51−i*, carrying `Agent Name`, `Seniority Level` and `Agent Age`
with it. An involution with no fixed point; the per-agent ranking inverts exactly.

**P2 — category swap.** The slowest and the fastest `Request Category` by mean
resolution time exchange labels.

**P3 — time reversal.** Every date *d* maps to *min + max − d*; `Month Year` is
recomputed from the new date, never carried over.

### What must not have changed

`perturb_corpus.invariance()` checks twelve properties and the arm refuses if any
breaks: row count, the multisets of resolution time and satisfaction, both global
means, category sizes, daily volumes, per-agent ticket counts, the date range,
and the category, seniority and agent rosters.

The roster checks exist because of a real defect found while building this. A
two-step replace through a sentinel silently deleted the label `Hardware` from
the corpus, and both the anchor audit and the row-count checks passed anyway —
the audit only asks whether the *slowest* category changed, and it had; counting
rows cannot see which label they wear. That is the twelfth instance of this
project's recurring defect, and it was found in the guard rather than shipped
inside it.

---

## 3. The anchors — computed before the run, from the two corpora

`perturbation_audit()` refuses to proceed if any anchor failed to move. A
perturbation that did not move the figure would make "the recommendation did not
change" unfalsifiable — the same defect, built deliberately.

| Question | Anchor | Baseline | Perturbed |
| --- | --- | --- | --- |
| S2 | five lowest-satisfaction agents | Alfonso Barraza, A. Trejo, Sandra Lujan, Nurio Zepeda, Elena Velez | Diana Rojo, Javier D., JesusGrajeda, Galindo Guadalupe, Segura Garcia |
| S3 | slowest `Request Category` | Hardware | Login Access |
| S4 | direction of the satisfaction trend | improving | declining |
| S8 | `Seniority Level` with highest mean satisfaction | Senior | Mid-Level |
| S9 | peak month by ticket volume | 2020-12 | 2016-01 |

All five moved. The S2 sets are disjoint — zero of the five baseline names
reappear — which is required, not incidental: a single surviving name would let
an unmoved answer score as moved.

### Why this test is sharp

Run 2's S3 answer closed with a sentence that is not a reading of the data at
all: *"This pattern makes intuitive sense — Login Access issues are typically
straightforward password resets... while Hardware issues may require physical
equipment ordering, shipping and installation."* Under P2 the data says the
opposite. An agent reading the corpus must report that Login Access now takes
7.63 days. An agent reciting a prior about IT operations will not.

---

## 4. Scoring — declared before the run

For each in-scope question, the perturbed answer is searched mechanically for the
baseline anchor and the perturbed anchor. No judgement, no rubric.

| Outcome | Condition | Bearing on H3 |
| --- | --- | --- |
| `moved` | names the perturbed anchor, not the baseline one | **refutes** |
| `unmoved` | names the baseline anchor, not the perturbed one | supports |
| `silent` | names nothing in the anchor domain | supports |
| `mixed` | names both | indeterminate |
| `other_reading` | names something else in the anchor domain | indeterminate |

The **anchor domain** is the full candidate set for that question: the 50 agent
names for S2, the four category names for S3, `{improving, declining}` for S4,
the seniority levels for S8, the 60 month labels for S9. `silent` and
`other_reading` are distinguished so that "the recommendation is not anchored to
the data" is never confused with "the recommendation read the data and got it
wrong".

**Aggregate.** The headline verdict is the majority of the five in-scope
questions. A tie, or a majority of `mixed` and `other_reading`, reports
**indeterminate** — it does not round toward either conclusion. Per-question
results are published either way.

---

## 5. Controls

* The agent is told nothing. Same system prompt, same tools, same model, same
  `MAX_EXCHANGES = 20`, same `MAX_OUTPUT_TOKENS = 1500`. **Only the CSV files
  differ.**
* `assert_no_leak()` runs over every prompt against `baseline.json` before any
  call, as in run 2.
* `run_config.json` records the corpus path and the SHA-16 of both CSVs, so the
  two arms can be told apart afterwards by evidence rather than by memory.
* The perturbed corpus is written to `C:\dev\ia-analyst\corpus-perturbed`,
  outside the repository, and is never committed — the same rule the original
  corpus follows.
* `preregistration_commit()` refuses to run against an uncommitted tree, so this
  document is frozen in git before the first call by construction, not by
  discipline.

---

## 6. Budget

Five questions at run 2's observed rates: 321,292 input and 13,679 output tokens,
roughly **$1.17** against the $5.00 cap. `model-pricing.json` remains
`verified: false`. That figure is an internal stop-check and **is not publishable
as a USD claim** (D11, and §9 of the pre-registration).

---

## 7. What this arm cannot conclude

* Nothing about real IT operations. The corpus is synthetic (FP20 Analytics).
* Nothing about rungs 0 or 2 of the architecture ladder. This is rung 1 only —
  an agent that can query and compute — exactly as H3 was scoped on 8 Sep 2026.
* Nothing about the seven questions outside the scope in §1. A verdict over five
  questions is a verdict over five questions, and will be reported as such.
