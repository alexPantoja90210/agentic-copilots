# Ops Triage Copilot — read-only agentic incident triage

A second agentic AI project that reuses the **same safe pattern** as my FinOps Copilot, ported to a new domain — proving the pattern is portable, not a one-off.

> Read-only signals → tool-use LLM → **code-forced structured plan** → **deterministic invariant in code** → **eval gate with negative tests** → a human approves. The agent **proposes**; it never acts.

## What it does
Given a read-only observability snapshot (`signals.json`: alerts, metrics vs. baseline, recent changes, runbook), the agent:
- reads the signals with read-only tools (no command execution, no writes),
- returns a **structured triage plan**: ranked probable causes (each grounded in real evidence ids), and a **proposed** next runbook step,
- writes `triage_plan.json` + `incident_brief.md`. Nothing in any system is changed.

## The guardrails (why it's safe)
1. **Read-only tools only** — `get_alerts`, `get_context`, `get_runbook`. No tool can restart, roll back, or scale anything.
2. **No invented signals** — the eval checks every service and evidence id in the plan exists in the snapshot.
3. **Evidence-grounded** — every ranked cause must cite ≥1 real evidence id (alert `AL-*`, change `CH-*`, or metric `M-*`); evidence is IDs only, never free text.
4. **Propose, never execute** — a policy check forbids "restarted / rolled back / scaled / resolved" language.
5. **Deterministic invariant in code** — the `top_cause` is *derived in code* from a severity score (severity × change-correlation), never trusted from the model. Self-consistent by construction.

## The eval harness (the differentiator)
`evals.py` scores every run against golden fixtures on 4 hard checks: **top-correct, no-hallucination, evidence-grounded, policy-ok**. A run is GREEN only if all pass — a go/no-go gate. A `--selftest` mode validates the checker itself with a good/bad plan and **no API calls** (free, deterministic).

Fixtures:
- `case1_deploy` — error spike right after a deploy → top = the deployed service (change-correlated), DB CPU is a red herring.
- `case2_saturation` — critical CPU with no deploy → top = the saturated service (change-correlation must not over-fire).
- `case3_allclear` — no alerts → no incident, empty causes (tests the "nothing's wrong" path).

## Run it
```
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install anthropic
setx ANTHROPIC_API_KEY "sk-..."                       # your key in an env var, never in code

python evals.py --selftest     # free: validates the checker (no API)
python triage.py               # live: reads signals.json, prints + writes the triage plan
python evals.py                # live: scores the agent against the fixtures (GREEN/RED gate)
```
Model is env-configurable: `set TRIAGE_MODEL=claude-...`.

## The input contract, and where the input comes from

`signals.json` used to be hand-written and loaded with a bare `json.load`. That
was survivable only while producer and consumer were the same person.

IA-4 introduced a collector, and the moment something *produces* this file the
two can drift apart. That is not hypothetical: it is IA-26 in the sibling agent,
where the tools were reading a schema `guardian.py` had stopped producing and
the model was fluent enough to hide the gap.

So the contract came first. `signals_contract.py` enforces one rule above all:

> **No alert and no metric may refer to a service that is not in `services`.**

An alert about a service nobody declared is the same defect as a plan ranking a
resource that does not exist. From inside the conversation the agent cannot tell
the difference. `load_snapshot` now refuses such a file by name:

```
alerts[1]: refers to service 'ghost-svc', which is not in 'services'.
An alert about a service nobody declared cannot be triaged.
```

Note what the contract does **not** check: whether the values are *right*.
Nothing here can know that. It checks that the shape is usable and the
references resolve.

## The CloudWatch collector

`collector.py` reads EC2 metrics from the live CloudWatch API, read-only, and
writes a snapshot that satisfies the contract.

```bash
python collector.py us-east-1 290 7      # region, window hours, baseline days
```

**It writes `signals.live.json`, never `signals.json`.** The committed file is
the hand-written worked example and stays that way; the collected one carries
real instance ids from a real account, and this repository is public. It is
gitignored for that reason.

### What it does not collect, and says so

`recent_changes` is deploy history — CloudTrail or a CI system, not CloudWatch.
`runbook` is operational knowledge, written by people. Both are declared in
`source.not_collected` **with the reason**, because an empty list would read as
"nothing was deployed" when the truth is "nobody is looking". Those are very
different things to a triage agent and identical in an empty array.

### Every metric is emitted or explained

The invariant is borrowed from the sibling planner because it was the right one:
a metric with no datapoints goes into `source.skipped_metrics` with its reason.
Never dropped. A collector that quietly discards a metric produces a snapshot
that looks complete and is not.

### Thresholds are a choice, not a truth

Nothing in AWS says 80% CPU is a problem. The thresholds live in `THRESHOLDS`,
are echoed into `source.thresholds` on every run, and exist to be argued with.
That is the difference between a threshold and a magic number.

### Two defects the first live run exposed

Both were constants where functions belonged, and **neither was caught by the
offline tests — because the test data was chosen by the same person who wrote
the assumption.**

1. **A fixed 3-hour window.** Against an account whose instances were terminated
   eleven days earlier it collected nothing, and the empty snapshot passed the
   contract cleanly while reading as a healthy quiet system. The collector now
   aborts when it finds targets but no usable data.
2. **A fixed 300-second period.** `get_metric_statistics` returns at most 1440
   datapoints, so widening the window to reach the data would have failed the
   call instead. The period is now computed from the window and the age of the
   data, respecting CloudWatch's own resolution limits.

The second would only have appeared *after* fixing the first. Running against
reality found both in one afternoon.

### Provenance, and why it matters here

A real run against a real account produced one alert: a CPU credit balance of
15.05 against a threshold of 30. It is correctly derived and it describes a
machine that no longer exists. Without `source.window_start` and
`source.collected_at`, that alert reads as an incident in progress.

## The gate needs no AWS account

`evals.py --selftest` drives the collector with recorded CloudWatch shapes: no
credentials, no network, no cost. The negative cases are the point — an account
with no data must abort, a quiet account must produce **zero** alerts rather
than an invented one, and a snapshot referencing an undeclared service must be
refused. Fixtures use invented instance ids.

## Honest limits
- Read-only by design; it proposes, a human approves. That's the feature.
- Evals run on fixtures (known answers), not live telemetry — intentional, to keep the gate deterministic.
- `signals.json` is a sample snapshot; the architecture is ready to consume a real read-only export (e.g., CloudWatch alarms + Logs Insights + deploy history) with no change to `triage()` or `evals.py`.

## Same pattern as FinOps — that's the point
| Slot | FinOps Copilot | Ops Triage Copilot |
|---|---|---|
| Data → snapshot | AWS → `report.json` | Observability → `signals.json` |
| Read-only tools | costs / waste / budget | alerts / context / runbook |
| Structured plan | ranked actions + brief | ranked causes + proposed step |
| Invariant in code | top_action = highest cost | top_cause = highest severity |
| Eval checks | total/top/halluc/policy | top/halluc/evidence/policy |

Two domains, one discipline. The transferable skill is moving the fragile invariant out of the LLM and proving the gate can fail.
