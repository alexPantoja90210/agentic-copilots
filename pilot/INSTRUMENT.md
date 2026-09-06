# Measuring what context is worth

**A reproducible instrument for the question Microsoft left open.**

Status: design. Batch 1 ran and produced a calibration result; the design below
is what the second attempt is built on. Nothing here is a claim about results.

---

## 1. The open question

Microsoft Research, *Large language models for automatic cloud incident
management* (2023), fed an LLM the **title and summary** of an incident and
asked for a root cause and mitigation steps, across 40,000+ incidents from
1,000+ production services. Their closing paragraph names what they did not try:

> "How can we incorporate additional context about the incident, such as
> discussion entries, logs, service metrics, and even dependency graphs of the
> impacted services to improve the diagnosis?"

Two things about that sentence are worth reading carefully.

**It is a limitation, not a result.** Their reported gains — *"at least 15.38%"*
for root cause — compare **GPT-3.5 against GPT-3**: two models, the same input.
They never ran with-context against without-context. There is no published
baseline for the comparison, because nobody published the comparison.

**It asks "how can we incorporate", not only "does it help".** Half the question
is engineering: what do you actually put in the prompt, where does it come from,
and how do you keep it true.

## 2. Why almost nobody can answer it

Their method needs 40,000 real incidents with human-written titles and
engineer-written root causes. That corpus is the study, and it is not
obtainable outside an organisation that already has it.

So the question sits open — not because it is uninteresting, but because the
entry price is a production estate.

**The instrument below removes that requirement.** It costs about a dollar of
compute and a weekend, and it is reproducible by anyone with an AWS account.

## 3. What it is

Four ingredients. Each replaces something the production corpus supplied.

| production study | this instrument |
| --- | --- |
| 40,000 incidents that happened | a handful of incidents **caused on purpose** |
| root cause written in a postmortem | root cause known because the operator **caused it** |
| scoring by text similarity to that write-up | scoring by **exact match** on the service named |
| no cost reported | input tokens per arm, measured |

### 3.1 A real topology, small

Three `t3.micro` instances in a chain that is genuinely dependent, not merely
labelled as such:

```
web  ──pulls from──▶  app  ──pulls from──▶  db
```

Each node serves **only what it fetched from its upstream**, and empties its
cache when the fetch fails. A failure therefore propagates the full length of
the chain. Measured: stopping `db` collapsed both `app` and `web` from ~3,105 KB
to ~13 KB per five-minute datapoint, against a healthy spread of ±2 KB.

The dependency is a property of the running system, not of a diagram.

### 3.2 Ground truth by causation

The label is written to an append-only log **before** the fault is induced. If
the operator stopped a service, the root cause is not a matter of opinion.

This is stronger than the production study's ground truth, which is what an
engineer wrote afterwards — an annotation carrying hindsight and judgement.
It is also the constraint that makes everything else delicate: see §5.

### 3.3 Arms that isolate one variable

Three arms, each the previous one plus exactly one thing:

| arm | content |
| --- | --- |
| **A** | incident title + a **symptom line** describing what was observed |
| **B** | A, plus the metric window for every node |
| **C** | B, plus the dependency graph |

**A→B measures what metrics are worth. B→C measures what the graph is worth.**
The two-arm design used in the first attempt conflated them, and the graph is
the thing the open question actually names.

Arm A's text must be contained verbatim in B, and B's in C, so the only
difference reaching the model is the added context.

### 3.4 Cost, measured alongside accuracy

Input and output tokens per arm, per incident. The first run measured arm B at
**12.4x** arm A's input tokens.

"Does context help" is half an answer for anyone deciding whether to deploy it.
The other half is what it costs per incident, at the volume they actually have.

## 4. How the context is incorporated

This is the engineering half of the open question, and these are the parts worth
copying.

**The graph is read back from the infrastructure that defines it.** Terraform
writes `ChainRole` and `DependsOn` tags onto the instances; the code reads them
back through the EC2 API. The topology the agent reasons over and the topology
that exists are the same object, so a graph maintained by hand cannot drift out
of date against reality — the usual failure of dependency documentation.

**Absence is named, never left as silence.** A stopped service publishes no
datapoints, and a gap in a metric series is the single most informative thing in
a stop fault. Every absence carries the same sentence — *"no datapoint at 03:46.
Absent data is not a value of zero"* — whether it falls in the middle of a series
or at its end.

**Transition buckets are labelled as transitions.** EC2 basic monitoring
publishes every five minutes, so the datapoint containing the injection is
neither the healthy value nor the failed one. It is shown, and marked, so
nothing downstream reads it as a state.

**The generator refuses to emit a prompt containing its own answer.** A leak
detector checks the built text for the fault class, its synonyms, and the
root-cause node's role, and refuses the build if any appear. It has its own
negative test: a fixture with a deliberately leaked cause must turn it red.

**Every clock face is UTC, converted explicitly.** The first live build declared
the window in UTC and printed the series in the operator's local time, six hours
apart. The arithmetic was right and the output was incoherent.

## 5. The trap, and how to avoid it

**This is the finding the first attempt produced, and it is the reason this
document exists.**

Ground truth by causation pulls towards faults that are easy to cause and easy
to verify: stop the instance, saturate its CPU. Both are all-or-nothing. And a
node that is off publishes nothing, while a node at 100% says so.

**The culprit identifies itself, and the graph becomes decorative.**

Batch 1 demonstrated this in the model's own words. Across six incidents the
identifying sentence was always local:

> *"Service 'db' **stopped reporting metrics entirely** from 03:02 to 03:22"*
> *"**Web service showed severe CPU exhaustion** ... downstream services showed
> normal behavior"*

The dependency graph appeared afterwards, to narrate the propagation. Delete it
from the prompt and every answer stands.

**The fix is not to weaken the ground truth. It is to inject subtler faults.**

| fault | how it is caused | what the culprit's metrics show |
| --- | --- | --- |
| latency | `tc netem delay 500ms` on the upstream | running, publishing, CPU normal |
| packet loss | `tc netem loss 30%` on the link | running, nothing obviously wrong |
| partial serve | upstream returns half the payload | running, throughput halved |

The operator still knows exactly what was broken and when. But now **all three
nodes degrade roughly in proportion**, none is absent, none is pinned — and the
only way to choose among them is to know which one feeds the others.

**A fault catalogue must contain at least one class in which the culprit does not
stand out, or the instrument cannot measure what it was built to measure.**

## 6. What it can and cannot answer

**Can.** Whether a given kind of context changes the answer, on faults of a
chosen difficulty, in a topology you control, at a measured cost per incident.
Whether a model abstains or confabulates when the evidence is insufficient.

**Cannot.** Anything about production incidents. Three nodes is not an estate;
injected faults are cleaner than real ones; a single model at one point in time
is not a claim about models. An instrument does not have to generalise to be
worth having — it has to be correct, cheap, and honest about its scope.

## 7. What it costs

Three `t3.micro` at $0.0104/hour, stopped between sessions. A twelve-call run
against a six-incident corpus consumed 18,930 tokens.

Total for the first attempt, infrastructure and inference: **under two dollars.**

## 8. Status

**Built and exercised against the live account:** the three-node chain and its
Terraform, the injection harness with append-only ground truth, the
contamination guard, the credit guard, the incident builder with its leak
detector, the two-arm runner, and the blind scorer that refuses to state a
verdict below the pre-registered sample size.

**Not built:** the degrading fault class of §5, the symptom line in arm A, and
the third arm. Those are the redesign, tracked as IA-63.

**Result to date:** a calibration finding, reported as such. Twelve answers,
twelve honouring the output contract, zero invented services, and five honest
abstentions out of five where the prompt supported no attribution. H1 remains
untested.
