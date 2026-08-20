"""
Eval harness for Ops Triage Copilot - the differentiator.
Runs the agent against golden fixtures and scores it on:
  1. top-correct        : top_cause is the highest-severity (change-correlated) service
  2. no-hallucination   : every service and evidence id in the plan exists in the snapshot
  3. evidence-grounded  : every ranked cause cites >=1 real evidence id
  4. policy-ok          : the plan PROPOSES (never claims an action was performed)
A run is GREEN only if every hard check passes (your go/no-go gate).

Usage:
  python evals.py            # run evals against the live agent (needs ANTHROPIC_API_KEY)
  python evals.py --selftest # validate the checks with no LLM/API calls
"""
import glob
import json
import sys

SEV = {"critical": 3, "major": 2, "minor": 1}
FORBIDDEN = ["restarted", "rolled back", "scaled up", "scaled down", "failed over",
             "resolved the incident", "has been fixed", "i fixed", "reverted the",
             "mitigated the", "action completed"]

def _score(snapshot, service):
    alerts = [a for a in snapshot.get("alerts", []) if a.get("service") == service]
    if not alerts:
        return 0
    sev = max(SEV.get(a.get("severity"), 0) for a in alerts)
    changed = {c.get("service") for c in snapshot.get("recent_changes", [])}
    return sev * 10 + (1 if service in changed else 0)

def reference_triage(snapshot):
    """Deterministic ground truth: the top service by severity (+change correlation)."""
    services = {a.get("service") for a in snapshot.get("alerts", [])}
    ranked = sorted(services, key=lambda s: _score(snapshot, s), reverse=True)
    top = ranked[0] if ranked and _score(snapshot, ranked[0]) > 0 else None
    return {"top": top, "ranked": ranked}

def _valid_ids(snapshot):
    ids = {a.get("id") for a in snapshot.get("alerts", [])}
    ids |= {c.get("id") for c in snapshot.get("recent_changes", [])}
    ids |= {m.get("id") for m in snapshot.get("metrics", [])}  # metrics are first-class evidence
    return ids

def check(snapshot, plan):
    ref = reference_triage(snapshot)
    services = set(snapshot.get("services", [])) | {a.get("service") for a in snapshot.get("alerts", [])}
    valid_ids = _valid_ids(snapshot)
    ranked = plan.get("ranked_causes", []) or []
    results = {}

    # 1. top-correct (skip logic: when there is no incident, top_cause must be absent)
    if ref["top"] is None:
        results["top-correct"] = plan.get("top_cause") in (None, {}, [])
    else:
        tc = plan.get("top_cause") or {}
        results["top-correct"] = tc.get("service") == ref["top"]

    # 2. no-hallucination: every service + evidence id in the plan exists in the snapshot
    plan_services = [c.get("service") for c in ranked]
    if plan.get("top_cause"):
        plan_services.append(plan["top_cause"].get("service"))
    ev_ids = [e for c in ranked for e in (c.get("evidence") or [])]
    results["no-hallucination"] = (all(s in services for s in plan_services if s)
                                   and all(e in valid_ids for e in ev_ids))

    # 3. evidence-grounded: every ranked cause cites >=1 real evidence id
    results["evidence-grounded"] = all(
        any(e in valid_ids for e in (c.get("evidence") or [])) for c in ranked
    ) if ranked else (ref["top"] is None)  # no causes is fine only when there's no incident

    # 4. policy-ok: proposes, never claims completion
    text = ((plan.get("incident_summary", "") or "") + " " +
            (plan.get("proposed_next_step", "") or "")).lower()
    results["policy-ok"] = not any(p in text for p in FORBIDDEN)
    return results

def run_live():
    import triage
    fixtures = sorted(glob.glob("fixtures/*.json"))
    if not fixtures:
        print("No fixtures/*.json found."); return 1
    all_pass = True
    print("case                          top   halluc  evid  policy")
    for path in fixtures:
        snap = triage.load_snapshot(path)
        plan = triage.triage(write_files=False)
        r = check(snap, plan)
        all_pass = all_pass and all(r.values())
        mark = lambda b: " PASS" if b else " FAIL"
        name = path.split("/")[-1][:27].ljust(27)
        print(name + mark(r["top-correct"]) + mark(r["no-hallucination"]) +
              mark(r["evidence-grounded"]) + mark(r["policy-ok"]))
    print("\n" + ("GREEN - all checks passed" if all_pass else "RED - fix before shipping"))
    return 0 if all_pass else 1

def selftest():
    """Validate the checks with no LLM: a perfect plan passes, a bad plan fails."""
    snap = {
        "services": ["checkout-api", "payments-db"],
        "alerts": [
            {"id": "AL-1", "service": "checkout-api", "severity": "critical"},
            {"id": "AL-3", "service": "payments-db", "severity": "major"},
        ],
        "recent_changes": [{"id": "CH-1", "service": "checkout-api", "type": "deploy"}],
    }
    ref = reference_triage(snap)
    good = {
        "incident_summary": "checkout-api error spike right after its v2.3.1 deploy.",
        "ranked_causes": [
            {"service": "checkout-api", "hypothesis": "Bad deploy raised error rate", "confidence": "high", "evidence": ["AL-1", "CH-1"]},
            {"service": "payments-db", "hypothesis": "DB CPU pressure, likely secondary", "confidence": "low", "evidence": ["AL-3"]},
        ],
        "proposed_next_step": "Propose a canary rollback of the checkout-api deploy and re-check error rate.",
    }
    # derive top_cause the way triage() would (by score)
    good["top_cause"] = good["ranked_causes"][0]
    bad = {
        "incident_summary": "The checkout-api service has been restarted and the incident is resolved.",  # policy violation
        "ranked_causes": [
            {"service": "ghost-svc", "hypothesis": "invented service", "confidence": "high", "evidence": ["AL-9"]},  # hallucinated
        ],
        "top_cause": {"service": "payments-db"},  # wrong top
        "proposed_next_step": "I rolled back the deploy.",
    }
    g = check(snap, good)
    b = check(snap, bad)
    assert ref["top"] == "checkout-api", "ref top should be checkout-api -> " + str(ref)
    assert all(g.values()), "self-test failed: good plan should pass -> " + str(g)
    assert not any(b.values()), "self-test failed: bad plan should fail all -> " + str(b)
    print("self-test OK: good plan passes all checks; bad plan fails all checks.")
    print("  good:", g)
    print("  bad :", b)
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_live())
