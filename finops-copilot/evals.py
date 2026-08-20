"""
Eval harness for FinOps Copilot - the differentiator.
Scores the agent on: total-correct, top-correct, no-hallucination, policy-ok.
GREEN only if every hard check passes (your go/no-go gate).
  python evals.py            # run against the live agent (needs ANTHROPIC_API_KEY)
  python evals.py --selftest # validate the checks with no LLM/API calls (free)
"""
import glob
import json
import sys

TOL = 0.01
FORBIDDEN = ["has been released", "was released", "has been deleted", "was deleted",
             "successfully deleted", "successfully released", "i released", "i deleted",
             "action completed", "resource removed"]

def reference_plan(report):
    waste = report.get("waste", [])
    ranked = sorted(waste, key=lambda w: w.get("monthly_cost", 0), reverse=True)
    total = round(sum(w.get("monthly_cost", 0) for w in waste), 2)
    top = ranked[0] if ranked else None
    return {"total": total, "ranked": ranked, "top": top}

def _plan_costs(plan):
    costs = [plan.get("total_monthly_waste_usd", 0)]
    if plan.get("top_action"):
        costs.append(plan["top_action"].get("monthly_cost_usd", 0))
    for a in plan.get("ranked_actions", []):
        costs.append(a.get("monthly_cost_usd", 0))
    return costs

def check(report, plan):
    ref = reference_plan(report)
    source_costs = [round(w.get("monthly_cost", 0), 2) for w in report.get("waste", [])]
    source_costs.append(ref["total"])
    results = {}
    results["total-correct"] = abs(plan.get("total_monthly_waste_usd", -1) - ref["total"]) <= TOL
    if ref["top"] is None:
        results["top-correct"] = plan.get("top_action") in (None, {}, [])
    else:
        results["top-correct"] = bool(plan.get("top_action")) and \
            plan["top_action"].get("resource") == ref["top"].get("resource")
    results["no-hallucination"] = all(
        any(abs(round(c, 2) - s) <= TOL for s in source_costs) for c in _plan_costs(plan)
    )
    brief = (plan.get("exec_brief", "") or "").lower()
    results["policy-ok"] = not any(p in brief for p in FORBIDDEN)
    return results

def run_live():
    import copilot
    fixtures = sorted(glob.glob("fixtures/*.json"))
    if not fixtures:
        print("No fixtures/*.json found."); return 1
    all_pass = True
    print("case                         total  top   halluc policy")
    for path in fixtures:
        report = copilot.load_report(path)
        plan = copilot.plan(write_files=False)
        r = check(report, plan)
        all_pass = all_pass and all(r.values())
        mark = lambda b: " PASS" if b else " FAIL"
        name = path.split("/")[-1][:26].ljust(26)
        print(name + mark(r["total-correct"]) + mark(r["top-correct"]) +
              mark(r["no-hallucination"]) + mark(r["policy-ok"]))
    print("\n" + ("GREEN - all checks passed" if all_pass else "RED - fix before shipping"))
    return 0 if all_pass else 1

def selftest():
    report = {"waste": [
        {"resource": "Elastic IP eip-1", "type": "unused_elastic_ip", "monthly_cost": 3.65, "fix": "Release it"},
        {"resource": "EBS vol-2", "type": "unattached_ebs", "monthly_cost": 0.80, "fix": "Delete it"},
    ]}
    ref = reference_plan(report)
    good = {
        "total_monthly_waste_usd": ref["total"],
        "top_action": {"resource": ref["top"]["resource"], "monthly_cost_usd": 3.65, "fix": "Release it"},
        "ranked_actions": [{"resource": w["resource"], "monthly_cost_usd": w["monthly_cost"], "fix": w["fix"]} for w in ref["ranked"]],
        "exec_brief": "I recommend releasing the unused Elastic IP to save $3.65/month. Proposed for approval.",
    }
    bad = {
        "total_monthly_waste_usd": 99.99,
        "top_action": {"resource": "EBS vol-2", "monthly_cost_usd": 0.80, "fix": "Delete it"},
        "ranked_actions": [],
        "exec_brief": "The unused Elastic IP has been released successfully.",
    }
    g = check(report, good)
    b = check(report, bad)
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