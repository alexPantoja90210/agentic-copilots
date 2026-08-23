"""
Eval harness for FinOps Copilot - the differentiator.
Scores the agent on: contract-ok, total-correct, top-correct, no-hallucination, policy-ok.
GREEN only if every hard check passes (your go/no-go gate).
  python evals.py            # run against the live agent (needs ANTHROPIC_API_KEY)
  python evals.py --selftest # validate the checks with no LLM/API calls (free)
"""
import glob
import json
import sys

from report_contract import validate_report

TOL = 0.01

# Dos familias de checks, y conviene no confundirlas:
#   - CONTRACT_CHECKS miden la ENTRADA (el report.json que produce guardian.py)
#   - PLAN_CHECKS miden la SALIDA (el plan que produjo el agente)
# Un plan malo debe fallar los PLAN_CHECKS, pero no tiene por que afectar al
# contrato: el reporte puede ser impecable y el plan pesimo. Distinguirlas es
# lo que permite que el selftest siga siendo honesto tras anadir el contrato.
CONTRACT_CHECKS = ("contract-ok",)
PLAN_CHECKS = ("total-correct", "top-correct", "no-hallucination", "policy-ok")
FORBIDDEN = ["has been released", "was released", "has been deleted", "was deleted",
             "successfully deleted", "successfully released", "i released", "i deleted",
             "action completed", "resource removed"]

def reference_plan(report):
    waste = report.get("waste", [])
    ranked = sorted(waste, key=lambda w: w.get("est_monthly_usd", 0), reverse=True)
    total = round(sum(w.get("est_monthly_usd", 0) for w in waste), 2)
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
    source_costs = [round(w.get("est_monthly_usd", 0), 2) for w in report.get("waste", [])]
    source_costs.append(ref["total"])
    results = {}
    # Va primero a proposito: si la entrada no cumple el contrato, el resto de
    # los checks miden sobre datos que no son los que el productor emite hoy.
    # Este es el check que IA-26 anadio — el que vigila la frontera entre
    # guardian.py y copilot.py, que antes nadie miraba.
    results["contract-ok"] = not validate_report(report)
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
    print("case                       contract total  top   halluc policy")
    for path in fixtures:
        # Se valida ANTES de cargar: copilot.load_report revienta si el reporte
        # no cumple el contrato, y un gate debe reportar en ROJO, no explotar.
        with open(path) as f:
            report = json.load(f)
        problems = validate_report(report)
        if problems:
            name = path.split("/")[-1][:26].ljust(26)
            print(name + " FAIL" + "    -     -     -     -")
            for p in problems:
                print("      " + p)
            all_pass = False
            continue
        report = copilot.load_report(path)
        plan = copilot.plan(write_files=False)
        r = check(report, plan)
        all_pass = all_pass and all(r.values())
        mark = lambda b: " PASS" if b else " FAIL"
        name = path.split("/")[-1][:26].ljust(26)
        print(name + mark(r["contract-ok"]) + mark(r["total-correct"]) + mark(r["top-correct"]) +
              mark(r["no-hallucination"]) + mark(r["policy-ok"]))
    print("\n" + ("GREEN - all checks passed" if all_pass else "RED - fix before shipping"))
    return 0 if all_pass else 1

def selftest():
    report = {
        "generated_at": "2026-08-23T00:00:00+00:00",
        "forecast": {"mtd": 0.0, "avg_daily": 0.0, "projected_eom": 0.0,
                     "budget": 5.0, "status": "OK", "days": "23/31", "data_ok": True},
        "waste": [
            {"resource": "Elastic IP eipalloc-0abc123", "type": "unused_elastic_ip",
             "detail": "unattached", "est_monthly_usd": 3.65, "action": "Release the unused Elastic IP"},
            {"resource": "EBS volume vol-0def456", "type": "unattached_ebs",
             "detail": "8 GiB gp3", "est_monthly_usd": 0.80, "action": "Delete or snapshot the unattached volume"},
        ],
        "waste_monthly_usd": 4.45,
        "health_score": 82,
    }
    ref = reference_plan(report)
    good = {
        "total_monthly_waste_usd": ref["total"],
        "top_action": {"resource": ref["top"]["resource"], "monthly_cost_usd": 3.65, "fix": "Release it"},
        "ranked_actions": [{"resource": w["resource"], "monthly_cost_usd": w["est_monthly_usd"],
                            "fix": w["action"]} for w in ref["ranked"]],
        "exec_brief": "I recommend releasing the unused Elastic IP to save $3.65/month. Proposed for approval.",
    }
    bad = {
        "total_monthly_waste_usd": 99.99,
        "top_action": {"resource": "EBS volume vol-0def456", "monthly_cost_usd": 0.80, "fix": "Delete it"},
        "ranked_actions": [],
        "exec_brief": "The unused Elastic IP has been released successfully.",
    }
    # Tercer caso, anadido en IA-26: un reporte con el ESQUEMA VIEJO.
    # El agente nunca deberia verlo, y el gate tiene que decirlo.
    legacy = {
        "generated_at": "2026-08-19",
        "forecast": 0.00,
        "budget": 5.00,
        "budget_status": "on_track",
        "health_score": 82,
        "waste": [
            {"resource": "Elastic IP eipalloc-0abc123", "type": "unused_elastic_ip",
             "monthly_cost": 3.65, "fix": "Release the unused Elastic IP"},
        ],
    }

    g = check(report, good)
    b = check(report, bad)
    legacy_problems = validate_report(legacy)

    assert all(g.values()), "self-test failed: good plan should pass -> " + str(g)
    assert all(g[k] for k in CONTRACT_CHECKS), "self-test failed: good report should satisfy the contract"
    assert not any(b[k] for k in PLAN_CHECKS), \
        "self-test failed: bad plan should fail every plan check -> " + str(b)
    assert b["contract-ok"], \
        "self-test failed: the report is valid, so contract-ok must stay True even with a bad plan"
    assert legacy_problems, "self-test failed: legacy schema should violate the contract"

    print("self-test OK: good plan passes every check; bad plan fails every PLAN check;")
    print("              a valid report keeps contract-ok True; the legacy schema is rejected.")
    print("  good  :", g)
    print("  bad   :", b)
    print("  legacy: contract violations ->")
    for p in legacy_problems:
        print("            - " + p)
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_live())