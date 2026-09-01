"""
Verify the injector role's policy against what the harness actually needs.

Why this exists: the first dry run of inject.py failed because the role could
not call ec2:DescribeInstanceCreditSpecifications — the policy was blocking the
harness's own safety check. Fixing that revealed a second gap, ssm:SendCommand,
which the dry run can never reach because a rehearsal skips the call. A real
injection would have hit it mid-window, with the ground-truth label already
written and the window wasted.

A rehearsal that cannot exercise a permission cannot verify it. This asks IAM
directly, through simulate-principal-policy, so every action the harness uses is
checked without any of them being performed.

It is written to be able to fail in both directions: actions the harness needs
must be allowed, and actions it must never have must be denied. A check that can
only pass is not a check.

Run as yourself, not as the injector role: simulating a policy requires
iam:SimulatePrincipalPolicy, which the injector deliberately does not have.

    python verify_policy.py --role-arn <arn> --instance i-...
"""

from __future__ import annotations

import argparse
import sys

# Everything inject.py calls, and why. If a call is added to the harness and
# not to this list, the list is wrong — that is the drift this file exists to
# catch, so keep them together.
#
# The third field is the resource to simulate against, and it is not cosmetic.
# Actions that support resource-level permissions must be simulated against the
# instance ARN, because that is where the policy's constraint lives. Actions
# that do NOT support them — every ec2:Describe*, the CloudWatch reads,
# ssm:GetCommandInvocation — must be simulated against "*": handing the
# simulator a specific ARN for such an action returns implicitDeny even when
# the policy grants it on "*", and the report then accuses a policy that is
# perfectly correct.
#
# The first version of this file chose the resource by SERVICE, sending every
# ec2 and ssm action at the instance ARN. It reported five false failures. They
# were caught because reality had already answered: a dry run under this same
# role had resolved the instance seconds earlier, so ec2:DescribeInstances was
# demonstrably allowed. A verifier that disagrees with an observed fact is
# wrong until proven otherwise — it is not evidence against the thing it
# measures.
INSTANCE, ANY = "instance", "*"

MUST_ALLOW = {
    "ec2:DescribeInstances": (ANY, "resolve the target and read its state"),
    "ec2:DescribeInstanceStatus": (ANY, "wait for running/stopped transitions"),
    "ec2:DescribeInstanceCreditSpecifications": (ANY, "refuse a CPU fault under 'unlimited'"),
    "ec2:DescribeTags": (ANY, "confirm the pilot tag"),
    "ec2:StartInstances": (INSTANCE, "restore the instance after F3"),
    "ec2:StopInstances": (INSTANCE, "F3, the unavailability fault"),
    "ssm:SendCommand": (INSTANCE, "F1/F2, the CPU load generator"),
    "ssm:GetCommandInvocation": (ANY, "read how the load run ended"),
    "cloudwatch:GetMetricStatistics": (ANY, "build the metric window"),
    "cloudwatch:ListMetrics": (ANY, "discover what the target emits"),
}

# The role must NOT be able to do these. Termination is the one that matters:
# an experiment that can destroy its own subject is not an experiment.
MUST_DENY = {
    "ec2:TerminateInstances": (INSTANCE, "destroy the pilot target"),
    "ec2:RunInstances": (ANY, "create new spend"),
    "ec2:CreateTags": (INSTANCE, "re-tag its way into a wider grant"),
    "iam:PutRolePolicy": (ANY, "widen its own policy"),
    "s3:GetObject": (ANY, "reach data unrelated to the pilot"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the injector role against what the harness needs.")
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args(argv)

    import boto3

    iam = boto3.client("iam", region_name=args.region)
    account = boto3.client("sts", region_name=args.region).get_caller_identity()["Account"]
    instance_arn = f"arn:aws:ec2:{args.region}:{account}:instance/{args.instance}"

    def decision(action: str, scope: str) -> str:
        resource = instance_arn if scope == INSTANCE else "*"
        res = iam.simulate_principal_policy(
            PolicySourceArn=args.role_arn,
            ActionNames=[action],
            ResourceArns=[resource],
            ContextEntries=[{
                "ContextKeyName": "aws:ResourceTag/Pilot",
                "ContextKeyValues": ["IA-45"],
                "ContextKeyType": "string",
            }],
        )
        return res["EvaluationResults"][0]["EvalDecision"]

    rows, failures = [], 0

    for action, (scope, why) in sorted(MUST_ALLOW.items()):
        got = decision(action, scope)
        ok = got == "allowed"
        failures += not ok
        rows.append(("PASS" if ok else "FAIL", "allow", action, got, why))

    for action, (scope, why) in sorted(MUST_DENY.items()):
        got = decision(action, scope)
        ok = got != "allowed"
        failures += not ok
        rows.append(("PASS" if ok else "FAIL", "deny ", action, got, why))

    w = max(len(a) for _, _, a, _, _ in rows)
    for status, kind, action, got, why in rows:
        print(f"{status:4}  {kind}  {action:<{w}}  {got:<20}  {why}")

    print(f"\n{len(rows) - failures}/{len(rows)} expectations met")
    if failures:
        print("\nThe policy and the harness disagree. Fix pilot.tf before injecting anything.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
