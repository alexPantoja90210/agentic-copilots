"""
Install the dependency chain over SSM.

Each node serves what it managed to FETCH FROM ITS UPSTREAM. The tail serves a
payload it generates locally; everyone else caches what they pulled, and empties
that cache when the pull fails.

That last clause is the whole design, and the first version got it wrong. It had
every node serve a locally generated file and throw the pulled bytes away, which
looked identical while healthy and broke the moment it mattered: stopping the
tail collapsed its immediate consumer and left the node behind it serving its own
file, perfectly happy. The failure propagated one hop and stopped -- so the
two-hop attribution the pilot exists to test could not happen. Pull, cache,
serve: a node with nothing to serve is a node whose upstream is gone, which is
what a dependency actually means.

Why bytes and not something richer: EC2 basic monitoring publishes every five
minutes and knows nothing about applications. A steady byte rate is the one
signal unambiguous at that resolution -- flat while healthy, near zero the
moment the upstream stops. An HTTP service returning 500s would be more
realistic and completely invisible to CloudWatch, which would leave the incident
untestable.

Installed over SSM rather than user_data because user_data would REPLACE the
existing instance, which IA-46 pinned specifically to prevent. As systemd units
with `enable`, so a node that comes back from an injection rejoins on its own
(IA-55 criterion 5).

Idempotent: run it as often as you like. It rewrites the units and restarts
them, which is also how you repair a chain someone has poked at.

    python chain_bootstrap.py --injector-role-arn <arn>
"""

from __future__ import annotations

import argparse
import sys
import time

import chain_topology as ct
from inject import InjectionError, session

PORT = 8080
PAYLOAD_BYTES = 262144      # 256 KiB
PULL_INTERVAL_S = 5         # ~50 KB/s steady, ~15 MB per 5-minute datapoint

SERVE_UNIT = """[Unit]
Description=IA-55 pilot payload server
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 -m http.server {port} --directory /opt/pilot
Restart=always
RestartSec=5
User=nobody

[Install]
WantedBy=multi-user.target
"""

# The pulled bytes become what this node serves. On failure the cache is removed,
# so the next consumer down the chain gets a 404 and its own traffic collapses:
# the fault travels the whole graph instead of stopping at the first hop.
PULL_UNIT = """[Unit]
Description=IA-55 pilot dependency pull ({role} -> {upstream})
After=network-online.target

[Service]
ExecStart=/bin/bash -c 'while true; do if curl -s --max-time 3 -o /opt/pilot/payload.next http://{upstream_ip}:{port}/payload.bin && [ -s /opt/pilot/payload.next ]; then mv -f /opt/pilot/payload.next /opt/pilot/payload.bin; else rm -f /opt/pilot/payload.next /opt/pilot/payload.bin; fi; sleep {interval}; done'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

INSTALL = """set -e
mkdir -p /opt/pilot
{seed}
cat > /etc/systemd/system/pilot-serve.service <<'UNIT'
{serve_unit}UNIT
{pull_block}
systemctl daemon-reload
systemctl enable --now pilot-serve.service
{pull_enable}
systemctl is-active pilot-serve.service
"""


# Read-only. Reports whether this node caches what it pulls (has an upstream)
# or originates its own payload (is the tail).
VERIFY = r"""
if grep -q 'payload.next' /etc/systemd/system/pilot-pull.service 2>/dev/null; then echo -n "cache=yes "; else echo -n "cache=no "; fi
if [ -f /etc/systemd/system/pilot-pull.service ]; then echo -n "puller=yes "; else echo -n "puller=no "; fi
if systemctl is-active --quiet pilot-serve.service; then echo -n "serving=yes "; else echo -n "serving=no "; fi
if [ -s /opt/pilot/payload.bin ]; then echo -n "has_payload=yes "; else echo -n "has_payload=no "; fi
# The tail is the only node whose install line generates bytes locally.
if [ ! -f /etc/systemd/system/pilot-pull.service ]; then echo -n "seed=yes"; else echo -n "seed=no"; fi
echo
"""


def build_script(role: str, upstream_ip: str | None, upstream_role: str | None) -> str:
    pull_block, pull_enable = "", ""
    if upstream_ip:
        unit = PULL_UNIT.format(role=role, upstream=upstream_role,
                                upstream_ip=upstream_ip, port=PORT,
                                interval=PULL_INTERVAL_S)
        pull_block = ("cat > /etc/systemd/system/pilot-pull.service <<'UNIT'\n"
                      + unit + "UNIT\n")
        pull_enable = ("systemctl enable --now pilot-pull.service\n"
                       "systemctl restart pilot-pull.service\n"
                       "systemctl is-active pilot-pull.service")
    if upstream_ip:
        # Has an upstream: starts with nothing. Serving a locally made file here
        # is exactly the defect this version removes -- it would keep the node
        # alive while its dependency was dead.
        seed = ("rm -f /opt/pilot/payload.bin\n"
                "# no local payload: this node serves only what it fetches.")
    else:
        # The tail. Somebody has to originate the bytes, and generating them
        # locally keeps the rate independent of the internet.
        seed = ("head -c %d /dev/urandom > /opt/pilot/payload.bin" % PAYLOAD_BYTES)

    return INSTALL.format(seed=seed,
                          serve_unit=SERVE_UNIT.format(port=PORT),
                          pull_block=pull_block, pull_enable=pull_enable)


def run_on(ssm, instance_id: str, script: str, timeout_s: int = 180) -> str:
    sent = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="IA-55 chain bootstrap",
        Parameters={"commands": [script]},
        TimeoutSeconds=timeout_s,
    )
    command_id = sent["Command"]["CommandId"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        inv = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            if inv["Status"] != "Success":
                raise InjectionError(
                    "%s: SSM ended as %s: %s"
                    % (instance_id, inv["Status"],
                       (inv.get("StandardErrorContent") or "")[:400]))
            return (inv.get("StandardOutputContent") or "").strip()
    raise InjectionError("%s: SSM command did not finish within %ds" % (instance_id, timeout_s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install the IA-55 dependency chain.")
    ap.add_argument("--injector-role-arn", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the script for each node without sending anything")
    ap.add_argument("--verify", action="store_true",
                    help="read back what is installed on each node, change nothing")
    args = ap.parse_args(argv)

    sess = session(args.injector_role_arn)
    ec2, ssm = sess.client("ec2"), sess.client("ssm")

    nodes = ct.discover(ec2)
    print(ct.describe(nodes))
    print()

    if args.verify:
        # Read back the installed units rather than inferring health from
        # traffic. The two designs -- serve-your-own-file and serve-what-you-
        # fetched -- produce IDENTICAL byte rates while everything is up. They
        # differ only when an upstream dies, which is the one experiment we do
        # not want to spend on finding out whether the install took.
        failures = 0
        for role in sorted(nodes):
            node = nodes[role]
            if node["state"] != "running":
                print(f"{role}: {node['state']} — cannot read")
                failures += 1
                continue
            out = run_on(ssm, node["instance_id"], VERIFY)
            caches = "cache=yes" in out
            seeds = "seed=yes" in out
            expected_cache = node["depends_on"] is not None
            ok = (caches == expected_cache) and (seeds == (not expected_cache))
            failures += not ok
            print("%-4s %s  %s" % (
                role, "OK  " if ok else "WRONG",
                out.replace("\n", "  ")))
        print()
        if failures:
            print("The installed units do not match the declared topology.")
            return 1
        print("Every node serves what its position in the graph says it should:")
        print("  the tail originates bytes; everyone else serves only what they fetched.")
        return 0

    stopped = sorted(r for r, n in nodes.items() if n["state"] != "running")
    if stopped and not args.dry_run:
        raise InjectionError(
            "these nodes are not running: %s. The chain cannot be installed on a "
            "stopped instance, and a partially installed chain would produce a "
            "topology that is real in Terraform and false in the metrics."
            % ", ".join(stopped))

    # Deepest first. Installing a puller before its upstream serves would make
    # the first minutes of traffic a record of the bootstrap, not of the system.
    order = sorted(nodes, key=lambda r: len(ct.chain_from(r, nodes)))
    for role in order:
        node = nodes[role]
        upstream_role = node["depends_on"]
        upstream_ip = nodes[upstream_role]["private_ip"] if upstream_role else None
        script = build_script(role, upstream_ip, upstream_role)

        if args.dry_run:
            print(f"--- {role} ({'serves only' if not upstream_role else 'serves and pulls from ' + upstream_role})")
            print(script)
            continue

        print(f"installing on {role} ...", end=" ", flush=True)
        out = run_on(ssm, node["instance_id"], script)
        print(out.replace("\n", " ") or "ok")

    if not args.dry_run:
        print()
        print("chain installed. Give it ~10 minutes before reading CloudWatch:")
        print("  basic monitoring publishes every 5 minutes, so a steady rate")
        print("  needs at least two datapoints to be visible as steady.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
