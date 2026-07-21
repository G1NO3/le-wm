#!/usr/bin/env python
"""Create a hashed gate record from an evidence JSON file."""

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gate", required=True)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--passed", action="store_true")
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f"Refusing to overwrite immutable gate record: {args.output}")
evidence = json.loads(args.evidence.read_text())
digest = hashlib.sha256(
    json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
record = {
    "gate": args.gate,
    "status": "passed" if args.passed else "failed",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "evidence": evidence,
    "evidence_sha256": digest,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
