#!/usr/bin/env python
"""Refuse downstream arrays unless the preceding immutable gate passed."""

import hashlib
import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
record = json.loads(path.read_text())
if record.get("status") != "passed":
    raise SystemExit(f"Gate did not pass: {path}")
required = {"gate", "status", "commit", "evidence", "created_at"}
missing = required - record.keys()
if missing:
    raise SystemExit(f"Gate record is missing {sorted(missing)}")
digest = hashlib.sha256(
    json.dumps(record["evidence"], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if digest != record.get("evidence_sha256"):
    raise SystemExit("Gate evidence hash does not match")
print(f"gate={record['gate']} status=passed evidence_sha256={digest}")
