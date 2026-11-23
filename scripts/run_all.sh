#!/usr/bin/env bash
# Full pilot: tune on dev -> run on test (unknown count, then oracle count) -> score -> cross-check -> docs
set -euo pipefail
cd "$(dirname "$0")/../eval"
PY=../.venv/bin/python
for s in wespeaker titanet eres2net; do $PY tune.py --system $s; done
for s in wespeaker titanet eres2net; do $PY run.py --system $s; done
for s in wespeaker titanet; do $PY run.py --system $s --oracle-count; done

$PY score.py
$PY crosscheck.py
cd .. && $PY docs/build.py
