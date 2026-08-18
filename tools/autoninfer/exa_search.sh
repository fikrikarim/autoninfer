#!/bin/bash
# Web search for the autoninfer research loop (EXA API).
#
# Usage: tools/autoninfer/exa_search.sh "<query>" [numResults=5]
#
# Prints compact results: title / url / ~400-char text snippet. The API key is
# read from the repository's git-ignored .env (EXA_API_KEY) or the environment.
# Designed for an agent to run inside an iteration: keep research to at most a
# couple of searches per iteration and log genuinely new actionable findings to
# docs/autoninfer/inspiration.md (deduplicated).
set -euo pipefail

query="${1:?usage: exa_search.sh \"<query>\" [numResults]}"
num="${2:-5}"

envfile="$(cd "$(dirname "$0")/../.." && pwd)/.env"
if [ -z "${EXA_API_KEY:-}" ] && [ -f "$envfile" ]; then
  EXA_API_KEY=$(sed -n 's/^EXA_API_KEY=//p' "$envfile" | head -1)
fi
[ -n "${EXA_API_KEY:-}" ] || { echo "EXA_API_KEY not set (.env or environment)" >&2; exit 2; }

body=$(python3 -c "import json,sys; print(json.dumps({'query': sys.argv[1], 'numResults': int(sys.argv[2]), 'contents': {'text': {'maxCharacters': 400}}}))" "$query" "$num")
resp=$(curl -sS -m 30 https://api.exa.ai/search -H "x-api-key: $EXA_API_KEY" -H "Content-Type: application/json" -d "$body")

python3 - "$resp" <<'EOF'
import json, sys
d = json.loads(sys.argv[1])
results = d.get("results") or []
if not results:
    print("(no results)")
for i, r in enumerate(results, 1):
    title = (r.get("title") or "").strip()
    url = r.get("url") or r.get("id") or "?"
    text = " ".join((r.get("text") or "").split())
    print(f"[{i}] {title}")
    print(f"    {url}")
    if text:
        print(f"    {text[:400]}")
    print()
cost = d.get("costDollars")
if cost:
    print(f"(exa cost: ${cost.get('total', '?')})")
EOF