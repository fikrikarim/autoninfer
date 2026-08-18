#!/bin/bash
# Fetch full text for one or more URLs (EXA /content API), truncated per page.
#
# Usage: tools/autoninfer/exa_content.sh <url> [url ...]
#
# For reading a promising paper/blog hit found via exa_search.sh. Each page's
# text is truncated to ~8000 chars (enough to extract the method; use the URL
# directly for the full document). The API key comes from .env or the
# environment (EXA_API_KEY).
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: exa_content.sh <url> [url ...]" >&2; exit 2; }

envfile="$(cd "$(dirname "$0")/../.." && pwd)/.env"
if [ -z "${EXA_API_KEY:-}" ] && [ -f "$envfile" ]; then
  EXA_API_KEY=$(sed -n 's/^EXA_API_KEY=//p' "$envfile" | head -1)
fi
[ -n "${EXA_API_KEY:-}" ] || { echo "EXA_API_KEY not set (.env or environment)" >&2; exit 2; }

body=$(python3 -c "import json,sys; print(json.dumps({'ids': sys.argv[1:], 'type': 'text', 'maxCharacters': 8000}))" "$@")
resp=$(curl -sS -m 60 https://api.exa.ai/contents -H "x-api-key: $EXA_API_KEY" -H "Content-Type: application/json" -d "$body")

python3 - "$resp" <<'EOF'
import json, sys
d = json.loads(sys.argv[1])
for r in d.get("results", []):
    url = r.get("url") or "?"
    text = " ".join((r.get("text") or "").split())
    print(f"===== {url} =====")
    if not text:
        print("(no text returned)")
    else:
        print(text[:8000])
        if len(text) > 8000:
            print("... [truncated]")
    print()
EOF