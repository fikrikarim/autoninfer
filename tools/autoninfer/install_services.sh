#!/bin/bash
# Install the autoninfer supervisor services (primary serve, standby serve, driver).
#
# Run after a fresh instance / recycle, once /workspace/autoninfer is cloned and the
# model artifact is in place (models/qwen3_8_27b_nvfp4.ninfer - git-ignored local
# prerequisite; restore it from wherever it was sourced). The primary serve
# (autostart) starts immediately and comes back on its own after every future
# instance restart.
#
# This is the single command that reconstructs the whole serving layer:
#   bash tools/autoninfer/install_services.sh
#
# Idempotent: re-running it re-syncs the wrappers/configs and re-registers the
# services without disturbing a healthy running primary (it restarts it only if
# the wrapper contents changed).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$REPO/tools/autoninfer/supervisor"

# 1. Wrappers -> /opt/supervisor-scripts (detect a live primary with changed flags)
primary_running=0
if supervisorctl status ninfer-serve 2>/dev/null | grep -q RUNNING; then
  primary_running=1
fi
wrapper_changed=0
if [ "$primary_running" = 1 ] && ! cmp -s "$SRC/ninfer-serve.sh" /opt/supervisor-scripts/ninfer-serve.sh 2>/dev/null; then
  wrapper_changed=1
fi
install -m 0755 "$SRC/ninfer-serve.sh" /opt/supervisor-scripts/ninfer-serve.sh
install -m 0755 "$SRC/ninfer-serve-standby.sh" /opt/supervisor-scripts/ninfer-serve-standby.sh

# 2. Service configs -> /etc/supervisor/conf.d
for conf in ninfer-serve ninfer-serve-standby autoninfer-driver; do
  install -m 0644 "$SRC/$conf.conf" "/etc/supervisor/conf.d/$conf.conf"
done

# 3. Register with supervisor
supervisorctl reread >/dev/null
supervisorctl update

# 4. Apply new flags to a running primary only when its wrapper actually changed
if [ "$primary_running" = 1 ] && [ "$wrapper_changed" = 1 ]; then
  echo "primary wrapper changed - restarting ninfer-serve to apply new flags"
  supervisorctl restart ninfer-serve
fi

echo "autoninfer services installed:"
supervisorctl status | grep -E "ninfer-serve|autoninfer-driver" || true
echo
echo "Note: the pi provider symlink is repo-tracked but the link itself is not:"
echo "  ln -sfn $REPO/.pi/models.json /root/.pi/agent/models.json"