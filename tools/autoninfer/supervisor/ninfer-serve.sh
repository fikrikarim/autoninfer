#!/bin/bash
# Primary ninfer-serve (GPU 0, 127.0.0.1:8080, model id qwen3.8-27b).
#
# This is the serve that powers the pi harness itself. It is supervisor-managed
# (autostart, autorestart) and intentionally has NO /etc/portal.yaml entry: it
# listens on localhost only and is never exposed through Caddy.
#
# Flags are the canonical live-serve configuration (see docs/autoninfer/README.md,
# "Serve management"). The standby twin is ninfer-serve-standby.sh.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/autoninfer
exec ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
    --host 127.0.0.1 --port 8080 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
    --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft