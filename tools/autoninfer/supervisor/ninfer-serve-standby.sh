#!/bin/bash
# Standby ninfer-serve (GPU 1, 127.0.0.1:8081, model id qwen3.8-27b-standby).
#
# Bridge serve used ONLY during primary restarts: the pi session is switched to
# this model (autoninfer_standby extension tool), the primary on GPU 0 is
# restarted, then the session switches back and this serve stops. It is not
# autostarted and must not be left running: it occupies ~30.5 GiB of GPU 1 and
# would collide with research workloads.
#
# The distinct --model-id makes misrouted requests fail loudly (the serve
# validates the request model field and 404s on mismatch).
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/autoninfer
exec ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
    --host 127.0.0.1 --port 8081 --device 1 \
    --model-id qwen3.8-27b-standby \
    --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
    --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft