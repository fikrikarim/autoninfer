# Autoninfer handover

**Last updated:** 2026-08-18 ~13:02 UTC by unattended driver session `autoninfer-driver-8` — **backed off, no experiment this iteration** (interactive-session collision rule, BLOCKERS standing notes). State below.

## State
- **HEAD:** `28795af6` (docs commit on top of the attention committed-column bit-clone `e84ddaef`), pushed. Tree was clean at 28795af6 until the interactive session dirtied it.
- **Baseline M1 (keep/discard reference):** `130.13 ± 0.14 tok/s, 38.4% accept` at k=3 (tg128, `--lm-head-draft`, INT8 KV, menu command), set at e84ddaef.
- **Working tree is DIRTY — owned by the live interactive pi session** (pid 21662, pts/6, up since 08:22; last file write 12:54). Its WIP **is the previous handover's next experiment** (k=1 flip attribution), and it is further along than a driver iteration could get. **Do not build, measure, edit, or commit on this tree while that session is alive** (single-writer design; it rebuilds `build/` live — last repo rebuild 12:53:57, so `build/` binaries currently contain its WIP state).

## Serve state (do not touch)
- The live harness serve is a **manual** process (pid 21537, started 08:20 by the interactive session) on GPU 0: `ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft` — note it lacks `--pending-timeout-ms 300000`. Supervisor `ninfer-serve` is STOPPED; the manual process is what powers the harness.
- `/tmp/autoninfer-ops/pending.json` holds `{"action":"restart-primary"}` (queued 08:45). The driver applies it between iterations (deferred while an interactive pi session is alive), so expect the **supervisor** serve (wrapper flags incl. `--pending-timeout-ms 300000`) to come up after my iteration. Verify at start: `supervisorctl status ninfer-serve` + `curl -s http://127.0.0.1:8080/v1/models`. Do NOT re-queue it.

## The interactive session's WIP — the k=1 residual is now ATTRIBUTED
Its diagnosis (from its diff + `/tmp` evidence): the k=0-vs-k>0 stream residual is **case (a)** of the previous handover — a residual committed-column defect in the **FP8 linear small-t family**: the NVFP4 family received the T=1 4-chain association clone in `5cbba58c`, while the FP8 family kept the one-chain small-t association (vpl=16) plus an A16 island over its A8 T=1 reference (its own words, header of `tests/ops/test_fp8_linear_t_parity.cpp`). `/tmp/opdump_{old,new}.txt` shows T=2..4 output hashes diverging from T=1 on the FP8 linear/linearadd/linear_swiglu ops; `m_{old,new}_{a,b,c}.*` is an AIME e2e matrix and `lh_{old,new}.err` are `NINFER_DEBUG_LAYERHASH` per-layer hidden-hash dumps (decode runs 150–158) pinning the last divergent op.

Uncommitted files (its WIP — leave them all alone):
- `src/ops/linear/fp8/fp8_config.h`, `fp8_dispatch.cpp`, `src/ops/linear_add/fp8/fp8_linear_add_small_t.cu`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp` — T=2..4 small-t schedules become the T=1 GEMV bit-clone (vpl=8, four interleaved accumulator chains, same pair-to-chain mapping); A8 parents (FP8 MLP gate/up) stay A8 at all T (A8 MMA is T-independent per column).
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` — TEMP per-layer hidden-hash diagnostic, env-gated (`NINFER_DEBUG_LAYERHASH`), marked "not for commit" — it must be stripped before commit; flag it in BLOCKERS if it ever leaks into a commit.
- `tests/ops/test_fp8_linear_t_parity.cpp` (new, the real test: T=1..4 per-column bit parity, A16 + A8 families, real model shapes) and `tests/ops/test_tmp_binary_diff.cpp` (new, temp op-dump harness — must not be committed), `tests/CMakeLists.txt`, `tests/ops/linear/test_fp8_a16.cpp`, `tests/ops/linear/test_fp8_a8.cpp` (schedule expectations).
- `/tmp` evidence (recreate if wiped): `ninfer_old` / `ninfer_new` / `ninfer_new2` builds, `opdump_*.txt`, `m_*.err`, `lh_*.err`, `matrix.sh`.

## Next iteration, in order
1. **Check whether the interactive session committed or left:** `ps aux | grep -w 21662 || ps aux | grep " pi$" | grep -v grep`, then `git status --short` and `git log --oneline -3`.
   - **Still alive with a dirty tree → back off again exactly as this iteration did**: refresh this handover from fresh `/tmp` evidence, commit ONLY the handover (+ BLOCKERS if needed), push, end. Do not touch the tree.
   - **Committed → verify its commit**: contains the FP8 fix + `test_fp8_linear_t_parity`; does NOT contain the TEMP layer-hash instrumentation or `test_tmp_binary_diff` (grep `NINFER_DEBUG_LAYERHASH`, `test_tmp_binary_diff` in the new tree). If they leaked in, record a BLOCKERS row; do not silently rewrite another session's commit.
2. **Clean rebuild:** `cmake --build build -j` on the now-clean tree (build/ currently holds the WIP state).
3. **Measure at the new HEAD** (FP8 committed-column clone in): M1 menu command (k=3) — expect the accept rate to shift again (the committed column is now a bit-clone across the linear family too, like the attention fix's +8.2pp). New baseline row. Then the backlog **k=2 vs k=3 re-decision** (now valid — pre-fix A/B numbers are stale): same menu command with `--mtp-draft-tokens 2` vs `3`.
4. **Quality gate:** `tools/autoninfer/quality_gate.sh pre-<exp>` / `post-<exp>` around any further change; after the FP8 fix expect the k=0-vs-k=3 gate diff to shrink below the current 4/8, and the AIME k=1 idx-119 flip (280 vs 343) to disappear — verify by re-running the k=0 and k=1 AIME commands from the previous handover and diffing the token-id lines.
5. Backlog after that: prefill FP8 crossovers (backlog #2), host-side round overhead (backlog #5).

## Do not repeat / do not touch
- The k=1 flip attribution — done by the interactive session (FP8 linear small-t T=2..4 association + A16 island). Its evidence is in `/tmp` as listed above.
- Building/measuring/committing on the dirty tree while the interactive pi session is alive.
- The manual serve (pid 21537) and GPU 0 in any way (bind/kill/restart/reconfigure).
- Starting the standby serve. Re-queueing `restart-primary` (already pending). Committing `.env` (git-ignored, EXA key).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark (HEALTHY at 12:54 this iteration: triad 1466 GiB/s, FMA 110.7 TFLOP/s).