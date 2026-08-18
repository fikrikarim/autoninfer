# Autoninfer handover

**Last updated:** 2026-08-18 ~13:01 UTC by unattended driver session `autoninfer-driver-10` — **backed off (3rd consecutive), no experiment this iteration** (interactive-session collision rule). State below.

## State
- **HEAD:** `5102bfa1` (iteration 9's backoff docs commit on top of `d0869112`), pushed.
- **Baseline M1 (keep/discard reference):** `130.13 ± 0.14 tok/s, 38.4% accept` at k=3 (tg128, `--lm-head-draft`, INT8 KV, menu command), set at e84ddaef.
- **Working tree is DIRTY — still owned by live interactive pi sessions.** At this iteration's check (13:01):
  - **pid 21662** (pts/6, up since 08:22, 27:21 CPU — frozen across a 12:59→13:00 sample, i.e. idle) and **pid 237820** (pts/5, started 12:55:16, ~2 min CPU) are both alive but both CPU-idle right now — likely mid-turn or awaiting the user; still owners by the single-writer rule.
  - `git status` identical to iteration 9's list (same 8 modified files + 2 new test files). No new files, no partial commits.
  - **Do not build, measure, edit, or commit on this tree while the interactive sessions are alive** (single-writer design; `build/` binaries currently contain the WIP state, last repo rebuild 12:53:57).

## Fresh /tmp evidence (read at 13:01, supersedes iteration 9's)
- `lh_old.err` (12:54:08) / `lh_new.err` (12:52:31) (NINFER_DEBUG_LAYERHASH AIME dumps): the runs **completed** (both files end with the summary block; 1143 `LH` lines each, run=150..158). **Still diverging:** 2190 `diff` lines (~1095 differing layer hashes) starting at layer 4 (mix+mlp) and spanning the remaining layers. The WIP FP8 fix was **not verified clean** as of completion.
- **No /tmp writes after 12:54:10** — `m_*.err`, `opdump_old.txt`, `matrix.sh`, `ninfer_old`/`ninfer_new`/`ninfer_new2` all unchanged since iteration 9.
- Nothing in /tmp or the tree suggests the FP8 fix was committed, rolled back, or abandoned.

## Serve state (do not touch)
- The live harness serve is the **manual** process (pid 21537, started 08:20) on GPU 0: `ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft` (lacks `--pending-timeout-ms 300000`). Supervisor `ninfer-serve` is STOPPED; the manual process powers the harness.
- `/tmp/autoninfer-ops/pending.json` holds `{"action":"restart-primary"}` (queued 08:45, deferred while interactive sessions are alive). Expect the **supervisor** serve (wrapper flags incl. `--pending-timeout-ms 300000`) to come up once no interactive session is alive. Verify at start: `supervisorctl status ninfer-serve` + `curl -s http://127.0.0.1:8080/v1/models`. Do NOT re-queue it.

## The interactive session's WIP (unchanged file list, leave it all alone)
- `src/ops/linear/fp8/fp8_config.h`, `fp8_dispatch.cpp`, `src/ops/linear_add/fp8/fp8_linear_add_small_t.cu`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp` — T=2..4 small-t schedules become the T=1 GEMV bit-clone (vpl=8, four interleaved accumulator chains, same pair-to-chain mapping); A8 parents (FP8 MLP gate/up) stay A8 at all T.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` — TEMP `NINFER_DEBUG_LAYERHASH` per-layer hidden-hash diagnostic, marked "not for commit"; must be stripped before commit.
- `tests/ops/test_fp8_linear_t_parity.cpp` (new, the real test: T=1..4 per-column bit parity), `tests/ops/test_tmp_binary_diff.cpp` (new, temp op-dump harness — must not be committed), `tests/CMakeLists.txt`, `tests/ops/linear/test_fp8_a16.cpp`, `tests/ops/linear/test_fp8_a8.cpp` (schedule expectations).

## Next iteration, in order
1. **Check whether the interactive sessions committed or left:** `ps -o pid,etime,cputime -p 21662,237820` (plus `ps aux | grep -w " pi"` for any new one), then `git status --short` and `git log --oneline -3`.
   - **Any interactive pi alive with a dirty tree → back off again exactly as iterations 8–10 did** (refresh this handover from fresh `/tmp` evidence, commit ONLY the handover + BLOCKERS state if it moved, push, end). Do not touch the tree. Note: at 13:01 both were CPU-idle — if they stay idle with no /tmp/tree writes for several more iterations, they may be abandoned terminals; the backoff rule still applies while they are alive (check for their parent tty/ssh session too).
   - **Committed → verify its commit:** contains the FP8 fix + `test_fp8_linear_t_parity`; does NOT contain `NINFER_DEBUG_LAYERHASH` or `test_tmp_binary_diff` (grep the new tree). If they leaked in, record a BLOCKERS row; do not silently rewrite another session's commit.
   - **Sessions gone, tree still dirty (abandoned WIP):** the driver may now take over — but treat the uncommitted diff as someone else's half-finished experiment: build it, run `test_fp8_linear_t_parity`, re-run the AIME k=0/k=1 diff (the `m_*.err` pattern / `matrix.sh` in /tmp), and only commit it if the evidence shows the fix is correct and the TEMP instrumentation is stripped; otherwise `git stash` it with a descriptive message and proceed to backlog #2, noting the stash in the handover.
2. **Clean rebuild** `cmake --build build -j` once the tree is clean/owned (build/ currently holds the WIP state).
3. **Measure at the new HEAD** (FP8 committed-column clone in): M1 menu command (k=3) — expect the accept rate to shift (compare to the attention fix's +8.2pp). New baseline row. Then the backlog **k=2 vs k=3 re-decision** (pre-fix A/B numbers are stale): same menu command with `--mtp-draft-tokens 2` vs `3`.
4. **Quality gate:** `tools/autoninfer/quality_gate.sh pre-<exp>` / `post-<exp>` around any further change; after the FP8 fix expect the k=0-vs-k=3 gate diff to shrink below the current 4/8 and the AIME k=1 idx-119 flip (280 vs 343) to disappear — verify by re-running the k=0 and k=1 AIME commands and diffing the token-id lines.
5. Backlog after that: prefill FP8 crossovers (backlog #2), host-side round overhead (backlog #5).

## Do not repeat / do not touch
- The k=1 flip attribution — owned by the interactive sessions (FP8 linear small-t T=2..4 association + A16 island). Their evidence is in `/tmp` as listed above; their WIP was **not verified clean at completion** (13:01: finished dumps still show ~1095 layer-hash diffs).
- Building/measuring/committing on the dirty tree while any interactive pi session is alive.
- The manual serve (pid 21537) and GPU 0 in any way (bind/kill/restart/reconfigure).
- Starting the standby serve. Re-queueing `restart-primary` (already pending). Committing `.env` (git-ignored, EXA key).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark (last known HEALTHY at 12:57:22: triad 1466 GiB/s, FMA 110.7 TFLOP/s; GPU 1 idle per the 13:01 snapshot).