# Autoninfer handover

**Last updated:** 2026-08-18 ~13:01 UTC by unattended driver session `autoninfer-driver-11` — **backed off (4th consecutive), no experiment this iteration** (interactive-session collision rule). State below.

## State
- **HEAD:** `83281ada` (iteration 10's backoff docs commit on top of `5102bfa1`), pushed.
- **Baseline M1 (keep/discard reference):** `130.13 ± 0.14 tok/s, 38.4% accept` at k=3 (tg128, `--lm-head-draft`, INT8 KV, menu command), set at e84ddaef.
- **Working tree is DIRTY — still owned by live interactive pi sessions.** At this iteration's check (13:00):
  - **pid 21662** (pts/6, up since 08:22, 27:21 CPU — unchanged since 13:00 sample of iter 10, i.e. idle) and **pid 237820** (pts/5, started 12:55:16, 0:02 CPU — idle) are both alive.
  - **NEW (this iteration): checked the parent chain.** Both bash shells (3785 on pts/6, 11899 on pts/5) are children of **`/root/.local/bin/herdr server` (pid 3505, Ssl, up 6h25m)** — *not* sshd. The ttys are daemon-held and **survive user disconnect**, so "abandoned terminal" is not a safe assumption and there is no process-death signal to wait for. The backoff rule applies for as long as the pi pids live; only the user (or a session crash) changes that.
  - `git status` identical to iterations 9–10 (same 8 modified files + 2 new test files); WIP diff verified unchanged (spot-checked `fp8_config.h`: T≤4 → vpl=8, 4 chains, bit-clone of T=1 GEMV; matches the description below).
  - **Do not build, measure, edit, or commit on this tree while the interactive sessions are alive** (single-writer design; `build/` binaries currently contain the WIP state, last repo rebuild 12:53:57).

## Fresh /tmp evidence (read at 13:00, supersedes iteration 10's)
- **Last interactive write anywhere: 12:54:09** — four FP8 WIP source files (`fp8_config.h`, `fp8_dispatch.cpp`, `fp8_linear_add_small_t.cu`, `fp8_linear_swiglu_plan.cpp`) mtime 12:54:09, and the final `/tmp` dump `lh_old.err`/`lh_old.out` at 12:54:08. Nothing in the tree or /tmp has changed since (checked 13:00).
- `lh_old.err` (12:54:08) / `lh_new.err` (12:52:31) (NINFER_DEBUG_LAYERHASH AIME dumps): runs **completed** (summary block present; 1143 `LH` lines each, run=150..158). **Still diverging:** 2190 `diff` lines (~1095 differing layer hashes) from layer 4 (mix+mlp) onward. The WIP FP8 fix was **not verified clean** at completion.
- All other /tmp artifacts (`m_*.err`, `opdump_*.txt`, `matrix.sh`, `ninfer_old`/`ninfer_new`/`ninfer_new2`) unchanged since iteration 9 (≤12:49).

## Serve state (do not touch)
- The live harness serve is the **manual** process (pid 21537, started 08:20) on GPU 0: `ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft` (lacks `--pending-timeout-ms 300000`). Supervisor `ninfer-serve` is STOPPED; the manual process powers the harness (verified 13:00: `/v1/models` → 200).
- `/tmp/autoninfer-ops/pending.json` still holds `{"action":"restart-primary"}` (queued 08:45, deferred while interactive sessions are alive; re-verified 13:00). Expect the **supervisor** serve (wrapper flags incl. `--pending-timeout-ms 300000`) to come up once no interactive session is alive. Verify at start: `supervisorctl status ninfer-serve` + `curl -s http://127.0.0.1:8080/v1/models`. Do NOT re-queue it.

## The interactive session's WIP (unchanged since 12:54:09, leave it all alone)
- `src/ops/linear/fp8/fp8_config.h`, `fp8_dispatch.cpp`, `src/ops/linear_add/fp8/fp8_linear_add_small_t.cu`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp` — T=2..4 small-t schedules become the T=1 GEMV bit-clone (vpl=8, four interleaved accumulator chains, same pair-to-chain mapping); A8 parents (FP8 MLP gate/up) stay A8 at all T.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` — TEMP `NINFER_DEBUG_LAYERHASH` per-layer hidden-hash diagnostic, marked "not for commit"; must be stripped before commit.
- `tests/ops/test_fp8_linear_t_parity.cpp` (new, the real test: T=1..4 per-column bit parity), `tests/ops/test_tmp_binary_diff.cpp` (new, temp op-dump harness — must not be committed), `tests/CMakeLists.txt`, `tests/ops/linear/test_fp8_a16.cpp`, `tests/ops/linear/test_fp8_a8.cpp` (schedule expectations).

## Next iteration, in order
1. **Check whether the interactive sessions are alive:** `ps -o pid,etime,cputime -p 21662,237820` (plus `ps aux | grep -w " pi"` for any new one), then `git status --short` and `git log --oneline -3`.
   - **Any interactive pi alive with a dirty tree → back off again exactly as iterations 8–11 did** (refresh this handover from fresh `/tmp` evidence — last-write scan `ls -lat /tmp | head`, `stat` the WIP files — commit ONLY the handover + BLOCKERS state if it moved, push, end). Do not touch the tree. Since the ttys are daemon-held (`herdr server`), don't wait for process death as a signal; the pids staying alive IS the signal.
   - **Committed → verify its commit:** contains the FP8 fix + `test_fp8_linear_t_parity`; does NOT contain `NINFER_DEBUG_LAYERHASH` or `test_tmp_binary_diff` (grep the new tree). If they leaked in, record a BLOCKERS row; do not silently rewrite another session's commit.
   - **Sessions gone, tree still dirty (abandoned WIP):** the driver may now take over — but treat the uncommitted diff as someone else's half-finished experiment: build it, run `test_fp8_linear_t_parity`, re-run the AIME k=0/k=1 diff (the `m_*.err` pattern / `matrix.sh` in /tmp), and only commit it if the evidence shows the fix is correct and the TEMP instrumentation is stripped; otherwise `git stash` it with a descriptive message and proceed to backlog #2, noting the stash in the handover.
2. **Clean rebuild** `cmake --build build -j` once the tree is clean/owned (build/ currently holds the WIP state).
3. **Measure at the new HEAD** (FP8 committed-column clone in): M1 menu command (k=3) — expect the accept rate to shift (compare to the attention fix's +8.2pp). New baseline row. Then the backlog **k=2 vs k=3 re-decision** (pre-fix A/B numbers are stale): same menu command with `--mtp-draft-tokens 2` vs `3`.
4. **Quality gate:** `tools/autoninfer/quality_gate.sh pre-<exp>` / `post-<exp>` around any further change; after the FP8 fix expect the k=0-vs-k=3 gate diff to shrink below the current 4/8 and the AIME k=1 idx-119 flip (280 vs 343) to disappear — verify by re-running the k=0 and k=1 AIME commands and diffing the token-id lines.
5. Backlog after that: prefill FP8 crossovers (backlog #2), host-side round overhead (backlog #5).

## Do not repeat / do not touch
- The k=1 flip attribution — owned by the interactive sessions (FP8 linear small-t T=2..4 association + A16 island). Their evidence is in `/tmp` as listed above; their WIP was **not verified clean at completion** (13:00: finished dumps still show ~1095 layer-hash diffs; no new writes since 12:54:09).
- Building/measuring/committing on the dirty tree while any interactive pi session is alive.
- The manual serve (pid 21537) and GPU 0 in any way (bind/kill/restart/reconfigure).
- Starting the standby serve. Re-queueing `restart-primary` (already pending). Committing `.env` (git-ignored, EXA key).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark (last known HEALTHY at 12:57:22: triad 1466 GiB/s, FMA 110.7 TFLOP/s; GPU 1 idle per the 13:00 snapshot). No GPU work ran this iteration (backoff).