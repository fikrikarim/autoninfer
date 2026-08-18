# Autoninfer handover

**Last updated:** 2026-08-18 13:22 UTC by unattended driver session (iteration 28) — **backed off (21st consecutive), no experiment this iteration** (interactive-session collision rule). State below.

## State
- **HEAD:** `6fb5df5c` (iteration 27's backoff docs commit); this iteration adds a backoff docs commit on top.
- **Baseline M1 (keep/discard reference):** `130.13 ± 0.14 tok/s, 38.4% accept` at k=3 (tg128, `--lm-head-draft`, INT8 KV, menu command), set at e84ddaef.
- **Working tree is DIRTY — still owned by live interactive pi sessions.** At this iteration's check (13:20):
  - **pid 21662** (pts/6, up 04:57, 27:21 CPU — flat, idle) and **pid 237820** (pts/5, up 25:24, 0:02 CPU — idle) are both alive.
  - Parent chain still `herdr server` — ttys are daemon-held and survive user disconnect; pids staying alive is the signal.
  - **WIP diff fingerprint (13:20) — compare before touching anything:** `git diff | sha256sum` = `d49b4f84…bb705` (8 modified + 2 new test files, +105/−41) — **byte-identical to iterations 11–28; nothing in the tree has changed since 12:54:09** (WIP file mtimes re-verified 13:20: `text_context_impl.h` 12:50:36, `fp8_config.h`/`fp8_dispatch.cpp` 12:54:09.132, `fp8_linear_add_small_t.cu`/`fp8_linear_swiglu_plan.cpp` 12:54:09.136, `test_fp8_linear_t_parity.cpp` 12:06:23, `test_tmp_binary_diff.cpp` 12:47:39, `test_fp8_a16.cpp`/`test_fp8_a8.cpp` 12:47:49).

## Fresh /tmp evidence (read at 13:20, supersedes iteration 27's)
- **Last interactive write anywhere: still 12:54:09.** The `lh_*_iter28` entries (13:20) are driver scratch, not interactive writes; newest /tmp files before those are iter-27's scratch at 13:19.
- **Layer-hash dump state, re-counted this iteration (regex-extract `LH run=(\d+) (mix|mlp) layer=(\d+) h=([0-9a-f]+)` from `lh_old.err` and `lh_new.err`, paste-join in file order, compare 5-field halves):** both dumps 1152 tuples; **1088 differing, 64 matching; first differing tuple: 0-indexed position 8 = run=150 mix layer=4** (old h=`478a57cad0f3a017` vs new h=`4a44cf19f097a851`). Unchanged since iteration 15. The WIP FP8 fix was **not verified clean** at completion, unchanged since 12:54:09.

## Budget note
Driver budget is 30 h wall clock from ~09:00 UTC (2026-08-18) → clean exit around ~15:00 UTC (~1 h 39 min remaining at 13:22). If the interactive sessions stay alive, remaining iterations continue to back off in docs-only mode (that is by design; the driver stops cleanly on budget, and `supervisorctl start autoninfer-driver` re-arms it if the user wants more time).

## Serve state (do not touch)
- The live harness serve is the **manual** process (**pid 21537, up 05:00 at 13:20**) on GPU 0: `ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft` (lacks `--pending-timeout-ms 300000`). Supervisor `ninfer-serve` is STOPPED (since 08:30); `ninfer-serve-standby` STOPPED (since 08:44). `/v1/models` → 200 at 13:20.
- `/tmp/autoninfer-ops/pending.json` still holds `{"action":"restart-primary"}` (queued 08:45, deferred while interactive sessions are alive; re-verified 13:20). Expect the **supervisor** serve (wrapper flags incl. `--pending-timeout-ms 300000`) to come up once no interactive session is alive. Verify at start: `supervisorctl status ninfer-serve` + `curl -s http://127.0.0.1:8080/v1/models`. Do NOT re-queue it.
- GPU 1 idle. No GPU work ran this iteration (backoff), so no fresh `tools/gpu_health.sh 1` stamp.

## The interactive session's WIP (unchanged since 12:54:09, leave it all alone)
- `src/ops/linear/fp8/fp8_config.h`, `fp8_dispatch.cpp`, `src/ops/linear_add/fp8/fp8_linear_add_small_t.cu`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp` — T=2..4 small-t schedules become the T=1 GEMV bit-clone (vpl=8, four interleaved accumulator chains, same pair-to-chain mapping); A8 parents (FP8 MLP gate/up) stay A8 at all T.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` — TEMP `NINFER_DEBUG_LAYERHASH` per-layer hidden-hash diagnostic, marked "not for commit"; must be stripped before commit.
- `tests/ops/test_fp8_linear_t_parity.cpp` (new, the real test: T=1..4 per-column bit parity), `tests/ops/test_tmp_binary_diff.cpp` (new, temp op-dump harness — must not be committed), `tests/CMakeLists.txt`, `tests/ops/linear/test_fp8_a16.cpp`, `tests/ops/linear/test_fp8_a8.cpp` (schedule expectations).

## Next iteration, in order
1. **Check whether the interactive sessions are alive:** `ps -o pid,etime,cputime,stat -p 21662,237820` (plus `ps aux | grep -w "pi"` — exclude the driver's own session via the parent chain: its grandparent is `drive.sh` under `supervisord`), then `git status --short`, `git log --oneline -3`, and `git diff | sha256sum` (expect `d49b4f84…bb705` if the WIP is untouched).
   - **Any interactive pi alive with a dirty tree → back off again exactly as iterations 8–28 did** (refresh this handover from fresh `/tmp` evidence — last-write scan `ls -lat /tmp | head -15`, `stat` the WIP files, re-count layer-hash diffs: `grep -oE 'LH run=[0-9]+ (mix|mlp) layer=[0-9]+ h=[0-9a-f]+' /tmp/lh_old.err` / same for `lh_new.err`, then `paste` the two extracts and compare the two 5-field halves per row — 1088/64 with first diff at 0-indexed 8 (line 9) = run=150 mix layer=4 (old `478a57cad0f3a017` vs new `4a44cf19f097a851`) is the current expectation — commit ONLY the handover + BLOCKERS state if it moved, push, end). Do not touch the tree. Remember: driver scratch files in /tmp (e.g. `lh_*.iterNN.txt`, `lh_pasted.iterNN.txt`) are not interactive writes; check mtimes against 12:54:09.
   - **Committed → verify its commit:** contains the FP8 fix + `test_fp8_linear_t_parity`; does NOT contain `NINFER_DEBUG_LAYERHASH` or `test_tmp_binary_diff` (grep the new tree). If they leaked in, record a BLOCKERS row; do not silently rewrite another session's commit.
   - **Sessions gone, tree still dirty (abandoned WIP):** the driver may now take over — but treat the uncommitted diff as someone else's half-finished experiment: build it, run `test_fp8_linear_t_parity`, re-run the AIME k=0/k=1 diff (the `m_*.err` pattern / `matrix.sh` in /tmp), and only commit it if the evidence shows the fix is correct and the TEMP instrumentation is stripped; otherwise `git stash` it with a descriptive message and proceed to backlog #2, noting the stash in the handover.
2. **Clean rebuild** `cmake --build build -j` once the tree is clean/owned (build/ currently holds the WIP state).
3. **Measure at the new HEAD** (FP8 committed-column clone in): M1 menu command (k=3) — expect the accept rate to shift (compare to the attention fix's +8.2pp). New baseline row. Then the backlog **k=2 vs k=3 re-decision** (pre-fix A/B numbers are stale): same menu command with `--mtp-draft-tokens 2` vs `3`.
4. **Quality gate:** `tools/autoninfer/quality_gate.sh pre-<exp>` / `post-<exp>` around any further change; after the FP8 fix expect the k=0-vs-k=3 gate diff to shrink below the current 4/8 and the AIME k=1 idx-119 flip (280 vs 343) to disappear — verify by re-running the k=0 and k=1 AIME commands and diffing the token-id lines.
5. Backlog after that: prefill FP8 crossovers (backlog #2), host-side round overhead (backlog #5). **Web-research status (iter 17): all five backlog hypotheses already have concrete seeded entries in `docs/autoninfer/inspiration.md`** (H2's vLLM NVFP4 tile-sweep template is a ready-made experiment for backlog #2) — no new research needed before the next experiment; search only if a measurement is surprising.

## Do not repeat / do not touch
- The k=1 flip attribution — owned by the interactive sessions (FP8 linear small-t T=2..4 association + A16 island). Their evidence is in `/tmp` as listed above; their WIP was **not verified clean at completion** (13:20: 1088/1152 layer-hash tuples still differ, first at line 9 = run=150 mix layer=4; no new interactive writes since 12:54:09).
- Building/measuring/committing on the dirty tree while any interactive pi session is alive.
- The manual serve (pid 21537) and GPU 0 in any way (bind/kill/restart/reconfigure).
- Starting the standby serve. Re-queueing `restart-primary` (already pending). Committing `.env` (git-ignored, EXA key).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark. No GPU work ran this iteration (backoff #21).
- Speculative web-research surveys while blocked: the backlog is seeded; research only feeds a live decision.