# Autoninfer handover

**Last updated:** 2026-08-18 13:45 UTC by the user's interactive session (acting for the user).
The tree is **clean** and the loop may proceed normally — no backoff state exists.

## State
- **HEAD:** `f8b9880b` (a driver docs commit). Working tree clean; `build/` is current with this
  tree (rebuilt after the final strip).
- **The FP8 committed-column fix is committed at `f58db502`** and op-level verified:
  `ninfer_fp8_linear_t_parity_test` PASS (T=1..4 per-column bit parity vs the T=1 GEMV route),
  `ninfer_linear_fp8_a16`/`a8` PASS with the updated schedule expectations. What it changed:
  T=2..4 FP8 linear/linear_add/swiglu small-T schedules are now the T=1 GEMV bit-clone
  (vpl=8, four chains, same pair-to-chain mapping); the MLP gate/up parent uses A8 at all token
  counts (T=1 decode route). T>=5 keeps the measured winners.
- **Baseline M1 (keep/discard reference):** `130.13 ± 0.14 tok/s, 38.4% accept` at k=3
  (tg128, `--lm-head-draft`, INT8 KV, menu command) — measured **before** f58db502; the accept
  rate may shift now that verification is bit-exact with decode.
- **Serve:** manual serve pid 21537 (up since ~08:30) on GPU 0, running the pre-f58db502 binary
  in memory; supervisor `ninfer-serve`/standby STOPPED. `/v1/models` 200. **Do not touch it.**
- **Pending op:** `/tmp/autoninfer-ops/pending.json` holds `restart-primary` — the driver
  applies it (supervisor serve with `--pending-timeout-ms 300000` + the current build) once no
  interactive session is alive. **Do not re-queue it.** It will pick up f58db502 at that point.

## Next iteration, in order (this is the whole iteration's plan)
1. `bash tools/gpu_health.sh 1` (gate).
2. **End-to-end ratification of f58db502** (the active BLOCKERS.md follow-up):
   - Quality gate: `tools/autoninfer/quality_gate.sh post-fp8col` (k=3, 8 prompts, greedy) and
     diff against the recorded k=0 reference (`quality_gate pre-*` rows exist in results.tsv;
     regenerate the k=0 gate if the recorded one is stale): expect the diff to drop below the
     4/8 of 13:00. The AIME k=1 idx-119 flip (280 vs 343) must disappear — verify by re-running
     the AIME k=0 and k=1 commands (menu in docs/autoninfer/README.md) and diffing token-id
     lines.
   - If the flip **persists**: the NVFP4 side of the verification path may have the same
     committed-column defect (gdn/attention linears at T=2..4) — attribute layer-by-layer before
     touching schedules again; record findings in BLOCKERS.md.
3. **Re-measure M1** (menu command, k=3) at the new HEAD → new baseline row in results.tsv.
4. **Re-decide k=2 vs k=3** (backlog; the pre-fix A/B numbers are stale).
5. Then backlog #2 (prefill FP8 crossovers; vLLM NVFP4 tile-sweep template in inspiration.md is
   a ready-made experiment).

## Do not repeat / do not touch
- The iteration 8–39 backoff spiral: it was caused by a self-propagating handover rule ("back off
  while the interactive session is alive") with no takeover path, and by misreading the driver's
  MAX_ITER cap as a time budget ("~1h34m to budget exit"). Both are now fixed in the driver
  prompt (drive.sh): at most 2 consecutive backoffs on the same dirt, then take over / stash /
  escalate; MAX_ITER (now 400) is a safety cap, not a clock. Do not propagate a backoff
  instruction into this handover.
- The manual serve (pid 21537) and GPU 0 in any way (bind/kill/restart/reconfigure).
- Starting the standby serve. Re-queueing `restart-primary` (already pending). Committing
  `.env` (git-ignored, EXA key).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark.
- The `/tmp` layer-hash evidence (`lh_*.err`, `m_*.err`, driver `lh_*.iterNN.txt` scratch) is
  the pre-fix diagnostic for the now-committed f58db502 — historical, safe to ignore.