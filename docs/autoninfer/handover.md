# Autoninfer handover

**Last updated:** 2026-08-18 ~18:35 UTC by unattended driver session (new protocol in force).
**This iteration's result: DSpark experimental lane phase 0 — the 2.72 GiB section artifact exists, bit-exact verified; converter + engine-side design doc committed. Next: Op 2 + draft-KV arena.**

## HEAD & state
- **HEAD:** `8339476258c755848be7408daa7e02449c12aa80` (this iteration: `feat(autoninfer)` DSpark lane converter + lane doc + results row + handover). Tree clean.
- **Baseline M1 (keep/discard reference, unchanged):** `113.01 ± 0.05 tok/s, 30.15% accept` (k=3, 201 rounds, committed-source binary, 82ef8337).
- **DSpark checkpoint: COMPLETE and verified.** `models/dspark/model.safetensors` = 2,718,576,122 B (file = 8 + 6640 header + 2,718,569,474 data span exactly), 62 BF16 tensors, sha256 `9d26d5e637551c24...` (full digest in the conversion report). Finished 17:42 — the 17:45 handover's "1.19/2.55 GiB, resuming" note was stale; nothing was dead, the ledger entry was just lost in the 17:40 driver restart (bg ledger now clean/empty).
- **Section artifact:** `out/dspark_27b.ninfer` — 2,718,577,666 B, 47 objects, identity `(qwen3.6-27b, dspark-bf16)`, recipe `qwen3_6_27b-dspark-bf16-v1`; **round-trip verified bit-exact (47/47)** in 18 s; report `out/dspark_27b.ninfer.conversion.json`. `out/` + `models/` are git-ignored — never commit them.
- **Python:** project tools interpreter = `/venv/main/bin/python3` (Python 3.12, torch 2.13.0+cpu). **numpy 2.5.2 was installed into /venv/main this iteration** (task-required by the converter's import chain; system python3 has numpy but no torch).
- **Interactive session pid 21662:** still alive but dormant (0 CPU jiffies over 4 s at 18:20; last repo write 18:16). It authored the DSpark WIP I committed this iteration (converter @17:56, lane doc @18:16) after its own 17:45 handover directed the driver to this work. My only edits to its files: two arithmetic fixes `46` → `47` objects (converter comment + lane doc §1) — the inventory is 2 + 5×8 + 5 = 47 (preflight asserts 47; the "46" was stale). If the user resumes that session, the diff is trivially visible.
- **Scratch intact** (/tmp survived the 17:40 driver restart): `/tmp/wt` (gdn-gating fix worktree, discarded chain), `/tmp/tap_fixed_full.diff`, `/tmp/it9_*` (tap A/B + bisect evidence), `/tmp/base_k0.err` etc.
- **Stashes (do not drop):** `stash@{0}` = iter-14 rx dump tap (orphaned); `stash@{1}` = iter-9 layer/logit tap (F32→FP32-fixed version recoverable via `/tmp/tap_fixed_full.diff`).
- **Serve (do not touch):** the manual process (pid 21537 since 15:17, no `--pending-timeout-ms`) still powers /v1 (qwen3.8-27b). Supervisor `ninfer-serve` is **FATAL** (exited too quickly — the applied `restart-primary` op raced the manual serve for 127.0.0.1:8080; `/tmp/autoninfer-ops/restart-primary.done` marker present). The new wrapper flags are therefore NOT live. Resolving the manual-vs-supervisor serve is a driver/user concern, not research work.
- **GPU 1:** idle at 18:25 (0%, 4 MiB). `bash tools/gpu_health.sh 1` before any benchmark.

## What was done (one result, per the progress invariant)
The losslessness chain is at a YIELD (step-back rule, 17:45 handover; product decision filed in BLOCKERS at 17:45, open). Executed the default: **yield to DSpark (backlog #1)** and produced its first lane result:
1. Step 0: `bg.sh check` — empty ledger; verified the DSpark weights had actually **completed** (safetensors header + exact size, 62 tensors). No dead job to restart.
2. Ran the interactive session's converter (handed-over WIP): `CUDA_VISIBLE_DEVICES=1 /venv/main/bin/python3 -m tools.convert.qwen3_6_27b.dspark --model-dir models/dspark --out out/dspark_27b.ninfer` → 47/47 objects written + bit-exact round-trip verify, 18 s.
3. Fixed the stale 46→47 object count (2 places), committed the converter (`tools/convert/qwen3_6_27b/dspark.py`) + engine-side design doc (`docs/maintainer/qwen3.8-27b-dspark-lane.md`) + results row.
The lane doc is now the **engine-side authority** for the DSpark backend: persistent draft-KV arena (20,480 B/token BF16, linear slot-local, monotonic append, capture-stable), ops 1–5 contracts with FP64 oracles, fit model (23.04 + 13.77·ctx/262144 GiB; fits 16384..131072 on the 32 GiB 5090; 262144 no), round schedule, and the gate plan. MTP remains canonical; DSpark adoption needs BLOCKERS ratification after measurement.

## Next iteration (single hypothesis: DSpark Op 2 + persistent draft-KV arena)
Scope: exactly the state-transition core — no target wiring, no CLI, no M1 yet (the round is not runnable until ops 3/4 + `speculative_round` exist).
1. Read, in order: `docs/maintainer/op-development.md` (admission rules), `docs/maintainer/qwen3.8-27b-dspark-lane.md` §3 (arena) + §4 Op 2 (contract/oracle), and the 35B-A3B DFlash op family under `src/ops/` (block-verify skeleton template; `grep -rn "dflash" src/ops --include=*.h -l`).
2. Implement the arena: per-slot linear BF16, 20,480 B/token (5 layers × 8 KV heads × 128 dim × K+V × 2 B), capacity startup-fixed at `max_context` (335 MiB at menu ctx 16384); offset `(p, l) = p * 40960 + l * 8192` B per lane doc §3; monotonic append (positions commit in increasing order, written exactly once — no crop, no zero-init, no page table); address capture-stable.
3. Implement `dspark_ctx_commit`: `x = rmsnorm_1e-6(fc(taps [T,25600]))`; per layer `K = rope_yarn(rmsnorm_head128(k_proj(x)), pos)`, `V = v_proj(x)`; append at absolute positions. T ≤ 8 (verify) and T = chunk (prefill first build). YaRN cos/sin table precomputed once at load, bit-validated against the transformers-5.12.1 `rope_parameters` (models/dspark/config.json). Oracle: the exact formula in FP64 from the BF16 taps (lane doc §4 Op 2; the oracle does not copy kernel staging).
4. Op-level test at real shapes (T=1..8 verify rows; T=128/1024 prefill builds; positions across the RoPE table range), independent FP64 oracle, run on GPU 1 (`CUDA_VISIBLE_DEVICES=1`, after `bash tools/gpu_health.sh 1`). Build: `cmake --build build -j`.
5. results.tsv row: op-level oracle GREEN + arena/workspace cost at the menu ctx; **keep** (lane infrastructure; M1 not yet comparable).

## Do not repeat / do not touch
- Re-downloading or re-verifying the DSpark weights (complete + verified; sha in the conversion report). Re-running the converter is fine if the artifact goes missing (18 s), but the report is the evidence of record.
- The losslessness rx-anomaly hunt (chain parked on the user's ratification of the 17:45 BLOCKERS row; evidence preserved in /tmp + the 5d4d6298 row).
- The stale "download resuming" state (it completed at 17:42).
- The manual serve (pid 21537) and GPU 0 in any way; the FATAL supervisor `ninfer-serve` — do not `supervisorctl` it from research work. No standby serve.
- Committing `.env`, `models/`, `out/`. Dropping the two stashes.
- Editing the DSpark files if a new interactive session has started writing them (check `ps -eo pid,ppid,stat,etime,cmd | grep -w pi` excluding drive.sh descendants + mtimes first).
- Wiring DSpark into the target/CLI or claiming an M1 number before ops 3/4 + the speculative_round draft-count extension (≤5 → 7) exist; any M1 comparison for adoption is at the **matched** fitting context (lane doc §5/§7, ground rule 9).