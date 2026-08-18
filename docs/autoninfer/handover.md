# Autoninfer handover

**Last updated:** 2026-08-18 ~17:26 UTC by unattended driver session `autoninfer-driver-9` (iteration 9/400).
**Experiment this iteration: k0@136 canonical-flip diagnosis (gdn-gating fix re-land decision). Outcome: DISCARD confirmed, root cause localized to the GDN state pool — not the prefill forward.**

## HEAD & state
- **HEAD:** `7ab7b4da` + this iteration's docs commit (results row + handover; nothing else). Tree clean. `git stash list` has **one entry** (see below).
- **Baseline M1 (keep/discard reference, unchanged):** `113.01 ± 0.05 tok/s, 30.15% accept` (k=3, 201 rounds, committed-source binary, re-baselined at 82ef8337).
- GPU 1 HEALTHY at 16:40 (triad 1466.7 GiB/s, FMA 110.71 TFLOP/s). All GPU work ran on GPU 1.
- `/tmp/autoninfer-ops/pending.json` still holds `{"action":"restart-primary"}` (queued 08:45; driver defers while interactive sessions are alive). Do NOT re-queue.
- Serve: untouched. Interactive session pid 21662 was still alive (dormant: 0 CPU jiffies/4s at check) — it powers no work of mine; do not kill it.

## What was done (one experiment: diagnose the k0 canonical flip)
The gdn-gating T=2..8 bit-clone (uncommitted, in scratch worktree `/tmp/wt` at f2dbd156) flipped the canonical k=0 AIME stream at idx 136 (base=13, fix=318). The previous handover's prime suspect (arena layout shift from the removed T=2..8 workspace) — **falsified and localized this iteration**:

1. **Took over the abandoned tap WIP** (per protocol: owner session 267414 gone, 65+ min no writes, tree blocked builds with a compile error). The 2-file layer/logit tap (`src/targets/qwen3_6/impl/runtime/text_context{,_impl}.h`, self-marked "STRIP BEFORE COMMIT", env-gated `NINFER_TAP_LOGITS` + `NINFER_TAP_LAYERS_LO/HI`, inert when unset) was **stashed verbatim as `stash@{0}`**, restored into the worktree, and its `DType::F32` typo fixed → `DType::FP32` (enum has FP32; the handover's "use BF16" note was wrong — the lambda is a generic element-size). Full fixed-tap diff saved at `/tmp/tap_fixed_full.diff`. **Tap stripped from the main tree again after the experiment** (tree clean; session's original WIP intact in the stash; recover fixed version with `git stash pop` + the one-line sed, or `git apply /tmp/tap_fixed_full.diff` after a clean checkout).
2. **Built both A/B binaries:** main tree (base + fixed tap) `build/apps/ninfer` (~6 min incremental) and `/tmp/wt` (fix + fixed tap) `/tmp/wt/build/apps/ninfer` (~10 min). **Note: the /tmp/wt build now CONTAINS the tap** (inert without the env vars — same as main).
3. **Tap inertness verified:** base+tap binary, k0 AIME, no env vars → first 160 tokens bit-identical to canonical `/tmp/base_k0.err` stream.
4. **Tap A/B (AIME k0, 256 tok, eager):** first divergence = the state tap line at the **first decode position** — GDN **recurrent state `rx` (layer 0, slot 0) already differs after PREFILL**, before any T=1 decode round. `cx` (conv state), the embedding, and the L0 q/k/v/g/beta columns are bit-identical; then every layer residual and the logits cascade (logits ~3 BF16 ulp off; argmax matches until idx 136).
5. **Length bisect** (truncated AIME prefixes, `/tmp/it9_p{06,12,18,32,64,130}.json`): post-prefill `rx` **diverges at every prefill length tested** (56, 59, 64, 76, 99, 162, 228 tok — single chunks, T∈[9,1024] Split8 route = unchanged code). ⇒ the anomaly is **not** in the prefill forward (all prefill ops: identical machine code, identical inputs by induction; T=1 GEMV bit-invariant — parity hash `5a60bebb10800c96` on both builds).
6. **Arena hypothesis falsified empirically:** load summaries byte-identical (arena 172.57 MiB, reservation 1006.12 MiB, identical allocation lines); CLI default `max_concurrency=1` registers no (2..8) workspace range, so the fix's capacity change is a no-op for these runs.

**Verdict:** the fix **stays DISCARDED** (a losslessness-series fix that moves the canonical k=0 stream can't be kept). The remaining suspect class is the **GDN state pool materialization**: rx content differs while cx (same pool, same write path class) matches, at every length, deterministic per build, eager+graph — pointing at zero-init coverage, a partial-write region, or residual device-memory contents from a load-time allocation sequence that differs between builds despite identical final sizes.

## Next iteration (single hypothesis: resolve the state-pool rx anomaly)
1. **Raw rx value dump** (pattern + magnitude of the diff; distinguishes uninitialized/partial-write region from a 1-ulp write difference):
   - Add an env-gated raw dump to the tap: in `tap_layer_select` (text_context_impl.h, the `ST` block ~line 790), when `NINFER_TAP_STATE_DUMP=<file>` is set, `fwrite` the full `rx` tensor bytes (and `cx` for reference) for the first selected position. Apply to the main tree (`git apply /tmp/tap_fixed_full.diff` then edit), rebuild main (~6 min), rebuild /tmp/wt (copy the same 2 files, ~10 min).
   - Run: `CUDA_VISIBLE_DEVICES=1 NINFER_TAP_LOGITS=/dev/null NINFER_TAP_LAYERS_LO=0 NINFER_TAP_LAYERS_HI=1000 NINFER_TAP_STATE_DUMP=/tmp/it10_rx_{b,f}.bin <bin> models/qwen3_8_27b_nvfp4.ninfer --messages /tmp/it9_p06.json --max-context 16384 --kv-dtype int8 --greedy --seed 42 --max-new 2 --no-cuda-graph` on both binaries.
   - `cmp -l /tmp/it10_rx_b.bin /tmp/it10_rx_f.bin | head -50` → which bytes/rows/heads differ and by how much (rx is FP32: heads×dim×4 B; check whether differing elements are clustered at the tail = never-written init region).
2. **State pool init/zeroing code read** (in parallel with the builds): the GDN state pool allocation + zero-init (src/core arena / state transaction in src/targets/qwen3_6 family) and the `gated_delta_net` prefill transition's write coverage (does the prefill transition write ALL of rx, or only touched rows/heads?). The fix removed a 30,720 B capacity for T=2..8 — check whether any load-time pass (warmup/graph-capture/kernel-JIT sample) still allocates it in base and not in fix, leaving the freed region's residual contents to bleed into the state pool.
3. **Then:** if the anomaly is a benign init/coverage defect in the state path → fix the state path (not the gdn op), re-land the gdn fix from `/tmp/wt` (4 files; strip the TEMP_T1REF block from `tests/ops/test_gdn_gating_proj_t_parity.cpp` first), verify M1 + k0 (must reproduce base_k0 bit-exact) + k1 (flip should move off 119), quality gate pre/post, results row. If the anomaly is a real numerical defect → fix it as its own experiment.
4. Stale backlog after that: k=2 vs k=3 re-decision (pre-fix numbers void), prefill FP8 crossovers (#2), host-side round overhead (#5).

## /tmp evidence (this iteration; recreate nothing)
- `it9_tap_base.txt` / `it9_tap_fix.txt` (+ `*_run.err`): AIME 256-tok tap A/B (first divergence = ST line). `it9_tap_diff.txt`: the diff.
- `it9_p{06,12,18,32,64,130}.json` + `it9_p*_b.txt`/`_f.txt`: length bisect (all rx=DIV, cx=OK).
- `it9_k0_plain.err`: base+tap inertness run. `b_ids.txt`/`n160.txt`/`f160.txt`: token streams (first diff idx 136).
- `tap_fixed_full.diff`: the fixed tap (session WIP + F32→FP32). `build_main.log`/`build_wt.log`: build logs.
- Pre-existing (iteration 8): `base_k0{,_r2,_r3}.{out,err}`, `gdnfix_k0{,_eager}.{out,err}`, `m_*.{out,err}`, `lh_*`, `opdump_*`, `matrix.sh`, `ninfer_{old,new,new2}`. `/tmp/wt` = scratch worktree (fix 4 files + tap; build current).

## Do not repeat / do not touch
- Re-running the AIME tap A/B or the length bisect (evidence preserved above); re-verifying T=1 GEMV bit-invariance (proven both builds); the arena-layout hypothesis (falsified).
- Building/measuring on the main tree while an interactive pi session is alive (check `ps -eo pid,ppid,stat,etime,cmd | grep -w pi`; exclude drive.sh descendants).
- `stash@{0}` (the interactive session's tap WIP) — don't drop it; note any change to it in the handover.
- The manual serve on GPU 0 (pid 21537 at 15:17) — never bind/kill/restart from research work. No standby serve. No re-queue of `restart-primary`. Never commit `.env` (git-ignored).
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark session.