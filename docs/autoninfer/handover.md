# Autoninfer handover

Rewritten at the end of every iteration; the next iteration's only inherited context.
Keep it complete and concise.

**Updated:** 2026-08-18 by the unattended driver iteration (MTP draft-window A/B + losslessness bug found).

## State

- HEAD: `4f3bc6d9` (feat: 5-min serve pending timeout, from the interactive session). Engine decode
  path is code-identical to the `59f9e54d` baseline for everything under test (commits since are
  docs/tools/serve-timeout only). Confirm with `git log --oneline -5`. Working tree: clean after
  this iteration's commit (results row + this file).
- **Concurrent interactive session (awareness):** still expected (committed `556c67fe` quality gate
  + EXA tooling, `4f3bc6d9` serve timeout). Do NOT kill it. It owns `tools/autoninfer/quality_gate*.sh`
  and `.env` (EXA key — never commit). Check `nvidia-smi` before any bench.
- Baseline M1 (tg128, `--lm-head-draft`, INT8 KV, MTP3): **111.42 ± 0.05 tok/s, 27.14% accept,
  1.803 tok/round** — re-confirmed this iteration at `4f3bc6d9` as **111.79 ± 0.07 tok/s,
  27.14% accept, 213 rounds** (fresh, same command). This is the keep/discard reference.
- Serve: still the loose process on GPU 0 (supervisor `ninfer-serve` STOPPED; `/v1/models` up).
  `{"action":"restart-primary"}` remains queued in `/tmp/autoninfer-ops/pending.json`; the driver
  applies it between iterations. Wrappers unchanged this iteration (canonical flags:
  `--spec mtp --draft-tokens 3 --lm-head-draft`, pinned 262144 ctx/KV).
- GPU 1: HEALTHY (triad ~1466 GiB/s, FMA ~110.7 TFLOP/s). All my temp serves ran on GPU 1 and
  are torn down; `nvidia-smi` shows only the GPU 0 serve.

## This iteration — draft-window A/B + a LOSSLESSNESS BUG (top priority)

### A/B: MTP `--draft-tokens` k=2 vs canonical k=3 (measured)

| menu | k=3 (canonical) | k=2 | Δ |
|---|---|---|---|
| M1 tg128 | 111.79 ± 0.07 tok/s, 27.14% accept, 1.803 tok/round | **116.52 ± 0.23 tok/s**, 37.93% accept, 1.753 tok/round | **k=2 +4.2%** |

k=2 removes the weak pos3 draft (per-position tg128: pos1 50.7%/pos2 24.7% are identical across k;
only pos3 7.0% is dropped) while the round shrinks 16.13→15.05 ms (−6.7%) faster than tok/round
drops (−2.7%) → net M1 win. **DECISION: keep k=3 canonical** — the real-corpus comparison is
INVALID (see bug) and the MTP output quality is untrusted until the bug is fixed. Re-run this A/B
after the fix.

### THE BUG: greedy (temp 0) MTP decode is NOT lossless — streams diverge across k

Contract (docs/maintainer/qwen3.6-27b-model.md §8): "A bad draft … must not change the
distribution of emitted target tokens." At temp 0 the distribution is a delta at the target
argmax, so **k=0/1/2/3 must all emit the identical stream** for a fixed prompt+seed. They do not.

Measured (AIME fixture `long_decode_aime26_01`, 1024 tok, `--greedy --seed 42`, GPU 1):

| config | hash | notes |
|---|---|---|
| k=0 (no MTP) | `c38d794e` | pure target greedy (reference stream) |
| k=1 | `1607489d` | diverges from k=0 at char 1118/2106 |
| k=2 | `d5f86e04` | diverges from k=0 at char 530/2106 |
| k=3 | `94aaa802` | diverges from k=0 at char 530/2106 |
| k=3 eager (`--no-cuda-graph`) | `cdc8b2ee`@512 | **== k=3 graph** (graph path is faithful) |

- Same-k is fully deterministic (k=3 auto-seed ×2 and k=3 `--seed 42` all hash-identical) →
  the divergence is **k-dependent, not seed/RNG**.
- Every k>0 produces its OWN distinct corrupted stream, and each diverges from the true k=0
  stream after a few hundred tokens → a **small persistent state perturbation** that occasionally
  flips the target argmax, not a hard logic fault (acceptance rates stay healthy: tg128 k=3
  per-position 105/51/15 of 213; AIME k=3 288/256/226 of 312 rounds).

**Isolated:** eager ≡ graph (identical hashes) → NOT a CUDA Graph capture/replay issue. The bug is
in the **shared MTP decode transaction** (the k>0 verify+accept+fold+trim path), affecting all k>0.

**Ruled out by inspection (do NOT re-derive these; they are correct):**
- `speculative_accept_greedy_drafts` greedy branch (`src/ops/kernel/speculative_round.cuh:72`):
  commits the longest matching draft prefix + target argmax at the divergence column — exact.
- GDN fold count (`program_impl.h:711` `resolve_pending_batch`): `commit_columns = licensed_counts
  = accepted+1` — correct. The "live state is one committed token behind" (missing the round's
  correction, which becomes next round's anchor) is **by design and correct by induction** — the
  anchor's transition is folded each round as record[0].
- KV trim (`program_impl.h:~820` `text_kv_valid = base_E + committed`): trims to exactly the
  committed positions — correct.
- `speculative_prepare_verify_inputs` (`include/ninfer/ops/speculative_round.h`): verify_ids =
  [anchor, d0..d_{k-1}], positions = base + min(j, Pcur) — correct.

**LEADING HYPOTHESIS (not yet confirmed — this is the task):** k=0 uses the **in-place** GDN
recurrent path (width-1 decode, `gdn_input_proj_conv_snapshot` in-place mode); k>0 uses the
**ReplaySSM record + fold** path (`gdn_input_proj_conv_record` during verify + `gdn_replay_fold`).
These are **two different code paths**. If the record+fold GDN state is not **bit-exact** with the
in-place state (the round-trip stores BF16 conv/key/value + FP32 {g,beta} records, then re-derives
the FP32 recurrent state; a different accumulation order or a precision round-trip would make it
close-but-not-equal), then the target logits computed from the folded state occasionally differ from
the true argmax → the greedy stream flips. This single mechanism explains ALL observations
(k=0≠all k>0; k=1/2/3 mutually distinct because window width changes the GDN verify computation;
divergence only after hundreds of tokens; healthy acceptance). The GDN is the only recurrent
(stateful) layer family, so it is the prime suspect.

**Concrete first steps (do these in order):**
1. **Confirm the hypothesis with a GDN state bit-comparison** — the decisive test. Run the same
   committed token prefix through both paths and dump the FP32 recurrent state after the update:
   - Path A (in-place): a k=0 ordinary decode of the prefix.
   - Path B (record+fold): a k>0 speculative round committing the same prefix.
   Compare the all-layer GDN FP32 recurrent states at the same position. Any mismatch localizes
   the bug to the record/fold round-trip. (Look for an existing state-dump/tap hook first — the
   reference `tools/reference/qwen3_6_27b` has a `tap` mechanism; the engine may have a similar
   debug path. `grep -rn "tap" src/targets/qwen3_6/impl/runtime/text_context_impl.h`.)
   If no hook exists, the cheapest is a focused unit test under `tests/` that drives
   `ops::gdn_replay_fold` and the in-place `gated_delta_net` recurrence on the same record and
   asserts bit-equality of the resulting FP32 state.
2. **If confirmed, read the exact round-trip** to find the precision/ordering loss:
   - record write: `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu` (conv record)
     + `src/ops/linear_attention/gated_delta_net/recurrent.cu` (recurrent record mode);
   - fold read/re-derive: `src/ops/linear_attention/gated_delta_net/replay.cpp`
     (`gdn_replay_fold`); record layout `src/core/gdn_replay_records.{h,cpp}`.
   The fix is to make the fold reproduce the in-place FP32 state exactly (same op ordering /
   no lossy round-trip), per the doc's "algebraically equivalent" requirement.
3. **After the fix, re-verify losslessness** (the acceptance gate for this bug): k=0/1/2/3 must
   all produce the SAME hash on the AIME fixture (`--greedy --seed 42`, GPU 1). Reusable harness
   (writes to /tmp, re-create if gone): boot `./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer
   --host 127.0.0.1 --port 8091 --max-context 16384 --kv-dtype int8 --max-concurrency 1 --greedy
   --seed 42 [--spec mtp --draft-tokens K --lm-head-draft]` with `CUDA_VISIBLE_DEVICES=1`, POST the
   AIME message to `/v1/chat/completions` (max_tokens 1024), hash `reasoning_content+"\n@@\n"+content`.
   Then re-run M1 (must still be ~111.8 @ k=3) and the k=2 A/B.
4. **Do NOT change sampling semantics** to "fix" this — the fix must make the folded state
   bit-exact, not loosen the accept op.

### Quality-gate coverage gap (methodology note — record, don't fix this iteration)

`tools/autoninfer/quality_gate.sh` boots a serve at a **fixed** `--draft-tokens 3` and hashes that
k=3 greedy stream. Because the k=3 stream is self-consistent (deterministic), the gate shows
"zero diff" across runs and **cannot detect** (a) the MTP-vs-no-MTP (k=0) divergence above, or
(b) any change to the non-MTP (k=0) path. For the "pure perf change must be token-identical" rule,
a k=0-path change would pass the gate invisibly. When the losslessness bug is fixed the gap
narrows (k=0==k=3), but until then treat gate "zero diff" as necessary, not sufficient, for
quality. (The interactive session owns the gate; coordinate before changing it.)

## Next experiment — ROOT-CAUSE the MTP losslessness bug (hypothesis above, step 1)

This is a **product-correctness** defect and outranks all performance backlog: the north star is
"faster at EQUAL or better quality," and the MTP path currently changes greedy output. Do step 1
(the GDN state bit-comparison) first — it is the decisive, cheap test and either confirms the
leading hypothesis (→ step 2 fix) or rules it out (→ profile the verify/fold path with nsys/ncu
on a single round to find where the k>0 state diverges from k=0).

## Do not repeat / watch out

- **The AIME tok/s comparison from the A/B (k=3 214 vs k=2 184 tok/s) is INVALID** — the two
  streams differ (bug), so it is not apples-to-apples. Discard those numbers; re-measure after the
  fix. Only the M1 tg128 A/B (111.79 vs 116.52) is a valid same-corpus measurement.
- Keep k=3 canonical until the bug is fixed; do NOT flip to k=2 on the M1 +4.2% alone (the real-
  corpus evidence is corrupted and the output quality is untrusted).
- Do NOT re-derive the ruled-out items above (accept kernel, fold count, KV trim, verify_ids) —
  they are verified correct; the bug is in the GDN record+fold round-trip (leading hypothesis).
- Never restart/touch the GPU 0 serve from an iteration (hard rule — driver owns it); queue
  `{"action":"restart-primary"}` for flag changes. Never start/leave the standby serve.
- GPU 1: run every GPU workload with `CUDA_VISIBLE_DEVICES=1`; `bash tools/gpu_health.sh 1` first.
- 35B-A3B / 27B groupwise-int artifacts absent locally (`models/` has only
  `qwen3_8_27b_nvfp4.ninfer`) — hypotheses needing them are out of reach.
- Bench JSONs this iteration: `/tmp/m1_k2_tg128.json` (k=2 tg128), `/tmp/det_k{0,1,2,3}_s42.json`
  + `/tmp/det3_k{0,3}_g{0,1}.json` (losslessness hashes), `/tmp/aime_req_k{2,3}.jsonl` (request
  logs with per-position acceptance). `/tmp` is cleared on reboot — re-run the harness if gone.