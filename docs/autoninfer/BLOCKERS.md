# Autoninfer — Blockers requiring user action

The single place the research loop records anything it cannot resolve itself and that needs
you. The loop updates and pushes this file; check it periodically on GitHub. The loop keeps
running through everything short of an unrecoverable state — at which point it stops itself,
records the reason here (and in `/tmp/autoninfer-stop`), and halts cleanly.

**Nothing here is urgent unless it says "loop stopped".** A running loop with no active
blocker needs nothing from you.

## Active blockers

| date (UTC) | blocker | needed action | state |
|---|---|---|---|
| 2026-08-18 17:45 | **Product decision: close the MTP losslessness series by accepting the k>0 near-tie class.** After the full bit-clone campaign (5cbba58c, e84ddaef, f58db502, a36570d3) the remaining k>0 divergence is single-token near-tie flips (gate 6/8, reconvergent, clearly-correct outputs) caused by BF16 non-associativity of the quantized target under batched verify (T=2..4 vs T=1) - a property of ANY speculator on this target, not a fixable speculator bug. The last fixable piece (gdn_gating_proj T=2..8 clone) is DISCARDED because it moves the k=0 canonical (912dd5cb), and the residual k0 anomaly traces to GDN state-pool materialization (5d4d6298) - a defect hunt with ~0 M1 upside (the chain cost M1 130.13 -> 113.2, ~13%, for bit-exact verify). k=0 canonical (the non-speculative path) is bit-exact and stays. **Proposal: ratify "k=0 canonical = bit-exact (invariant); k>0 = accepted near-tie class (within the 2026-08-18 quality policy)" - closes the series, stops the M1 bleed, and lets the loop commit to DSpark (backlog #1).** Alternative: keep chasing (expected: more M1 cost, unfixable-without-side-effects residual). | Say yes/no (a comment on this row or in the pi session is enough; the loop records the verdict). Until then the loop defaults to yielding to DSpark per the step-back rule. | open since 17:45 |
| 2026-08-18 19:55 | **MTP k=4 draft window: ratify adopting as canonical? (M1 +23.2%, near-tie-class gate).** Measured 2026-08-18 19:30-19:40 on the current engine (committed-source worktree binary at 18198e78, GPU 1, fixed M1 menu tg128 -r 3, same session): k=4 = **140.36 +/- 0.53 tok/s, 33.8% accept, 165 rounds** vs k=3 = 113.93 +/- 0.26 / 30.15% / 201 rounds = **+23.2%**; k=5 = 138.33 +/- 0.17 / 28.96% / 159 rounds = +21.4% (k=4 wins; k=2/k=3 were A/B'd earlier: k=3 kept at 117.35 vs 118.16). Mechanism: verify width 5 leaves the T<=4 GEMV bit-clone route (the bit-exact-verify cost that moved M1 130.13 -> 113.0) for the fast wide route, and the wide route's column hidden states improve draft quality (tg128 per-position accept: pos1 53.7% -> 67.3%, pos2 23.9% -> 41.8%). Quality gate (8 fixed prompts, greedy, 256 tok): k=4 vs k=3 = **8/8 prompts differ** - the same near-tie class as the ratified k=3 status quo (k=3 vs k=0 = 5/8, ratified 0ab130b8): token lengths match within ~5 tok (256/256, 224/225, 193/190, 244/256), no loops or cut-offs; haiku + code-c-bug spot check: word-level different reasoning that is semantically identical, both reach the correct answer. Per quality policy rule 4 (diff >= 3/8, speedup > 20%): measurement kept and logged; **ask: yes/no on adopting k=4 as the canonical draft window** (on yes: serve wrappers `--draft-tokens 4`, M1 menu + baseline -> 140.4, gate canonical MTP4; the loop flips the config in the next iteration). Companion to the 17:45 row: the near-tie class now buys +23% speed, not just lossless status quo. | Say yes/no. Until then k=3 remains canonical (config unchanged). | open since 19:55 |

(None — the MTP blocker below was resolved and committed on 2026-08-18 by the user's
interactive session (f58db502); end-to-end ratification was completed by the driver on
2026-08-18 (~14:20, results.tsv @ 0ab130b8) — idx-119 flip gone, M1 re-baselined at
118.16 tok/s, k=3 kept canonical. MTP stays on at k=3 per the ratification in the
Resolved row below; the 5/8 gate residual is loop-owned (draft-column root cause
identified, see handover) and needs no user action.)

## Resolved

| date (UTC) | blocker | resolution |
|---|---|---|
| 2026-08-18 11:53–13:38 | MTP losslessness: after the attention committed-column bit-clone (28795af6, M1 +13.8%), the k=3 gate stream still differed from pure greedy k=0 on 4/8 prompts (> 2/8 policy; AIME k=1: near-tie flip at idx 119, 280 vs 343). Mechanism: residual committed-column defect in the FP8 linear small-T family (T=2..4 kept a one-chain vpl=16 association + an A16 island over the A8 T=1 reference, while the NVFP4 family had already received the T=1 clone in 5cbba58c). | Fixed by **f58db502** (committed 2026-08-18 13:38 by the user's interactive session): the FP8 linear small-T committed column (T=2..4) is now a T=1 GEMV bit-clone (vpl=8, four interleaved accumulator chains, same pair-to-chain mapping) and the MLP gate/up parent uses A8 at all token counts — mirroring the NVFP4 clone. Op-level verified before commit: `ninfer_fp8_linear_t_parity_test` PASS (T=1..4 per-column bit parity), `ninfer_linear_fp8_a16/a8` PASS with updated schedule expectations; the TEMP `NINFER_DEBUG_LAYERHASH` instrumentation and the throwaway `test_tmp_binary_diff` harness were stripped. **Ratified by the driver 2026-08-18 ~14:20 (results.tsv @ 0ab130b8):** AIME idx-119 flip (280 vs 343) **gone** — k=0/k=1 agree through ~token 233 (1024-tok greedy AIME). Residual is the draft-column near-tie class: gate k=0-vs-k=3 **5/8** (not below the pre-fix 4/8 — near-tie flips are a chaotic function of the (margin, ε) pair, so the FP8 fix changed ε rather than cutting a monotone noise floor; same single-token class, all outputs clearly correct; code-neighbor-sum became exact, summarize + code-c-bug newly differ). Root cause of the residual identified in code: the GQA small-T **draft-column** split partition derives from the committed window only (the T=1 clone covers committed columns of all three families + the GDN fold + the KV write — those are clean). **M1 re-baselined: 118.16 ± 0.16 tok/s / 32.3% accept at k=3** (−9.2% vs the pre-fix 130.13/38.4% — the cost of bit-exact verify: every T=2..4 verify round now runs the T=1 GEMV bit-clone; interactive-session cross-check 118.09 ± 0.14 at 13:57). k=2 re-decided: 117.35 ± 0.27 / 42.3% = tied within noise → **k=3 kept canonical; MTP stays on at k=3.** k=0 AIME-512 canonical stream moved 434280f5 → 0cc41a24fbcd0df8 (prefill T≤4 chunks now use the new FP8 T=2..4 schedule — same benign mechanism as the 5cbba58c move; eager==graph). Next loop step (loop-owned): make the GQA draft columns a T=1 bit-clone (see handover) to drive the 5/8 residual toward 0/8; no user action needed. |

## Standing notes (awareness, no action needed)

- **Concurrent interactive sessions (2026-08-18):** interactive `pi` TUI sessions may work in
  this repo alongside the unattended driver. Collisions are handled by the driver prompt's
  concurrent-session rule: the driver never edits a live session's files, backs off at most
  2 consecutive iterations on the same dirt, and on the 3rd must act (take over a converged
  WIP, stash an abandoned one, or escalate here). A no-op backoff loop beyond 2 iterations is
  a protocol violation, not a deferral policy.
- **Instance restart (stop/start or reboot) is always safe**: the supervised primary serve and
  this loop both autostart, and the container filesystem (including the model artifact)
  survives. If a GPU-wedge blocker above asks for a restart, that is the *only* known recovery
  (userspace GPU reset is unavailable on this host).
- **Instance recycle/destroy is not safe**: it wipes the container filesystem. Reconstruction:
  re-clone `fikrikarim/autoninfer`, restore the git-ignored `models/qwen3_8_27b_nvfp4.ninfer`
  (21 GiB) from its source, run `bash tools/autoninfer/install_services.sh`, restore the models
  symlink (`ln -sfn /workspace/autoninfer/.pi/models.json /root/.pi/agent/models.json`).
- **Budget**: the loop runs ~30 h wall clock (from 2026-08-18 ~09:00 UTC), capped by a safety
  ceiling of 400 iterations (iteration count is not a time budget). If it has stopped with no
  active blocker and you want it to continue: `supervisorctl start autoninfer-driver`.
- **Monitoring**: `docs/autoninfer/results.tsv` (one row per experiment), `git log` (one commit
  per iteration), `docs/autoninfer/handover.md` (what the next iteration will do),
  `/var/log/autoninfer-drive.log` and `/var/log/portal/ninfer-serve.log` (over SSH).
- **Design note — why one agent, not several**: the research GPU (one 5090) serializes every
  experiment anyway; the repository is single-writer; and the serve's `--max-concurrency 2`
  leaves at most one lane for a second agent. A second agent would contend for the harness's own
  model lane and the research GPU without adding throughput. The loop is single-agent by design.
  Interactive sessions and the loop share the serve safely (`--pending-timeout-ms 300000` lets
  the second request queue through a long interactive turn).
- **Quality policy** (user rule, 2026-08-18): the north star is maximum speed; a small *measured*
  quality degradation is acceptable for a large speedup (BF16→NVFP4-class tradeoffs are fine). It
  is quantified with `tools/autoninfer/quality_gate.sh` (greedy 8-prompt gate, per-prompt token
  hashes), not assumed: pure perf changes must be token-identical; numerics changes keep only at
  ≥5% M1 gain with a small diff; ≥3/8 prompts changed goes here for your ratification when the
  speedup is large. Full rules: Ground rules 7 in `docs/autoninfer/README.md`.