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
| 2026-08-18 12:50 | MTP stream residual vs pure greedy (k=0): after the committed-column bit-clone fix (e84ddaef, kept as a correctness fix, M1 +13.8%), the k=3 gate stream still differs from k=0 on 4/8 prompts within 256 tok (AIME k=1: single near-tie token flip at idx 119, 280 vs 343, reconverges 7 tok later). This exceeds the quality policy's small-diff threshold (2/8). Mechanism attributed 2026-08-18 ~12:55 by the interactive session (live WIP, uncommitted): **case (a) — residual committed-column defect in the FP8 linear small-t family** (T=2..4 kept the one-chain vpl=16 association + an A16 island over the A8 T=1 reference, while the NVFP4 family got the 4-chain T=1 clone in 5cbba58c). Bit-clone fix for T=2..4 implemented + op-level bit-parity test in progress. | Ratify keeping MTP at k=3 once the FP8 committed-column fix lands and the gate re-measures (expect the 4/8 diff to shrink; north-star speed, committed column becomes exactly the canonical greedy computation). | open — fix still in the interactive session's WIP (uncommitted as of 12:58; their 12:54 layer-hash dumps still diverge, so the fix was not yet verified). Driver iterations 8–9 backed off. Re-measure after it commits (next driver iteration's steps 3–4) |

## Resolved

| date (UTC) | blocker | resolution |
|---|---|---|

## Standing notes (awareness, no action needed)

- **Concurrent interactive session (2026-08-18 ~09:15):** an interactive `pi` TUI (pid 21662,
  up since 08:22) works in this repo alongside the unattended driver — it committed `18b4b674`
  and `855acadd` and is researching MTP acceptance (Exa helpers at `tools/autoninfer/exa_*.sh`).
  The driver's serve-op deferral already detects live pi sessions. If the interactive session and
  a driver iteration collide on GPU 1 or `handover.md`, the interactive one wins by default — the
  driver iteration should back off. Killed only by the user.
- **Instance restart (stop/start or reboot) is always safe**: the supervised primary serve and
  this loop both autostart, and the container filesystem (including the model artifact)
  survives. If a GPU-wedge blocker above asks for a restart, that is the *only* known recovery
  (userspace GPU reset is unavailable on this host).
- **Instance recycle/destroy is not safe**: it wipes the container filesystem. Reconstruction:
  re-clone `fikrikarim/autoninfer`, restore the git-ignored `models/qwen3_8_27b_nvfp4.ninfer`
  (21 GiB) from its source, run `bash tools/autoninfer/install_services.sh`, restore the models
  symlink (`ln -sfn /workspace/autoninfer/.pi/models.json /root/.pi/agent/models.json`).
- **Budget**: the loop runs ~30 h (wall clock, from `2026-08-18 ~09:00 UTC`) then exits cleanly.
  If it has stopped with no active blocker and you want it to continue:
  `supervisorctl start autoninfer-driver`.
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