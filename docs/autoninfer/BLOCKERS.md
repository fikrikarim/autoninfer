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

*(none — loop is running)*

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
- **Quality policy** (user rule, 2026-08-18): the north star is maximum speed; a small *measured*
  quality degradation is acceptable for a large speedup (BF16→NVFP4-class tradeoffs are fine). It
  is quantified with `tools/autoninfer/quality_gate.sh` (greedy 8-prompt gate, per-prompt token
  hashes), not assumed: pure perf changes must be token-identical; numerics changes keep only at
  ≥5% M1 gain with a small diff; ≥3/8 prompts changed goes here for your ratification when the
  speedup is large. Full rules: Ground rules 7 in `docs/autoninfer/README.md`.