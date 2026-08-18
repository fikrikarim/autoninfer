# Autoninfer — agent-driven optimization of NInfer

Autoninfer runs an autonomous research loop (in the sense of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)) on this repository: an agent
proposes a performance hypothesis, implements it, measures it on the dedicated research GPU,
records the result, and keeps or discards the change. It follows the harness design patterns of
["Harness Engineering for Self-Improvement"](https://lilianweng.github.io/posts/2026-07-04-harness/):
the file system is the persistent memory (this document, the results log, git), every experiment is
explicit and inspectable, and the loop is a fixed protocol rather than a live conversation.

The loop is self-referential by design: the agent harness (the pi coding agent) is served by the
engine being optimized. The pi `ninfer` provider points at `http://127.0.0.1:8080/v1`
(`qwen3.8-27b`, NVFP4 artifact). The provider definition is tracked at [`.pi/models.json`](../../.pi/models.json)
(canonical) with `~/.pi/agent/models.json` symlinked to it; after a recycle the symlink must be
recreated (`ln -sfn /workspace/autoninfer/.pi/models.json ~/.pi/agent/models.json`). Faster NInfer
serves the researcher faster.

The loop runs unattended (see [Unattended driver](#unattended-driver)). Its state across
iterations lives in the repository, per the harness design: `docs/autoninfer/handover.md` carries
the forward state (what the next iteration does), `results.tsv` the experiment log, and
[`BLOCKERS.md`](BLOCKERS.md) is the single user-facing point for anything that needs a human.

The `contextWindow` (196,608) is deliberately below the engine's 262,144 per-sequence ceiling so
pi's auto-compaction (window minus its 16,384-token reserve) leaves ~65.5K of the shared 262,144-
token KV pool for a second concurrent request. Raise it toward 262,144 only if single-request
long-context sessions become the norm and interruptions are rare.

## Environment

| Resource | State |
|---|---|
| Machine | 2× NVIDIA GeForce RTX 5090 (32 GiB, `sm_120a`), CUDA 13.3 toolkit, driver 580.173.02 |
| Repository | `/workspace/autoninfer`, branch `master`, remote `origin` (`fikrikarim/autoninfer`) |
| Artifact | `models/qwen3_8_27b_nvfp4.ninfer` (local prerequisite, git-ignored) |
| GPU 0 | **Reserved.** The primary `ninfer-serve` (supervisor service `ninfer-serve`) runs here and saturates the GPU; that serve is the harness's own model. Research work never binds to GPU 0. Restarting the primary is a managed operation — see [Serve management](#serve-management) below. |
| GPU 1 | **Research GPU.** All tests, benchmarks, and profile captures run here. Verify it before a benchmarking session: `bash tools/gpu_health.sh 1` (host-timed, high-occupancy bandwidth + compute probe; do not trust nvidia-smi counters or single-warp `clock64()` spins — see the changelog below). Temporarily occupied by the standby serve only during a primary switchover (see [Serve management](#serve-management)). |
| Build | `build/` is configured with `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON`. Rebuild: `cmake --build build -j` (no numeric `-j`). |

The live serve is supervisor-managed (service `ninfer-serve`, autostart + autorestart): after an
instance restart it comes back on its own. The canonical flags live in
`/opt/supervisor-scripts/ninfer-serve.sh` (repo copy: `tools/autoninfer/supervisor/`); health
check: `supervisorctl status ninfer-serve` and `curl -s http://127.0.0.1:8080/v1/models`.

**After a recycle** the container filesystem is rebuilt from the image and the supervisor files
are gone — reconstruct the environment with:

```bash
git clone git@github.com:fikrikarim/autoninfer.git /workspace/autoninfer
cd /workspace/autoninfer
# restore the git-ignored model artifact (local prerequisite - restore from its source):
#   models/qwen3_8_27b_nvfp4.ninfer
bash tools/autoninfer/install_services.sh    # wrappers + service configs + supervisor registration
ln -sfn /workspace/autoninfer/.pi/models.json /root/.pi/agent/models.json
```

(The supervised primary starts immediately; it crash-loops until the model artifact is in
place, then `supervisorctl restart ninfer-serve`.)

`--kv-capacity 262144` pins the shared KV pool at 262,144 tokens — the exact size
`--kv-capacity auto` resolved to at the previous 131,072 context ceiling, so the change costs
zero extra memory while raising the per-sequence ceiling to the model's native 262,144. Do not
combine a higher `--max-context` with `--kv-capacity auto`: auto sizes the pool as
`max_concurrency × max_context` tokens, so at a 262,144 context with C=2 it would request
524,288 tokens ≈ 17.7 GiB (33,792 B/token INT8-G64 for this model; formula in
[the paged KV reference](../maintainer/paged-kv-cache.md)), which exceeds the ~10.9 GiB that
remains after the ~20 GiB of weights and the 1 GiB automatic headroom. The pinned KV pool and
the `contextWindow` in [`.pi/models.json`](../../.pi/models.json) must move together: the window is
the compaction ceiling, the engine limit is the hard one.

### Serve management

Three supervisor services own the serving layer (configs in `/etc/supervisor/conf.d/`, wrappers
in `/opt/supervisor-scripts/`):

| Service | GPU | Port / model id | Lifecycle |
|---|---|---|---|
| `ninfer-serve` | 0 | 8080 / `qwen3.8-27b` | autostart, autorestart — the harness's model |
| `ninfer-serve-standby` | 1 | 8081 / `qwen3.8-27b-standby` | manual — bridge serve, only during a switchover; must not be left running (occupies ~30.5 GiB of the research GPU) |
| `autoninfer-driver` | — | — | manual — the unattended research loop (below) |

**Why the standby exists:** the primary serve is the harness's own model — killing it mid-turn
kills the agent doing the killing. A switchover therefore bridges: the session first moves onto
the standby (GPU 1, distinct model id so misrouted requests 404 instead of silently hitting the
wrong instance), the primary is restarted on GPU 0, then the session moves back and the standby
stops. The whole sequence is driven by the `autoninfer_standby` tool registered by the pi
extension (`.pi/extensions/autoninfer.ts`), which an agent can call in-turn:

1. `autoninfer_standby {action: "start"}` — starts the standby (if needed, ~45 s first load),
   waits for `/v1/models` on 8081, and switches the session's model onto it (`pi.setModel`).
2. `supervisorctl restart ninfer-serve` via bash; poll `curl -s http://127.0.0.1:8080/v1/models`.
3. `autoninfer_standby {action: "stop"}` — verifies the primary is healthy, switches the session
   back, stops the standby.

If a session is interrupted between `start` and `stop`, the snapshot injected into the next agent
turn says so; the fix is to call `stop` again (or: `supervisorctl stop ninfer-serve-standby` and
switch the model back with `/model`).

**Flag changes:** to apply new serve flags, edit `/opt/supervisor-scripts/ninfer-serve.sh` (and
keep this document in sync), then restart via one of the two paths above. Unattended
(`autoninfer-driver`) sessions do not use the bridge — they write
`{"action":"restart-primary"}` to `/tmp/autoninfer-ops/pending.json` and the driver applies it
between iterations, when no driver pi process is alive (and it defers while an interactive pi
session is running).

### Unattended driver

`tools/autoninfer/drive.sh` (service `autoninfer-driver`, autostart) runs the research loop
headlessly. Each iteration is a **fresh** `pi -p` session (`autoninfer-driver-N`, visible via
`pi -r`): the repository is the persistent memory and `docs/autoninfer/handover.md` carries the
state across iterations, so no session context grows unbounded over a day of work. An iteration
picks one hypothesis from the backlog (or the handover's next step), implements it, measures
with the protocol menu, logs the row to `results.tsv`, rewrites the handover, and commits +
pushes. Between iterations the driver also:

- applies pending serve ops (see [Serve management](#serve-management)) — deferred while an
  interactive pi session is alive;
- self-heals the primary serve if it is down (the harness has no model to think with otherwise);
- gates on GPU 1 health (`tools/gpu_health.sh 1`) with 5-minute backoff **without consuming an
  iteration** — wedges have cleared on their own within the hour on this host; after 2 h it
  records a blocker in `BLOCKERS.md` and stops (needs an instance restart).

Budget: `DURATION_HOURS=30` wall clock (set in the service config), `MAX_ITER=120` safety cap,
`ITER_TIMEOUT_S=3600` hard cap per iteration. A clean exit (budget, stop flag, recorded blocker)
is **not** restarted; `supervisorctl start autoninfer-driver` re-arms it. After an instance
restart the loop resumes on its own (autostart). Log: `/var/log/autoninfer-drive.log`. Controls:

```bash
supervisorctl start autoninfer-driver     # (re)arm the loop
supervisorctl stop autoninfer-driver      # stop after the current iteration
touch /tmp/autoninfer-stop                # agent-side stop (also read between iterations)
echo '{"action":"restart-primary"}' > /tmp/autoninfer-ops/pending.json   # queue a serve restart
```

The driver's iterations and an interactive session share the primary serve
(`--max-concurrency 2` absorbs the overlap; excess requests queue with a 30 s pending timeout).
The driver stops itself when the primary cannot be restored (no model to iterate on) and records
why in `BLOCKERS.md`.

**Single-agent by design**: the research GPU serializes experiments anyway, the repository is
single-writer, and the serve's two lanes leave at most one for a second agent — parallel agents
would contend without adding throughput.

## Ground rules

1. **GPU discipline.** Every GPU-touching command used for testing runs with `CUDA_VISIBLE_DEVICES=1`
   (or `--device 1` for tools that take a device flag). GPU 0 belongs to the live serve.
2. **Fixed measurement menu.** Comparability beats coverage. Use the menu below; do not invent new
   measurement commands for routine keep/discard decisions.
3. **Correctness first.** Numerical changes pass the affected explicit tests (see `tests/README.md`)
   under the oracle rules of `AGENTS.md` before they are measured.
4. **One hypothesis per commit.** Commit before measuring; the results row references the commit.
5. **Record everything.** Every experiment — keep, discard, or crash — gets a `results.tsv` row.
6. **Push cadence.** Push after every kept change (and after informative discards), so progress
   survives an instance recycle. Never push `models/` or `build/`.

## The loop

1. Pick a hypothesis from the backlog below (or from fresh profiling).
2. Profile and implement the change; run the affected tests.
3. `git commit` (Conventional Commit subject).
4. Measure on GPU 1 with the fixed menu.
5. Append a `results.tsv` row; decide keep / discard / crash (revert with `git revert`).
6. Push. Next hypothesis.

Simplicity criterion (inherited from autoresearch): all else equal, simpler wins; an improvement
that deletes code is the best kind of outcome.

## Fixed measurement menu

All entries run on GPU 1 against `models/qwen3_8_27b_nvfp4.ninfer` with INT8 KV and CUDA Graphs,
matching the published qwen3.8-27b nvfp4 serving profile.

| ID | Purpose | Command |
|---|---|---|
| M1 | end-to-end decode (primary metric) | `CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft` |
| M2 | end-to-end prefill | `CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -p 2048 -r 3 --warmup 1 --kv-dtype int8` |
| M3 | kernel attribution | the `bench/` Op benchmarks (see `bench/README.md`), always with `CUDA_VISIBLE_DEVICES=1` |
| M4 | deep kernel profiling | `ncu`/`nsys` on a single bench point (`--profile` / `--profile-measured`), output under `profiles/` |

M1 reports decode tok/s and MTP acceptance over `tg128` from the generic `bench_corpus.ids`
at a 16,384-token context ceiling (the published campaign's setting). M1 is the number
`results.tsv` tracks. MTP acceptance is content-dependent: the generic corpus gives ~27%
acceptance at `tg128`, while the sustained AIME reasoning stream of the published campaign
gives ~49%. Compare M1-to-M1 across commits; use the published corpus runner
(`tools/bench/run_serve_concurrency.py`, see `docs/performance.md`) for campaign-scale claims.

## Results log

`results.tsv` is tab-separated:

```text
commit	metric	status	description
```

- `commit` — short hash of the commit under test.
- `metric` — primary M1 value, `tok/s, accept%` (e.g. `143.8 tok/s, 48.9% accept`); `0.0` for crashes.
- `status` — `keep`, `discard`, or `crash`.
- `description` — one line, no tabs.

## Baseline

Published reference (qwen3.8-27b nvfp4, RTX 5090, INT8 KV, MTP3 — full methodology in
[docs/performance.md](../performance.md)):

| Point | Decode | MTP acceptance |
|---|---:|---:|
| C=1 | 143.8 tok/s | 48.9% |
| C=2 | 267.6 tok/s | 48.1% |
| C=4 | 461.1 tok/s | 45.8% |
| C=8 | 766.6 tok/s | 46.0% |

Single-request MTP0: 8,340.4 prefill tok/s at a 7,680-token prompt; 71.2 decode tok/s.

First local M1 measurement (HEAD `5a3aab20`, GPU 1, 2026-08-18):

- `tg128` decode: **111.82 ± 0.07 tok/s**, MTP acceptance **27.1%** (213 rounds, 0 fallbacks).
  Generic-corpus operating point; see the corpus-dependence note above before comparing it to
  the 143.8 / 48.9% published C=1 point (sustained AIME stream). Reproduced at the 2026-08-18
  experiment HEAD: 111.42 ± 0.05 tok/s, 27.1%.

## Hypothesis backlog

Seeds, not a mandate; profile first. Ranked by expected end-to-end impact:

1. **MTP acceptance (45.8–48.9%) is far below the other measured profiles (67–71%).** Audit this
   artifact's draft path: proposal-head quality on the mixed FP8/NVFP4 weights, `--lm-head-draft`
   versus full-head drafting, and per-position acceptance to find which draft positions collapse.
2. **Prefill trails the qwen3.6-27b nvfp4 counterpart by 1.34×** (8,340 versus 11,191 tok/s at
   7,680 tokens). The qwen3.8 nvfp4 profile keeps FP8 on the token embedding, attention
   projections, GDN projections, output head, and remaining MLP weights; check the A8 crossovers of
   `fp8_linear_add` / `fp8_linear_swiglu` and the FP8 GDN input projection at prefill token counts.
3. **Batched decode at concurrency ≥ 4** trails the qwen3.6 nvfp4 profile in absolute tok/s despite
   a good 5.33× C=1→C=8 scaling; compare the batched round path (compaction, sampling, MTP pack)
   between the two 27B packages.
4. **Qwen3.8-27B `groupwise-int` is supported but has no published benchmark** — a second profile on
   the same execution package; useful as an execution-path control and as a newly measured identity.
5. **Host-side round overhead**: argmax over the 248,077-vocabulary, MTP pack/split transforms, and
   round-boundary compaction; cheap Op-level checks via `bench/ops` (`argmax`, `mtp_pack`).

## Changelog (autoninfer setup)

- **2026-08-18 — MTP head A/B measured; full head discarded (M1 regression).**
  Per-position MTP acceptance on the generic corpus (tg512, `--lm-head-draft`): pos1 59.3% /
  pos2 31.4% / pos3 12.3% — the collapse is along the sequential draft chain, not at pos1. A/B
  against full-head drafting (same commands minus `--lm-head-draft`): on tg512 full head wins
  (128.13 ± 0.10 vs 125.04 ± 0.25 tok/s, 42.64% vs 34.35% acceptance, per-position
  459/252/150 vs 447/237/93, 675 vs 756 rounds, 0 vs 3 fallbacks), but on the M1 menu (tg128)
  it **regresses**: 105.16 ± 0.09 vs 111.42 ± 0.05 tok/s (28.78% vs 27.14% acceptance,
  per-position 120/45/12 vs 105/51/15). Mechanism: the full head's 248,320-row FP8 output GEMV
  (~1.27 GB read per draft step, bandwidth-bound) costs ~1.5 ms more per round than the
  131,072-row shortlist head; the +2.9% tokens/round on low-predictability tg128 content does not
  amortize it, while the +12% on settled tg512 streams does. Decision: keep `--lm-head-draft`
  as the canonical configuration (M1 is the protocol metric); output quality is unaffected by
  either setting (exact target verification, `include/ninfer/ops/speculative_round.h`), so the
  trade-off is speed-only. Follow-ups: (a) re-test the head swap on a real-text corpus
  (`tools/bench/run_serve_corpus.py`) to decide the product regime; (b) the binding constraint
  is the MTP layer's one-step quality (pos1 58–68% vs the 0.835 teacher-forced offline oracle),
  which no head choice fixes.
- **2026-08-18 — initial setup.** Changes made to the repository and harness:
  - `build/` reconfigured with `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON` (the default
    configuration builds apps only).
  - `AGENTS.md`: autoninfer environment rules added (GPU ownership, measurement discipline).
  - `README.md`: clone URL points at this repository; autoninfer section added.
  - `docs/README.md`: this protocol listed under repository-local guides.
  - `.gitignore`: `models/` ignored (large local prerequisite).
  - `.pi/extensions/autoninfer.ts`: pi project extension that injects the live GPU/serve/git
    snapshot into the agent system prompt and registers `/autoninfer`.
  - **GPU 1 wedged at ~07:54**, then **recovered by ~08:40 without a restart.** Wedged state:
    SM busy pinned at 100% with no attached process, triad 52 GiB/s versus ~1.5 TB/s. The initial
    diagnosis leaned on a single-warp `clock64()` spin (69 ms instead of ~70 µs) — that test is
    now known to be a bad indicator: a 32-thread spin leaves the SM clock-gated at ~2.9 MHz
    whether the card is healthy or wedged, so it read identically in both states. The trustworthy
    signals are host-timed, high-occupancy workloads (`tools/gpu_health.sh`: triad ≥ 500 GiB/s and
    all-SM FMA ≥ 20 TFLOP/s; observed healthy 2026-08-18: 1,467 GiB/s and 110.7 TFLOP/s). If a
    wedge recurs and persists, an instance restart is the only known recovery; it also kills the
    live serve on GPU 0 — relaunch it with the recorded command. GPU 0 was healthy throughout.
- **2026-08-18 — context ceiling raised; models config tracked in repo.**
  - Serve restart command changed to `--max-context 262144 --kv-capacity 262144` (pinned pool at
    the previous auto-resolved size: zero memory delta, per-sequence ceiling doubled to the model
    native 262,144). Takes effect at the next serve restart.
  - pi provider config moved into the repo: `.pi/models.json` is canonical,
    `~/.pi/agent/models.json` is a symlink to it (pi only reads the global path; restore the
    symlink after a recycle). `contextWindow` raised 131,072 → 196,608 (compaction at 180,224;
    keeps ~65.5K of pool headroom for concurrent requests). Applies to the next pi session.
  - The two changes must be applied together: with the window raised, pi will happily build
    requests past the old 131,072 engine ceiling, which a still-old serve rejects.
  - Git identity set to `Fikri Karim <fkfikrikarim@gmail.com>`; the two setup commits were
    re-authored accordingly.
- **2026-08-18 — GPU health probe added; first local baseline.**
  - `tools/gpu_health.sh <gpu-index>`: one-command research-GPU check (host-timed, high-occupancy
    triad bandwidth + all-SM FP32 FMA; pass thresholds ≥ 500 GiB/s and ≥ 20 TFLOP/s). Replaces
    the spin-based check, which read identically on healthy and wedged cards. Observed healthy:
    1,467 GiB/s, 110.7 TFLOP/s. Run it before every benchmarking session (see GPU 1 row).
  - First local M1 baseline recorded: `tg128` 111.82 ± 0.07 tok/s, 27.1% MTP acceptance at HEAD
    `5a3aab20` (row 1 of `results.tsv`).
- **2026-08-18 — fully autonomous serve management; fixed `--device` bug.**
  - **Engine bug fix:** `ConcurrentExecutor`'s worker thread never called `cudaSetDevice`, so CUDA
    calls on that thread defaulted to device 0. With the engine on `--device 1` (second physical
    GPU visible to the process) the worker's first launch failed with `cudaErrorInvalidValue`
    (cross-GPU stream/pointer mismatch). `CUDA_VISIBLE_DEVICES`-based selection masked the bug
    because device 0 == the only visible device. Fix: the worker establishes its current device
    from `EngineOptions.device` at `worker_loop` entry (`src/runtime/engine/concurrent_executor.h`).
    Verified: `--device 1` serve now warms up and serves generations on GPU 1 end-to-end.
  - Serve layer moved under supervisor (see [Serve management](#serve-management)): `ninfer-serve`
    (GPU 0, autostart/autorestart), `ninfer-serve-standby` (GPU 1, 8081, `qwen3.8-27b-standby`,
    manual bridge serve), `autoninfer-driver` (manual unattended loop). The primary serve now
    survives instance restarts on its own.
  - `autoninfer_standby` pi tool (extension): in-turn zero-gap switchover — session moves onto the
    standby, the agent restarts the primary, the session moves back. `pi.setModel` makes the
    mid-session model switch possible.
  - Unattended driver (`tools/autoninfer/drive.sh`): headless `pi -p` iterations on the
    `autoninfer-driver` session; applies serve ops from `/tmp/autoninfer-ops/pending.json`
    between iterations only (deferred while an interactive pi session is alive); MAX_ITER cap.
  - `.pi/models.json` now carries both provider entries (`ninfer`, `ninfer-standby`); the
    standby's distinct model id makes misrouted requests 404 loudly.
- **2026-08-18 — 24 h+ unattended operation hardened.**
  - Driver rewritten for long-horizon autonomy: fresh `pi -p` session per iteration
    (`autoninfer-driver-N`; the repository is the persistent memory), wall-clock budget
    (`DURATION_HOURS=30`) + safety caps, 1 h per-iteration timeout, emergency primary-serve
    self-heal between iterations, and a GPU 1 health gate that backs off 5 min at a time
    without consuming an iteration (2 h patience, then a recorded blocker + clean stop).
  - New state documents: `docs/autoninfer/handover.md` (rewritten every iteration; the next
    iteration's only inherited context) and `docs/autoninfer/BLOCKERS.md` (the single
    user-facing point for anything needing a human; the loop stops itself and records there
    on unrecoverable states).
  - **Driver stdin bug fixed (caught pre-launch):** supervisor children get stdin as an event
    pipe that never EOFs, and `pi -p` reads piped stdin before starting the agent loop - every
    iteration hung before its first model call (alive process, no session file, no connection
    to the serve). Fix: `< /dev/null` on the pi invocation. Lesson for any future headless pi
    under supervisor: always redirect stdin.
