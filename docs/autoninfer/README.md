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
| GPU 0 | **Reserved.** The live `ninfer-serve` runs here and saturates the GPU; that serve is the harness's own model. Never bind tests, benchmarks, or profilers to GPU 0; never kill or reconfigure its process. |
| GPU 1 | **Research GPU.** All tests, benchmarks, and profile captures run here. Verify it before a benchmarking session: `bash tools/gpu_health.sh 1` (host-timed, high-occupancy bandwidth + compute probe; do not trust nvidia-smi counters or single-warp `clock64()` spins — see the changelog below). |
| Build | `build/` is configured with `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON`. Rebuild: `cmake --build build -j` (no numeric `-j`). |

The live serve is a plain process, not a supervisor service. After an instance restart it must be
relaunched manually:

```bash
cd /workspace/autoninfer
setsid ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft \
  > /var/log/portal/ninfer-serve.log 2>&1 &
ln -sfn /workspace/autoninfer/.pi/models.json /root/.pi/agent/models.json   # restore after recycle
```

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

Health check: `curl -s http://127.0.0.1:8080/v1/models`.

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
  the 143.8 / 48.9% published C=1 point (sustained AIME stream).

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
