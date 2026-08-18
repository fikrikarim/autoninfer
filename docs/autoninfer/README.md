# Autoninfer — agent-driven optimization of NInfer

Autoninfer runs an autonomous research loop (in the sense of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)) on this repository: an agent
proposes a performance hypothesis, implements it, measures it on the dedicated research GPU,
records the result, and keeps or discards the change. It follows the harness design patterns of
["Harness Engineering for Self-Improvement"](https://lilianweng.github.io/posts/2026-07-04-harness/):
the file system is the persistent memory (this document, the results log, git), every experiment is
explicit and inspectable, and the loop is a fixed protocol rather than a live conversation.

The loop is self-referential by design: the agent harness (the pi coding agent) is served by the
engine being optimized. `~/.pi/agent/models.json` points the pi `ninfer` provider at
`http://127.0.0.1:8080/v1` (`qwen3.8-27b`, NVFP4 artifact). Faster NInfer serves the researcher
faster.

## Environment

| Resource | State |
|---|---|
| Machine | 2× NVIDIA GeForce RTX 5090 (32 GiB, `sm_120a`), CUDA 13.3 toolkit, driver 580.173.02 |
| Repository | `/workspace/autoninfer`, branch `master`, remote `origin` (`fikrikarim/autoninfer`) |
| Artifact | `models/qwen3_8_27b_nvfp4.ninfer` (local prerequisite, git-ignored) |
| GPU 0 | **Reserved.** The live `ninfer-serve` runs here and saturates the GPU; that serve is the harness's own model. Never bind tests, benchmarks, or profilers to GPU 0; never kill or reconfigure its process. |
| GPU 1 | **Research GPU.** All tests, benchmarks, and profile captures run here. |
| Build | `build/` is configured with `-DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON`. Rebuild: `cmake --build build -j` (no numeric `-j`). |

The live serve is a plain process, not a supervisor service. After an instance restart it must be
relaunched manually:

```bash
cd /workspace/autoninfer
setsid ./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 131072 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft \
  > /var/log/portal/ninfer-serve.log 2>&1 &
```

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
| M1 | end-to-end decode (primary metric) | `CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft` |
| M2 | end-to-end prefill | `CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -p 2048 -r 3 --warmup 1 --kv-dtype int8` |
| M3 | kernel attribution | the `bench/` Op benchmarks (see `bench/README.md`), always with `CUDA_VISIBLE_DEVICES=1` |
| M4 | deep kernel profiling | `ncu`/`nsys` on a single bench point (`--profile` / `--profile-measured`), output under `profiles/` |

M1 reports decode tok/s and MTP acceptance over `tg128`. M1 is the number `results.tsv` tracks.

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

First local M1 measurement at current HEAD: **pending** — GPU 1 was wedged when this document was
written (see the changelog below). Run M1 immediately after GPU 1 is healthy and record it as the
baseline row.

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
  - **GPU 1 is wedged** as of this writing: SM busy counter pinned at 100% with no attached
    process, a 200 M-cycle spin kernel takes 69.5 ms instead of ~70 µs (effective SM clock ~1000×
    below the reported 2872 MHz), triad 52 GiB/s versus ~1.8 TB/s HBM3e. Userspace recovery is not
    available (clock reset lacks permission, `--gpu-reset` unsupported, persistence toggle had no
    effect). Recovery requires an instance restart, which also kills the live serve on GPU 0 —
    relaunch it with the recorded command. GPU 0 is healthy.