/**
 * autoninfer project extension for the pi coding agent.
 *
 * - Appends a live environment snapshot (GPU ownership, live serve state, git state, most
 *   recent experiments) to the system prompt on every agent turn, so every session —
 *   including agents resuming after an instance restart — starts with correct environment
 *   context instead of re-discovering it (or, worse, assuming it).
 * - Registers `/autoninfer` to print the full snapshot in the TUI.
 *
 * Protocol this extension serves: docs/autoninfer/README.md
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

const pexec = promisify(execFile);

const SERVE_URL = "http://127.0.0.1:8080/v1/models";
const RESULTS_TSV = "docs/autoninfer/results.tsv";
const SNAPSHOT_TTL_MS = 20_000;

interface GpuInfo {
  lines: { idx: string; util: number; used: number; total: number }[];
  /** pid counts per GPU index */
  appsPerGpu: Record<string, number>;
}

async function sh(cmd: string[], cwd: string, timeoutMs = 4000): Promise<string> {
  const { stdout } = await pexec(cmd[0], cmd.slice(1), { cwd, timeout: timeoutMs });
  return stdout.trim();
}

export default function (pi: ExtensionAPI) {
  let cache: { at: number; brief: string; detail: string } | null = null;

  async function snapshotGpus(cwd: string): Promise<GpuInfo | null> {
    try {
      const out = await sh(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        cwd,
      );
      const lines = out
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => {
          const [idx, util, used, total] = l.split(",").map((s) => s.trim());
          return { idx, util: Number(util), used: Number(used), total: Number(total) };
        });
      const appsPerGpu: Record<string, number> = {};
      try {
        const gpus = await sh(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"], cwd);
        const busByIdx = new Map<string, string>();
        for (const l of gpus.split("\n").filter(Boolean)) {
          const [idx, bus] = l.split(",").map((s) => s.trim());
          busByIdx.set(idx, bus);
        }
        const apps = await sh(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader,nounits"], cwd);
        for (const l of apps.split("\n").filter(Boolean)) {
          const bus = l.split(",").map((s) => s.trim())[1];
          for (const [idx, b] of busByIdx) if (b === bus) appsPerGpu[idx] = (appsPerGpu[idx] ?? 0) + 1;
        }
      } catch {
        // per-GPU app counts are best-effort
      }
      return { lines, appsPerGpu };
    } catch {
      return null;
    }
  }

  async function snapshotServe(cwd: string): Promise<{ process: boolean; http: string }> {
    let process = false;
    try {
      const out = await sh(["ps", "-eo", "args="], cwd);
      process = out.split("\n").some((l) => l.includes("ninfer-serve"));
    } catch {
      // ps failed
    }
    let http = "unreachable";
    try {
      const res = await fetch(SERVE_URL, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        const body = (await res.json()) as { data?: { id: string }[] };
        const ids = (body.data ?? []).map((m) => m.id).join(", ");
        http = `up (${ids || "no models"})`;
      } else {
        http = `http ${res.status}`;
      }
    } catch {
      // serve down
    }
    return { process, http };
  }

  async function snapshotGit(cwd: string): Promise<string> {
    try {
      const [branch, head, dirty] = await Promise.all([
        sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
        sh(["git", "log", "-1", "--format=%h %s"], cwd),
        sh(["git", "status", "--porcelain"], cwd),
      ]);
      const dirtyCount = dirty ? dirty.split("\n").filter(Boolean).length : 0;
      const subject = head.slice(head.indexOf(" ") + 1).trim();
      const hash = head.slice(0, head.indexOf(" "));
      return `git: ${branch} @ ${hash} "${subject}"${dirtyCount > 0 ? ` (${dirtyCount} dirty files)` : " (clean)"}`;
    } catch {
      return "git: unavailable";
    }
  }

  async function recentExperiments(cwd: string): Promise<string[]> {
    try {
      const text = await fs.readFile(path.join(cwd, RESULTS_TSV), "utf8");
      const lines = text.trimEnd().split("\n").filter(Boolean);
      return lines.slice(-3);
    } catch {
      return [];
    }
  }

  async function buildSnapshot(): Promise<{ brief: string; detail: string }> {
    const cwd = process.cwd();
    const [gpus, serve, gitLine, recent] = await Promise.all([
      snapshotGpus(cwd),
      snapshotServe(cwd),
      snapshotGit(cwd),
      recentExperiments(cwd),
    ]);

    const brief: string[] = ["AUTONINFER ENVIRONMENT (live snapshot)"];
    const detail: string[] = [...brief];

    if (gpus) {
      for (const g of gpus.lines) {
        const apps = gpus.appsPerGpu[g.idx] ?? 0;
        const role =
          g.idx === "0"
            ? "RESERVED: live ninfer-serve (this harness's model)"
            : `research GPU: run tests/benchmarks with CUDA_VISIBLE_DEVICES=${g.idx}`;
        let line = `GPU ${g.idx}: ${g.util}% util, ${(g.used / 1024).toFixed(1)}/${(g.total / 1024).toFixed(0)} GiB, ${apps} compute app(s) - ${role}`;
        if (g.idx !== "0" && g.util > 90 && g.used < 100 && apps === 0) {
          line += " [WARN: SM counter pinned with no process - possible wedged GPU, verify with a timed CUDA kernel before benchmarking]";
        }
        brief.push(line);
        detail.push(line);
      }
    } else {
      brief.push("GPUs: nvidia-smi unavailable");
      detail.push("GPUs: nvidia-smi unavailable");
    }

    const serveLine = `Serve: process ${serve.process ? "running" : "MISSING"}; /v1/models ${serve.http}`;
    brief.push(serveLine);
    detail.push(serveLine);
    detail.push("Serve restart (after instance restart): see the recorded command in docs/autoninfer/README.md");
    detail.push(gitLine);
    if (recent.length > 0) {
      detail.push("Recent experiments (docs/autoninfer/results.tsv):");
      for (const line of recent) detail.push(`  ${line}`);
    }

    const rules =
      "Rules: GPU 0 belongs to the live serve - never bind, kill, or reconfigure it. All test/benchmark/profile work goes to the research GPU with CUDA_VISIBLE_DEVICES set to its index. " +
      "Log every experiment to docs/autoninfer/results.tsv; protocol: docs/autoninfer/README.md.";
    brief.push(rules);
    detail.push("");
    detail.push(rules);

    return { brief: brief.join("\n"), detail: detail.join("\n") };
  }

  async function getSnapshot(force = false): Promise<{ brief: string; detail: string }> {
    const now = Date.now();
    if (!force && cache && now - cache.at < SNAPSHOT_TTL_MS) return cache;
    const snap = await buildSnapshot().catch(() => ({
      brief: "AUTONINFER ENVIRONMENT (snapshot unavailable - check nvidia-smi and the serve manually)",
      detail: "AUTONINFER ENVIRONMENT (snapshot unavailable - check nvidia-smi and the serve manually)",
    }));
    cache = { at: now, ...snap };
    return snap;
  }

  pi.on("before_agent_start", async (event) => {
    const snap = await getSnapshot();
    return { systemPrompt: `${event.systemPrompt}\n\n${snap.brief}` };
  });

  pi.registerCommand("autoninfer", {
    description: "Show the live autoninfer environment snapshot (GPUs, serve, git, recent experiments)",
    handler: async (_args, ctx) => {
      const snap = await getSnapshot(true);
      ctx.ui.notify(snap.detail, "info");
    },
  });
}