/**
 * autoninfer project extension for the pi coding agent.
 *
 * - Appends a live environment snapshot (GPU ownership, live serve state, git state, most
 *   recent experiments) to the system prompt on every agent turn, so every session —
 *   including agents resuming after an instance restart — starts with correct environment
 *   context instead of re-discovering it (or, worse, assuming it).
 * - Registers `/autoninfer` to print the full snapshot in the TUI.
 * - Registers the `autoninfer_standby` tool: zero-gap primary-serve restarts. The agent
 *   itself starts the GPU 1 standby serve and switches this session onto it, restarts the
 *   primary on GPU 0, then switches back and stops the standby. Protocol: docs/autoninfer/README.md
 *   ("Serve management" / "Unattended driver").
 */
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

const pexec = promisify(execFile);

const PRIMARY_URL = "http://127.0.0.1:8080/v1/models";
const STANDBY_URL = "http://127.0.0.1:8081/v1/models";
const PRIMARY_MODEL = { provider: "ninfer", id: "qwen3.8-27b" };
const STANDBY_MODEL = { provider: "ninfer-standby", id: "qwen3.8-27b-standby" };
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

async function urlUp(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

async function pollUrl(url: string, attempts: number, delayMs: number): Promise<boolean> {
  for (let i = 0; i < attempts; i++) {
    if (await urlUp(url)) return true;
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return await urlUp(url);
}

/** Parse `supervisorctl status <name>` into the state word (RUNNING, STOPPED, EXITED, ...). */
async function serviceState(name: string, cwd: string): Promise<string> {
  try {
    const out = await sh(["supervisorctl", "status", name], cwd);
    const m = out.match(/(\w+)\s+$/m);
    return m ? m[1] : "UNKNOWN";
  } catch {
    return "UNAVAILABLE";
  }
}

export default function (pi: ExtensionAPI) {
  let cache: { at: number; brief: string; detail: string } | null = null;
  let sessionOnStandby = false; // this pi session's model was switched to the standby serve

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
      const res = await fetch(PRIMARY_URL, { signal: AbortSignal.timeout(1500) });
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
    const [gpus, serve, gitLine, recent, primaryState, standbyState] = await Promise.all([
      snapshotGpus(cwd),
      snapshotServe(cwd),
      snapshotGit(cwd),
      recentExperiments(cwd),
      serviceState("ninfer-serve", cwd),
      serviceState("ninfer-serve-standby", cwd),
    ]);

    const brief: string[] = ["AUTONINFER ENVIRONMENT (live snapshot)"];
    const detail: string[] = [...brief];

    if (gpus) {
      for (const g of gpus.lines) {
        const apps = gpus.appsPerGpu[g.idx] ?? 0;
        const role =
          g.idx === "0"
            ? "RESERVED: live ninfer-serve (this harness's model)"
            : `research GPU: run tests/benchmarks with CUDA_VISIBLE_DEVICES=${g.idx} (check bash tools/gpu_health.sh ${g.idx} before benchmark sessions)`;
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

    const serveLine =
      `Serve: primary ${primaryState} (supervisor) - /v1/models ${serve.http}` +
      (sessionOnStandby ? "; SESSION ON STANDBY (complete the switchover with autoninfer_standby stop)" : "") +
      `; standby ${standbyState}${standbyState === "RUNNING" ? ` (/v1/models ${await urlUp(STANDBY_URL) ? "up" : "down"})` : ""}`;
    brief.push(serveLine);
    detail.push(serveLine);
    detail.push(gitLine);
    if (recent.length > 0) {
      detail.push("Recent experiments (docs/autoninfer/results.tsv):");
      for (const line of recent) detail.push(`  ${line}`);
    }

    const rules =
      "Rules: GPU 0 holds the primary serve (this harness's model) - research work never binds or reconfigures it without a switchover. " +
      "All test/benchmark/profile work goes to the research GPU with CUDA_VISIBLE_DEVICES set to its index. " +
      "Restarting the primary: interactive sessions use the autoninfer_standby tool (start -> restart primary via bash -> stop); " +
      "unattended (autoninfer-driver) sessions write {\"action\":\"restart-primary\"} to /tmp/autoninfer-ops/pending.json between iterations. " +
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

  pi.registerTool({
    name: "autoninfer_standby",
    label: "Autoninfer standby",
    description:
      "Zero-gap switchover for the GPU 0 primary serve, which powers this session. " +
      "start: starts the GPU 1 standby serve and switches THIS session's model onto it - call it BEFORE restarting the primary. " +
      "stop: verifies the primary is healthy, switches this session back, and stops the standby - call it AFTER the primary restart completes. " +
      "status: shows the switchover state. If a session is interrupted between start and stop, call stop to complete it.",
    parameters: Type.Object({
      action: Type.Union([Type.Literal("start"), Type.Literal("stop"), Type.Literal("status")], {
        description: "start | stop | status",
      }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx: ExtensionContext) {
      const cwd = process.cwd();
      const text = (t: string, isError = false) => ({ content: [{ type: "text" as const, text: t }], details: {}, isError });

      if (params.action === "status") {
        const [primary, standby] = await Promise.all([
          serviceState("ninfer-serve", cwd),
          serviceState("ninfer-serve-standby", cwd),
        ]);
        return text(
          `session on standby: ${sessionOnStandby}; primary: ${primary}; standby: ${standby}` +
            (sessionOnStandby ? " - complete the switchover by calling action=stop" : ""),
        );
      }

      if (params.action === "start") {
        if (sessionOnStandby) {
          return text("Already on the standby serve. Restart the primary now, then call action=stop.");
        }
        const standbyState = await serviceState("ninfer-serve-standby", cwd);
        if (standbyState !== "RUNNING") {
          try {
            await sh(["supervisorctl", "start", "ninfer-serve-standby"], cwd, 10_000);
          } catch (e) {
            return text(`Failed to start the standby serve: ${String(e)}`, true);
          }
        }
        const up = await pollUrl(STANDBY_URL, 90, 2000); // first start loads a 20 GiB model (~45 s)
        if (!up) {
          return text(
            "The standby serve did not come up within 3 min (check /var/log/portal/ninfer-serve-standby.log). " +
              "Primary untouched; session still on the primary.",
            true,
          );
        }
        const model = ctx.modelRegistry.find(STANDBY_MODEL.provider, STANDBY_MODEL.id);
        if (!model) {
          await sh(["supervisorctl", "stop", "ninfer-serve-standby"], cwd).catch(() => {});
          return text(
            `Standby serve is up but the model ${STANDBY_MODEL.provider}/${STANDBY_MODEL.id} is unknown to this pi session ` +
              "(restart the session so it loads .pi/models.json). Standby stopped to free the GPU.",
            true,
          );
        }
        const ok = await pi.setModel(model);
        if (!ok) {
          await sh(["supervisorctl", "stop", "ninfer-serve-standby"], cwd).catch(() => {});
          return text("Failed to switch this session onto the standby model; standby stopped. Primary untouched.", true);
        }
        sessionOnStandby = true;
        return text(
          "SWITCHED ONTO STANDBY: this session now runs on " + STANDBY_MODEL.id + " (GPU 1). " +
            "You may now restart the primary: run `supervisorctl restart ninfer-serve` via bash, then poll " +
            "`curl -s http://127.0.0.1:8080/v1/models` until it answers, then call autoninfer_standby with action=stop. " +
            "Do not run GPU 1 benchmarks while the standby occupies it.",
        );
      }

      // stop
      if (!sessionOnStandby) {
        const standbyState = await serviceState("ninfer-serve-standby", cwd);
        if (standbyState === "RUNNING") {
          return text(
            "This session is not on the standby, but the standby serve is running - stopping it to free the research GPU.",
            false,
          );
        }
        return text("Nothing to do: session is on the primary and the standby serve is not running.");
      }
      const primaryUp = await pollUrl(PRIMARY_URL, 60, 2000);
      if (!primaryUp) {
        return text(
          "Primary serve not healthy yet (no response on 127.0.0.1:8080). Session stays on the standby. " +
            "Check `supervisorctl status ninfer-serve` and /var/log/portal/ninfer-serve.log, then call action=stop again.",
          true,
        );
      }
      const model = ctx.modelRegistry.find(PRIMARY_MODEL.provider, PRIMARY_MODEL.id);
      if (!model) {
        return text("Primary model unknown to this pi session (models.json not loaded?). Standby left running.", true);
      }
      const ok = await pi.setModel(model);
      if (!ok) {
        return text("Failed to switch this session back to the primary model. Standby left running; retry.", true);
      }
      sessionOnStandby = false;
      let stopped = "stopped";
      try {
        const state = await serviceState("ninfer-serve-standby", cwd);
        if (state === "RUNNING") {
          await sh(["supervisorctl", "stop", "ninfer-serve-standby"], cwd, 60_000);
        }
      } catch (e) {
        stopped = `stop command failed (${String(e)}) - check with supervisorctl`;
      }
      return text(
        `SWITCHED BACK TO PRIMARY: session runs on ${PRIMARY_MODEL.id} (GPU 0) again; standby ${stopped}. GPU 1 is free for research.`,
      );
    },
  });
}