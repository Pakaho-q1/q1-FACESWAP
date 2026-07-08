import { usePipelineStore } from "../store/pipeline-store";

export function StatusIndicator() {
  const status = usePipelineStore((state) => state.status);
  const errorMessage = usePipelineStore((state) => state.errorMessage);

  const statusConfigs = {
    idle: {
      color: "bg-slate-500",
      label: "Idle",
      bg: "bg-slate-500/10 border-slate-500/20 text-slate-400",
    },
    running: {
      color: "bg-amber-500 animate-pulse",
      label: "Processing...",
      bg: "bg-amber-500/10 border-amber-500/20 text-amber-400",
    },
    done: {
      color: "bg-emerald-500",
      label: "Completed",
      bg: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    },
    error: {
      color: "bg-rose-500",
      label: "Error Failed",
      bg: "bg-rose-500/10 border-rose-500/20 text-rose-400",
    },
  };

  const config = statusConfigs[status];

  return (
    <div className="w-full flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400">Pipeline Status</span>
        <div className={`inline-flex items-center gap-1.5 border px-2.5 py-0.5 rounded-full text-xs font-semibold ${config.bg}`}>
          <span className={`w-2 h-2 rounded-full ${config.color}`} />
          <span>{config.label}</span>
        </div>
      </div>
      {status === "error" && errorMessage && (
        <div className="text-xs text-rose-400 font-medium bg-rose-500/5 border border-rose-500/10 p-3 rounded-lg mt-1 select-text">
          {errorMessage}
        </div>
      )}
    </div>
  );
}
