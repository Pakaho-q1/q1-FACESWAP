import { usePipelineStore } from "../store/pipeline-store";

export function ProgressBar() {
  const progress = usePipelineStore((state) => state.progress);
  const status = usePipelineStore((state) => state.status);

  if (status === "idle") return null;

  const { completed, total } = progress;
  const percentage = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0;

  return (
    <div className="w-full">
      <div className="flex justify-between mb-2 text-xs font-medium text-slate-300">
        <span>Processing Status</span>
        <span>{percentage}% ({completed}/{total})</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-4 overflow-hidden p-[2px] border border-slate-700 shadow-inner relative">
        <div
          className={`h-full transition-all duration-300 ease-out rounded-full bg-gradient-to-r ${
            status === "error"
              ? "from-rose-500 to-red-600"
              : status === "done"
              ? "from-emerald-400 to-teal-500"
              : "from-sky-400 via-violet-500 to-fuchsia-500 animate-pulse"
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
