import { usePipelineStore } from "../store/pipeline-store";

export function PreviewPane() {
  const currentPreviewUrl = usePipelineStore((state) => state.currentPreviewUrl);
  const status = usePipelineStore((state) => state.status);
  const previewEnabled = usePipelineStore((state) => state.previewEnabled);
  const setPreviewEnabled = usePipelineStore((state) => state.setPreviewEnabled);

  if (!previewEnabled) {
    return (
      <div className="w-full aspect-video rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center gap-3 text-slate-400 text-sm font-medium shadow-inner relative p-6">
        <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.895 7.895L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
        </svg>
        <div className="text-center">
          <p className="text-slate-300 font-semibold">Live Monitor Disabled</p>
          <p className="text-xs text-slate-500 mt-1">Rendering is suspended to optimize hardware processing speed.</p>
        </div>
        <button
          onClick={() => setPreviewEnabled(true)}
          className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white transition-all cursor-pointer shadow-md active:scale-[0.98]"
        >
          Enable Preview
        </button>
      </div>
    );
  }

  if (status === "idle") {
    return (
      <div className="w-full aspect-video rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-500 text-sm font-medium shadow-inner">
        Monitor Ready. Start a job to inspect live frames.
      </div>
    );
  }

  return (
    <div className="w-full aspect-video rounded-xl bg-slate-950 border border-slate-800 relative overflow-hidden flex items-center justify-center shadow-lg group">
      {currentPreviewUrl ? (
        <img
          src={currentPreviewUrl}
          alt="Live pipeline frame preview"
          className="w-full h-full object-contain select-none"
          draggable="false"
        />
      ) : (
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-slate-400 font-medium">Initializing pipeline...</span>
        </div>
      )}
      
      {/* Live Badge */}
      {status === "running" && (
        <span className="absolute top-3 left-3 bg-red-500/90 text-white text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded shadow flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 bg-white rounded-full animate-ping" />
          Live Preview
        </span>
      )}

      {/* Closable X Control */}
      <button
        onClick={() => setPreviewEnabled(false)}
        aria-label="Close preview"
        className="absolute top-3 right-3 w-7 h-7 bg-slate-900/80 hover:bg-rose-600 text-slate-400 hover:text-white rounded-lg flex items-center justify-center border border-slate-800 transition-all shadow-md cursor-pointer opacity-0 group-hover:opacity-100 focus:opacity-100"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}
