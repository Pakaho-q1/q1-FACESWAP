import { useState, useEffect } from "react";
import { usePipelineStore } from "../../faceswap-pipeline/store/pipeline-store";
import { openFolder } from "../../../lib/tauri-ipc";

export interface OutputPreviewProps {
  outputPath?: string;
  kind?: "image" | "video";
}

export function OutputPreview({ outputPath, kind }: OutputPreviewProps) {
  const status = usePipelineStore((state) => state.status);
  const [resolvedPath, setResolvedPath] = useState<string>("");

  useEffect(() => {
    if (outputPath) {
      setResolvedPath(outputPath);
    }
  }, [outputPath]);

  if (status !== "done" || !resolvedPath) return null;

  const isVideo = kind === "video" || resolvedPath.match(/\.(mp4|mkv|avi|mov)$/i);

  return (
    <div className="w-full flex flex-col gap-3 p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg mt-4 animate-fade-in">
      <h3 className="text-sm font-semibold text-slate-200">Execution Output</h3>
      <div className="w-full aspect-video rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-800 relative">
        {isVideo ? (
          <video
            src={resolvedPath}
            controls
            className="w-full h-full object-contain"
          />
        ) : (
          <img
            src={resolvedPath}
            alt="Final face swap output"
            className="w-full h-full object-contain"
            loading="lazy"
          />
        )}
      </div>
      <div className="flex gap-2 justify-end items-center">
        <span className="text-xs text-slate-400 self-center truncate max-w-[250px] mr-auto">
          Saved: {resolvedPath}
        </span>
        <button
          onClick={() => {
            // Find parent folder of the final file path (handles windows/unix slashes)
            const lastSlash = Math.max(resolvedPath.lastIndexOf("/"), resolvedPath.lastIndexOf("\\"));
            const dir = lastSlash !== -1 ? resolvedPath.substring(0, lastSlash) : resolvedPath;
            openFolder(dir);
          }}
          className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white transition-all cursor-pointer flex items-center gap-1.5"
        >
          Open Folder
        </button>
      </div>
    </div>
  );
}
