import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { usePipelineStore } from "../../faceswap-pipeline/store/pipeline-store";
import type { PipelineOutput } from "../../faceswap-pipeline/store/pipeline-store";
import { openFolder } from "../../../lib/tauri-ipc";

// Detect if running inside Tauri
const isTauri = typeof window !== "undefined" && (window as any).__TAURI_INTERNALS__ !== undefined;

export interface OutputListProps {
  outputPath: string;
}

export function OutputList({ outputPath }: OutputListProps) {
  const status = usePipelineStore((state) => state.status);
  const outputs = usePipelineStore((state) => state.outputs);
  const setOutputs = usePipelineStore((state) => state.setOutputs);
  
  const [currentPage, setCurrentPage] = useState(1);
  const [previewItem, setPreviewItem] = useState<PipelineOutput | null>(null);
  
  const gridContainerRef = useRef<HTMLDivElement>(null);

  const rawHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
  const host = (rawHost === "localhost" || rawHost === "tauri.localhost" || rawHost.endsWith(".localhost") || rawHost === "localhost.localdomain" || rawHost === "[::1]" || !rawHost) ? "127.0.0.1" : rawHost;
  const apiBase = `http://${host}:8234`;

  const fetchOutputs = async () => {
    try {
      const res = await fetch(`${apiBase}/api/outputs?path=${encodeURIComponent(outputPath)}`);
      if (res.ok) {
        const data = await res.json();
        setOutputs(data);
      }
    } catch (err) {
      console.warn("Failed to fetch outputs:", err);
    }
  };

  // Poll outputs directory every 5 seconds
  useEffect(() => {
    fetchOutputs();
    const interval = setInterval(fetchOutputs, 5000);
    return () => clearInterval(interval);
  }, [outputPath]);

  // Reset to first page when files or path changes
  useEffect(() => {
    setCurrentPage(1);
    if (gridContainerRef.current) {
      gridContainerRef.current.scrollTop = 0;
    }
  }, [outputPath, outputs.length]);

  const handleDelete = async (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this output file?")) return;
    try {
      const res = await fetch(`${apiBase}/api/file?path=${encodeURIComponent(path)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        fetchOutputs();
      } else {
        alert("Failed to delete file from disk");
      }
    } catch (err: any) {
      alert(`Error deleting file: ${err?.message || err}`);
    }
  };

  // Pagination calculation: 24 items per page (4x6 grid)
  const itemsPerPage = 24;
  const totalPages = Math.ceil(outputs.length / itemsPerPage);
  const displayedItems = outputs.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div className="w-full p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl mt-6 flex flex-col gap-5">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-slate-200">Generated Outputs ({outputs.length})</h3>
          {status === "running" && (
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {/* Refresh Button */}
          <button
            onClick={fetchOutputs}
            aria-label="Refresh list"
            title="Refresh list"
            className="p-1.5 bg-slate-950 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 text-emerald-400 hover:text-emerald-300 rounded-lg transition-all cursor-pointer flex items-center justify-center shadow-md shrink-0"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.306 7M7 9h8V1" />
            </svg>
          </button>

          {/* Open Folder Button - Only visible in GUI/Tauri mode */}
          {isTauri && (
            <button
              onClick={() => openFolder(outputPath)}
              aria-label="Open output folder"
              title="Open output folder"
              className="p-1.5 bg-slate-950 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 rounded-lg text-indigo-400 hover:text-indigo-300 transition-all cursor-pointer flex items-center justify-center shadow-md shrink-0"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {outputs.length === 0 ? (
        <div className="py-10 flex flex-col items-center justify-center text-slate-500 text-sm font-medium">
          <svg className="w-8 h-8 text-slate-700 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 13.5h3.86a2.25 2.25 0 012.008 1.24l.885 1.77a2.25 2.25 0 002.007 1.24h1.98a2.25 2.25 0 002.007-1.24l.885-1.77a2.25 2.25 0 012.007-1.24h3.86m-18 0a7.5 7.5 0 0015 0m-15 0a7.5 7.5 0 1115 0" />
          </svg>
          <span>No output files found in this directory.</span>
        </div>
      ) : (
        <>
          <div
            ref={gridContainerRef}
            className="gap-4 max-h-[720px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent"
            style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gridAutoRows: 'auto' }}
          >
            {displayedItems.map((item) => {
              const fileUrl = `${apiBase}/api/file?path=${encodeURIComponent(item.path)}`;
              return (
                <div
                  key={item.id}
                  className="bg-slate-950/60 hover:bg-slate-950 border border-slate-800/80 hover:border-indigo-500/30 rounded-xl overflow-hidden flex flex-col transition-all group shadow-md relative aspect-video"
                >
                  {/* Media Content - 100% card size */}
                  <div className="w-full h-full bg-slate-950 flex items-center justify-center overflow-hidden relative">
                    {item.kind === "video" ? (
                      <div className="w-full h-full flex items-center justify-center relative">
                        <div className="absolute inset-0 bg-slate-950/20 group-hover:bg-slate-950/40 transition-all flex items-center justify-center z-0">
                          <span className="w-8 h-8 rounded-full bg-teal-500/25 border border-teal-500/40 flex items-center justify-center text-teal-400 group-hover:scale-110 transition-transform">
                            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                          </span>
                        </div>
                        <video src={fileUrl} className="w-full h-full object-contain bg-slate-950 opacity-80" muted />
                      </div>
                    ) : (
                      <img
                        src={fileUrl}
                        alt={item.name}
                        className="w-full h-full object-contain bg-slate-950 group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                    )}
                  </div>

                  {/* Absolute hover overlay containing only filename and buttons */}
                  <div className="absolute inset-0 bg-slate-950/85 opacity-0 group-hover:opacity-100 transition-all duration-200 flex flex-col justify-between p-3.5 z-10 invisible group-hover:visible">
                    {/* Top: Only filename */}
                    <span className="text-xs font-bold text-white truncate w-full" title={item.name}>
                      {item.name}
                    </span>

                    {/* Bottom: Action buttons */}
                    <div className="flex gap-1.5 w-full">
                      {/* Open Icon Button */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setPreviewItem(item);
                        }}
                        aria-label="Open preview"
                        title="Open preview"
                        className="flex-1 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition-all cursor-pointer shadow flex items-center justify-center gap-1"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>

                      {/* Delete Icon Button */}
                      <button
                        onClick={(e) => handleDelete(item.path, e)}
                        aria-label="Delete file"
                        title="Delete file"
                        className="px-2.5 py-1.5 bg-slate-900 hover:bg-rose-950/40 text-slate-400 hover:text-rose-400 border border-slate-800 hover:border-rose-900/60 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center justify-center"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Web Pagination controls */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-1.5 mt-2 border-t border-slate-800 pt-4 text-xs font-semibold select-none">
              {/* Prev Button */}
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                aria-label="Previous page"
                className="w-8 h-8 rounded-lg bg-slate-950 hover:bg-slate-850 border border-slate-800 disabled:opacity-30 disabled:pointer-events-none transition-all flex items-center justify-center cursor-pointer text-slate-400 hover:text-white font-bold"
              >
                &lt;
              </button>
              
              {/* Page Numbers */}
              {Array.from({ length: totalPages }, (_, idx) => {
                const pNum = idx + 1;
                if (
                  pNum === 1 ||
                  pNum === totalPages ||
                  Math.abs(pNum - currentPage) <= 1
                ) {
                  return (
                    <button
                      key={pNum}
                      onClick={() => setCurrentPage(pNum)}
                      className={`w-8 h-8 rounded-lg border transition-all cursor-pointer ${
                        currentPage === pNum
                          ? "bg-indigo-600 border-indigo-500 text-white font-bold"
                          : "bg-slate-950 border-slate-800 hover:bg-slate-850 text-slate-400 hover:text-white"
                      }`}
                    >
                      {pNum}
                    </button>
                  );
                }
                
                if (pNum === 2 || pNum === totalPages - 1) {
                  return (
                    <span key={pNum} className="px-0.5 text-slate-600 font-bold">
                      ...
                    </span>
                  );
                }
                
                return null;
              })}

              {/* Next Button */}
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                aria-label="Next page"
                className="w-8 h-8 rounded-lg bg-slate-950 hover:bg-slate-850 border border-slate-800 disabled:opacity-30 disabled:pointer-events-none transition-all flex items-center justify-center cursor-pointer text-slate-400 hover:text-white font-bold"
              >
                &gt;
              </button>
            </div>
          )}
        </>
      )}

      {/* Popup Preview Modal for Image/Video */}
      {previewItem && typeof document !== "undefined" && createPortal(
        <div className="fixed inset-0 bg-black/85 backdrop-blur-sm z-[9999] flex flex-col items-center justify-center p-4 animate-fade-in text-slate-200">
          <button
            onClick={() => setPreviewItem(null)}
            className="absolute top-4 right-4 text-slate-400 hover:text-white bg-slate-900/80 p-2 rounded-xl border border-slate-800 hover:border-slate-700 transition-all cursor-pointer z-10"
            aria-label="Close popup preview"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          <div className="w-full max-w-4xl bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
            <div className="flex-1 bg-black/60 flex items-center justify-center p-4 min-h-0 overflow-hidden">
              {previewItem.kind === "video" ? (
                <video
                  src={`${apiBase}/api/file?path=${encodeURIComponent(previewItem.path)}`}
                  controls
                  autoPlay
                  className="max-w-full max-h-[70vh] rounded-lg shadow-lg"
                />
              ) : (
                <img
                  src={`${apiBase}/api/file?path=${encodeURIComponent(previewItem.path)}`}
                  alt={previewItem.name}
                  className="max-w-full max-h-[70vh] rounded-lg object-contain shadow-lg"
                />
              )}
            </div>

            <div className="p-4 border-t border-slate-900 bg-slate-950 flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between shrink-0">
              <div className="min-w-0">
                <p className="text-sm font-bold text-white truncate">{previewItem.name}</p>
                <p className="text-xs text-slate-500 truncate mt-0.5">{previewItem.path}</p>
              </div>
              <div className="flex gap-2">
                {isTauri && (
                  <button
                    onClick={() => openFolder(previewItem.path)}
                    className="px-3.5 py-2 bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded-lg text-xs font-semibold text-slate-200 transition-all cursor-pointer"
                  >
                    Reveal on PC
                  </button>
                )}
                <button
                  onClick={() => setPreviewItem(null)}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition-all cursor-pointer"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
