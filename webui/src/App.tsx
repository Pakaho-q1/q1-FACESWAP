import { useState, useEffect } from "react";
import { usePipelineEvents } from "./features/faceswap-pipeline/hooks/use-pipeline-events";
import { usePipelineStore } from "./features/faceswap-pipeline/store/pipeline-store";
import { ProgressBar } from "./features/faceswap-pipeline/components/ProgressBar";
import { PreviewPane } from "./features/faceswap-pipeline/components/PreviewPane";
import { StatusIndicator } from "./features/faceswap-pipeline/components/StatusIndicator";
import { OutputList } from "./features/output/components/OutputList";
import { selectFile, selectDirectory } from "./lib/tauri-ipc";

// Detect if running inside Tauri or a normal web browser (LAN mobile)
const isTauri = typeof window !== "undefined" && (window as any).__TAURI_INTERNALS__ !== undefined;

export default function App() {
  // Subscribe to web server status poller
  usePipelineEvents();

  const status = usePipelineStore((state) => state.status);
  const progress = usePipelineStore((state) => state.progress);
  const tunerStatus = usePipelineStore((state) => state.tunerStatus);
  const resetStore = usePipelineStore((state) => state.reset);
  const error = usePipelineStore((state) => state.errorMessage);
  const setError = usePipelineStore((state) => state.setError);
  const setStatus = usePipelineStore((state) => state.setStatus);

  // Auto-dismiss errors after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError("");
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, setError]);

  // Collapsible panel states
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [developerOpen, setDeveloperOpen] = useState(false);
  const [customizeProviders, setCustomizeProviders] = useState(false);

  // Tier 1 - Main Form States
  const [inputFace, setInputFace] = useState("");
  const [inputTarget, setInputTarget] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [format, setFormat] = useState("video");
  const [enableSwapper, setEnableSwapper] = useState(true);
  const [swaperWeigh, setSwaperWeigh] = useState(0.70);

  const [enableRestore, setEnableRestore] = useState(true);
  const [restoreChoice, setRestoreChoice] = useState("1");
  const [restoreWeigh, setRestoreWeigh] = useState(0.70);
  const [restoreBlend, setRestoreBlend] = useState(0.70);

  const [enableParser, setEnableParser] = useState(true);

  // Tier 2 - Advanced Options
  // Performance / Tuner
  const [workersPerStage, setWorkersPerStage] = useState(8);
  const [workerQueueSize, setWorkerQueueSize] = useState(64);
  const [outQueueSize, setOutQueueSize] = useState(128);
  const [tunerMode, setTunerMode] = useState("auto");
  const [gpuTargetUtil, setGpuTargetUtil] = useState(95);
  const [highWatermark, setHighWatermark] = useState(12);
  const [lowWatermark, setLowWatermark] = useState(4);
  const [switchCooldown, setSwitchCooldown] = useState(0.35);

  // Providers
  const [providerAll, setProviderAll] = useState("trt");
  const [providerSwaper, setProviderSwaper] = useState("auto");
  const [providerRestore, setProviderRestore] = useState("auto");
  const [providerParser, setProviderParser] = useState("auto");
  const [providerDetect, setProviderDetect] = useState("auto");

  // Parser details
  const [preserveSwapEyes, setPreserveSwapEyes] = useState(true);
  const [parserMaskBlur, setParserMaskBlur] = useState(21);

  // Run behavior
  const [maxFrames, setMaxFrames] = useState(0);
  const [maxRetries, setMaxRetries] = useState(2);
  const [skipExisting, setSkipExisting] = useState(true);
  const [outputSuffix, setOutputSuffix] = useState("");
  const [fileSorting, setFileSorting] = useState("date_modified_newest");

  // Project settings
  const [projectPath, setProjectPath] = useState("");
  const [preloadModels, setPreloadModels] = useState(false);

  // Tier 3 - Developer Options
  const [dryRun, setDryRun] = useState(false);
  const [printEffectiveConfig, setPrintEffectiveConfig] = useState(false);
  const [logLevel, setLogLevel] = useState("warning");

  // Load settings once at startup
  useEffect(() => {
    const saved = localStorage.getItem("q1_faceswap_settings");
    if (saved) {
      try {
        const cfg = JSON.parse(saved);
        if (cfg.inputFace !== undefined) setInputFace(cfg.inputFace);
        if (cfg.inputTarget !== undefined) setInputTarget(cfg.inputTarget);
        if (cfg.outputPath !== undefined) setOutputPath(cfg.outputPath);
        if (cfg.format !== undefined) setFormat(cfg.format);
        if (cfg.enableSwapper !== undefined) setEnableSwapper(cfg.enableSwapper);
        if (cfg.swaperWeigh !== undefined) setSwaperWeigh(cfg.swaperWeigh);
        if (cfg.enableRestore !== undefined) setEnableRestore(cfg.enableRestore);
        if (cfg.restoreChoice !== undefined) setRestoreChoice(cfg.restoreChoice);
        if (cfg.restoreWeigh !== undefined) setRestoreWeigh(cfg.restoreWeigh);
        if (cfg.restoreBlend !== undefined) setRestoreBlend(cfg.restoreBlend);
        if (cfg.enableParser !== undefined) setEnableParser(cfg.enableParser);
        if (cfg.workersPerStage !== undefined) setWorkersPerStage(cfg.workersPerStage);
        if (cfg.workerQueueSize !== undefined) setWorkerQueueSize(cfg.workerQueueSize);
        if (cfg.outQueueSize !== undefined) setOutQueueSize(cfg.outQueueSize);
        if (cfg.tunerMode !== undefined) setTunerMode(cfg.tunerMode);
        if (cfg.gpuTargetUtil !== undefined) setGpuTargetUtil(cfg.gpuTargetUtil);
        if (cfg.highWatermark !== undefined) setHighWatermark(cfg.highWatermark);
        if (cfg.lowWatermark !== undefined) setLowWatermark(cfg.lowWatermark);
        if (cfg.switchCooldown !== undefined) setSwitchCooldown(cfg.switchCooldown);
        if (cfg.providerAll !== undefined) setProviderAll(cfg.providerAll);
        if (cfg.providerSwaper !== undefined) setProviderSwaper(cfg.providerSwaper);
        if (cfg.providerRestore !== undefined) setProviderRestore(cfg.providerRestore);
        if (cfg.providerParser !== undefined) setProviderParser(cfg.providerParser);
        if (cfg.providerDetect !== undefined) setProviderDetect(cfg.providerDetect);
        if (cfg.preserveSwapEyes !== undefined) setPreserveSwapEyes(cfg.preserveSwapEyes);
        if (cfg.parserMaskBlur !== undefined) setParserMaskBlur(cfg.parserMaskBlur);
        if (cfg.maxFrames !== undefined) setMaxFrames(cfg.maxFrames);
        if (cfg.maxRetries !== undefined) setMaxRetries(cfg.maxRetries);
        if (cfg.skipExisting !== undefined) setSkipExisting(cfg.skipExisting);
        if (cfg.outputSuffix !== undefined) setOutputSuffix(cfg.outputSuffix);
        if (cfg.fileSorting !== undefined) setFileSorting(cfg.fileSorting);
        if (cfg.projectPath !== undefined) setProjectPath(cfg.projectPath);
        if (cfg.preloadModels !== undefined) setPreloadModels(cfg.preloadModels);
        if (cfg.dryRun !== undefined) setDryRun(cfg.dryRun);
        if (cfg.printEffectiveConfig !== undefined) setPrintEffectiveConfig(cfg.printEffectiveConfig);
        if (cfg.logLevel !== undefined) setLogLevel(cfg.logLevel);
      } catch (e) {
        console.error("Failed to parse settings", e);
      }
    }
  }, []);

  // Save settings automatically on any change
  useEffect(() => {
    const cfg = {
      inputFace,
      inputTarget,
      outputPath,
      format,
      enableSwapper,
      swaperWeigh,
      enableRestore,
      restoreChoice,
      restoreWeigh,
      restoreBlend,
      enableParser,
      workersPerStage,
      workerQueueSize,
      outQueueSize,
      tunerMode,
      gpuTargetUtil,
      highWatermark,
      lowWatermark,
      switchCooldown,
      providerAll,
      providerSwaper,
      providerRestore,
      providerParser,
      providerDetect,
      preserveSwapEyes,
      parserMaskBlur,
      maxFrames,
      maxRetries,
      skipExisting,
      outputSuffix,
      fileSorting,
      projectPath,
      preloadModels,
      dryRun,
      printEffectiveConfig,
      logLevel,
    };
    localStorage.setItem("q1_faceswap_settings", JSON.stringify(cfg));
  }, [
    inputFace,
    inputTarget,
    outputPath,
    format,
    enableSwapper,
    swaperWeigh,
    enableRestore,
    restoreChoice,
    restoreWeigh,
    restoreBlend,
    enableParser,
    workersPerStage,
    workerQueueSize,
    outQueueSize,
    tunerMode,
    gpuTargetUtil,
    highWatermark,
    lowWatermark,
    switchCooldown,
    providerAll,
    providerSwaper,
    providerRestore,
    providerParser,
    providerDetect,
    preserveSwapEyes,
    parserMaskBlur,
    maxFrames,
    maxRetries,
    skipExisting,
    outputSuffix,
    fileSorting,
    projectPath,
    preloadModels,
    dryRun,
    printEffectiveConfig,
    logLevel,
  ]);

  const handleResetDefaults = () => {
    if (confirm("Reset all settings to default values?")) {
      setInputFace("");
      setInputTarget("");
      setOutputPath("");
      setFormat("video");
      setEnableSwapper(true);
      setSwaperWeigh(0.70);
      setEnableRestore(true);
      setRestoreChoice("1");
      setRestoreWeigh(0.70);
      setRestoreBlend(0.70);
      setEnableParser(true);
      setWorkersPerStage(8);
      setWorkerQueueSize(64);
      setOutQueueSize(128);
      setTunerMode("auto");
      setGpuTargetUtil(95);
      setHighWatermark(12);
      setLowWatermark(4);
      setSwitchCooldown(0.35);
      setProviderAll("trt");
      setProviderSwaper("auto");
      setProviderRestore("auto");
      setProviderParser("auto");
      setProviderDetect("auto");
      setPreserveSwapEyes(true);
      setParserMaskBlur(21);
      setMaxFrames(0);
      setMaxRetries(2);
      setSkipExisting(true);
      setOutputSuffix("");
      setFileSorting("date_modified_newest");
      setProjectPath("");
      setPreloadModels(false);
      setDryRun(false);
      setPrintEffectiveConfig(false);
      setLogLevel("warning");
      localStorage.removeItem("q1_faceswap_settings");
    }
  };

  const handleWebUpload = async (event: React.ChangeEvent<HTMLInputElement>, targetField: "face" | "target") => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const host = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
      const actualHost = (host === "localhost" || host === "tauri.localhost" || host.endsWith(".localhost") || host === "localhost.localdomain" || host === "[::1]" || !host) ? "127.0.0.1" : host;

      const fileData = await file.arrayBuffer();
      const res = await fetch(`http://${actualHost}:8234/api/upload?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream"
        },
        body: fileData
      });

      if (res.ok) {
        const data = await res.json();
        if (data.path) {
          if (targetField === "face") {
            setInputFace(data.path);
          } else {
            setInputTarget(data.path);
            const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
            if ([".mp4", ".mkv", ".avi", ".mov"].includes(ext)) {
              setFormat("video");
            } else {
              setFormat("image");
            }
          }
        }
      } else {
        alert("Upload failed");
      }
    } catch (err: any) {
      alert(`Upload error: ${err?.message || err}`);
    }
  };

  // Pickers Handler
  const handleSelectFace = async () => {
    if (!isTauri) return;
    const file = await selectFile(["jpg", "jpeg", "png", "webp", "bmp", "safetensors"]);
    if (file) setInputFace(file);
  };

  const handleSelectTargetFile = async () => {
    if (!isTauri) return;
    const file = await selectFile(["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mkv", "avi", "mov"]);
    if (file) {
      setInputTarget(file);
      const ext = file.substring(file.lastIndexOf(".")).toLowerCase();
      if ([".mp4", ".mkv", ".avi", ".mov"].includes(ext)) {
        setFormat("video");
      } else {
        setFormat("image");
      }
    }
  };

  const handleSelectTargetFolder = async () => {
    if (!isTauri) return;
    const dir = await selectDirectory();
    if (dir) setInputTarget(dir);
  };

  const handleSelectOutput = async () => {
    if (!isTauri) return;
    const dir = await selectDirectory();
    if (dir) setOutputPath(dir);
  };

  const handleSelectProjectPath = async () => {
    if (!isTauri) return;
    const dir = await selectDirectory();
    if (dir) setProjectPath(dir);
  };

  // Submit & Cancel
  const handleStart = async () => {
    if (!inputFace || !inputTarget) {
      alert("Please specify both Source Face and Target Target");
      return;
    }
    if (enableParser && parserMaskBlur % 2 === 0) {
      alert("Parser mask blur must be an odd number");
      return;
    }

    try {
      resetStore();
      setStatus("running");

      const payload = {
        INPUT_FACE: inputFace,
        INPUT_TARGET: inputTarget,
        OUTPUT_PATH: outputPath,
        FORMAT: format,
        USE_SWAPER: enableSwapper,
        SWAPER_WEIGH: swaperWeigh,
        USE_RESTORE: enableRestore,
        RESTORE_CHOICE: restoreChoice,
        RESTORE_WEIGH: restoreWeigh,
        RESTORE_BLEND: restoreBlend,
        USE_PARSER: enableParser,
        PRESERVE_SWAP_EYES: preserveSwapEyes,
        PARSER_MASK_BLUR: parserMaskBlur,
        PROVIDER_ALL: providerAll,
        PROVIDER_SWAPER: providerSwaper,
        PROVIDER_RESTORE: providerRestore,
        PROVIDER_PARSER: providerParser,
        PROVIDER_DETECT: providerDetect,
        WORKERS_PER_STAGE: workersPerStage,
        WORKER_QUEUE_SIZE: workerQueueSize,
        OUT_QUEUE_SIZE: outQueueSize,
        TUNER_MODE: tunerMode,
        GPU_TARGET_UTIL: gpuTargetUtil,
        HIGH_WATERMARK: highWatermark,
        LOW_WATERMARK: lowWatermark,
        SWITCH_COOLDOWN_S: switchCooldown,
        MAX_FRAMES: maxFrames,
        MAX_RETRIES: maxRetries,
        SKIP_EXISTING: skipExisting,
        OUTPUT_SUFFIX: outputSuffix,
        FILE_SORTING: fileSorting,
        PROJECT_PATH: projectPath,
        PRELOAD_MODELS: preloadModels,
        DRY_RUN: dryRun,
        PRINT_EFFECTIVE_CONFIG: printEffectiveConfig,
        LOG_LEVEL: logLevel,
      };

      // Always send start command to web server (keeps PC and mobile in 100% sync!)
      const rawHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
      const host = (rawHost === "localhost" || rawHost === "tauri.localhost" || rawHost.endsWith(".localhost") || rawHost === "localhost.localdomain" || rawHost === "[::1]" || !rawHost) ? "127.0.0.1" : rawHost;
      const res = await fetch(`http://${host}:8234/api/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Failed to start job via web server");
    } catch (err: any) {
      setError(err?.message || String(err));
    }
  };

  const handleCancel = async () => {
    try {
      const rawHost = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
      const host = (rawHost === "localhost" || rawHost === "tauri.localhost" || rawHost.endsWith(".localhost") || rawHost === "localhost.localdomain" || rawHost === "[::1]" || !rawHost) ? "127.0.0.1" : rawHost;
      const res = await fetch(`http://${host}:8234/api/cancel`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to cancel job via web server");
      setStatus("idle");
    } catch (err: any) {
      alert(`Failed to cancel: ${err?.message || err}`);
    }
  };

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-100 font-sans flex flex-col overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-slate-900 bg-slate-950 px-6 flex justify-between items-center shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-xs text-white shadow-md">
            Q1
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight">
              q1-FaceSwap {!isTauri && <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded font-mono ml-2 border border-emerald-500/20">LAN Browser Control</span>}
            </h1>
            <p className="text-[10px] text-slate-500">Desktop Inference Swarm Engine</p>
          </div>
        </div>
        <StatusIndicator />
      </header>

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden h-[calc(100vh-3.5rem)]">
        {/* Left Sidebar (fixed width, scrolling controls) */}
        <aside className="w-80 border-r border-slate-900 bg-slate-900/40 flex flex-col h-full shrink-0 overflow-hidden">
          <div className="flex-1 overflow-y-auto min-h-0 p-4 flex flex-col gap-5">
            {/* Common Inputs */}
            <div className="flex flex-col gap-3.5">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest border-b border-slate-900 pb-1">
                Inputs Setup
              </h3>

              {/* Source Face */}
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-semibold text-slate-400">Source Face</label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    placeholder={isTauri ? "None chosen..." : "Type PC face path / name..."}
                    value={inputFace}
                    onChange={(e) => setInputFace(e.target.value)}
                    className="flex-1 min-w-0 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                  />
                  {isTauri && (
                    <button
                      onClick={handleSelectFace}
                      aria-label="Select source face file"
                      title="Select source face file"
                      className="px-2.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center justify-center shrink-0"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                      </svg>
                    </button>
                  )}
                  {!isTauri && (
                    <label className="px-2.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center justify-center shrink-0">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      <input
                        type="file"
                        accept="image/*,.safetensors"
                        className="hidden"
                        onChange={(e) => handleWebUpload(e, "face")}
                      />
                    </label>
                  )}
                </div>
              </div>

              {/* Target Target */}
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-semibold text-slate-400">Target Target</label>
                <div className="flex gap-1.5 w-full items-center">
                  <input
                    type="text"
                    placeholder={isTauri ? "None chosen..." : "Type PC target file / folder path..."}
                    value={inputTarget}
                    onChange={(e) => setInputTarget(e.target.value)}
                    className="flex-1 min-w-0 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                  />
                  {isTauri && (
                    <div className="flex gap-1.5 shrink-0">
                      {/* Pick File Icon */}
                      <button
                        onClick={handleSelectTargetFile}
                        aria-label="Select target file"
                        title="Select target file"
                        className="p-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-all cursor-pointer flex items-center justify-center"
                      >
                        <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                      </button>
                      {/* Pick Folder Icon */}
                      <button
                        onClick={handleSelectTargetFolder}
                        aria-label="Select target folder"
                        title="Select target folder"
                        className="p-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-all cursor-pointer flex items-center justify-center"
                      >
                        <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                      </button>
                    </div>
                  )}
                  {!isTauri && (
                    <label className="p-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-all cursor-pointer flex items-center justify-center shrink-0">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      <input
                        type="file"
                        accept="image/*,video/*"
                        className="hidden"
                        onChange={(e) => handleWebUpload(e, "target")}
                      />
                    </label>
                  )}
                </div>
              </div>

              {/* Output Directory */}
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-semibold text-slate-400">Output Folder</label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    placeholder={isTauri ? "Default folder..." : "Type PC output folder path..."}
                    value={outputPath}
                    onChange={(e) => setOutputPath(e.target.value)}
                    className="flex-1 min-w-0 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                  />
                  {isTauri && (
                    <button
                      onClick={handleSelectOutput}
                      aria-label="Select output folder"
                      title="Select output folder"
                      className="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center justify-center shrink-0"
                    >
                      <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              {/* Format Toggle */}
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-semibold text-slate-400">Format</label>
                <div className="flex bg-slate-950 border border-slate-800 rounded-lg p-[3px]">
                  <button
                    onClick={() => setFormat("video")}
                    className={`flex-1 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer ${format === "video" ? "bg-indigo-600 text-white" : "text-slate-500 hover:text-slate-300"
                      }`}
                  >
                    Video
                  </button>
                  <button
                    onClick={() => setFormat("image")}
                    className={`flex-1 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer ${format === "image" ? "bg-indigo-600 text-white" : "text-slate-500 hover:text-slate-300"
                      }`}
                  >
                    Image
                  </button>
                </div>
              </div>
            </div>

            {/* Pipeline Configuration Switches */}
            <div className="flex flex-col gap-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest border-b border-slate-900 pb-1">
                Core Pipeline
              </h3>

              {/* Swapper Switch */}
              <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-3.5 flex flex-col gap-3">
                <label className="flex items-center justify-between text-xs font-semibold text-slate-300 cursor-pointer">
                  <span>Enable Swapper Stage</span>
                  <input
                    type="checkbox"
                    checked={enableSwapper}
                    onChange={(e) => setEnableSwapper(e.target.checked)}
                    className="accent-indigo-500 w-4 h-4 rounded border-slate-800"
                  />
                </label>
                {enableSwapper && (
                  <div className="flex flex-col gap-1 pl-1 border-l-2 border-indigo-500/30 ml-1">
                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>Blend Weight</span>
                      <span className="font-bold text-indigo-400">{swaperWeigh.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={swaperWeigh}
                      onChange={(e) => setSwaperWeigh(parseFloat(e.target.value))}
                      className="w-full accent-indigo-500"
                    />
                  </div>
                )}
              </div>

              {/* Restore Switch */}
              <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-3.5 flex flex-col gap-3">
                <label className="flex items-center justify-between text-xs font-semibold text-slate-300 cursor-pointer">
                  <span>Enable Face Restore</span>
                  <input
                    type="checkbox"
                    checked={enableRestore}
                    onChange={(e) => setEnableRestore(e.target.checked)}
                    className="accent-indigo-500 w-4 h-4 rounded border-slate-800"
                  />
                </label>
                {enableRestore && (
                  <div className="flex flex-col gap-3 pl-1 border-l-2 border-indigo-500/30 ml-1">
                    {/* Restore Choice */}
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Model Choice</label>
                      <select
                        value={restoreChoice}
                        onChange={(e) => setRestoreChoice(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200"
                      >
                        <option value="1">GFPGAN v1.4</option>
                        <option value="2">GPEN 512</option>
                        <option value="3">GPEN 1024</option>
                        <option value="4">CodeFormer</option>
                      </select>
                    </div>
                    {/* Restore Weight */}
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Fidelity Weight</span>
                        <span className="font-bold text-indigo-400">{restoreWeigh.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={restoreWeigh}
                        onChange={(e) => setRestoreWeigh(parseFloat(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                    {/* Restore Blend */}
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Blend Ratio</span>
                        <span className="font-bold text-indigo-400">{restoreBlend.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={restoreBlend}
                        onChange={(e) => setRestoreBlend(parseFloat(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Parser Switch */}
              <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-3.5 flex flex-col gap-3">
                <label className="flex items-center justify-between text-xs font-semibold text-slate-300 cursor-pointer">
                  <span>Enable Mask Parser</span>
                  <input
                    type="checkbox"
                    checked={enableParser}
                    onChange={(e) => setEnableParser(e.target.checked)}
                    className="accent-indigo-500 w-4 h-4 rounded border-slate-800"
                  />
                </label>
              </div>
            </div>

            <div className="border border-slate-900 rounded-xl overflow-hidden bg-slate-950/20">
              <button
                onClick={() => setAdvancedOpen(!advancedOpen)}
                className="w-full p-3 flex justify-between items-center text-xs font-bold text-slate-300 hover:bg-slate-900/50 transition-colors cursor-pointer"
              >
                <span>Advanced Options</span>
                <span className="text-slate-500">{advancedOpen ? "▼" : "▶"}</span>
              </button>

              {advancedOpen && (
                <div className="p-4 border-t border-slate-900 flex flex-col gap-4.5 bg-slate-900/10">
                  {/* Performance / Tuner */}
                  <div className="flex flex-col gap-3">
                    <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Performance / Tuner</h4>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Workers per Stage</label>
                      <input
                        type="number"
                        min="1"
                        max="128"
                        value={workersPerStage}
                        onChange={(e) => setWorkersPerStage(parseInt(e.target.value) || 8)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Worker Queue Size</label>
                      <input
                        type="number"
                        min="4"
                        max="4096"
                        value={workerQueueSize}
                        onChange={(e) => setWorkerQueueSize(parseInt(e.target.value) || 64)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Out Queue Size</label>
                      <input
                        type="number"
                        min="8"
                        max="8192"
                        value={outQueueSize}
                        onChange={(e) => setOutQueueSize(parseInt(e.target.value) || 128)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Tuner Mode</label>
                      <select
                        value={tunerMode}
                        onChange={(e) => setTunerMode(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      >
                        <option value="auto">Auto</option>
                        <option value="max_util">Max Utilization</option>
                        <option value="stable">Stable</option>
                      </select>
                    </div>

                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Target GPU Util</span>
                        <span>{gpuTargetUtil}%</span>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="100"
                        value={gpuTargetUtil}
                        onChange={(e) => setGpuTargetUtil(parseInt(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">High Watermark</label>
                      <input
                        type="number"
                        value={highWatermark}
                        onChange={(e) => setHighWatermark(parseInt(e.target.value) || 12)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Low Watermark</label>
                      <input
                        type="number"
                        value={lowWatermark}
                        onChange={(e) => setLowWatermark(parseInt(e.target.value) || 4)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Switch Cooldown (s)</label>
                      <input
                        type="number"
                        step="0.05"
                        value={switchCooldown}
                        onChange={(e) => setSwitchCooldown(parseFloat(e.target.value) || 0.35)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>
                  </div>

                  {/* Providers */}
                  <div className="flex flex-col gap-3">
                    <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Providers</h4>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Default Provider</label>
                      <select
                        value={providerAll}
                        onChange={(e) => setProviderAll(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      >
                        <option value="trt">TensorRT</option>
                        <option value="cuda">CUDA</option>
                        <option value="cpu">CPU</option>
                      </select>
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="flex items-center gap-2 text-[10px] text-slate-400 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={customizeProviders}
                          onChange={(e) => setCustomizeProviders(e.target.checked)}
                          className="accent-indigo-500"
                        />
                        <span>Customize per stage</span>
                      </label>

                      {customizeProviders && (
                        <div className="flex flex-col gap-2 pl-2 border-l border-slate-800 ml-1">
                          <div className="flex flex-col gap-1">
                            <label className="text-[9px] text-slate-500">Detect Provider</label>
                            <select
                              value={providerDetect}
                              onChange={(e) => setProviderDetect(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-0.5 text-xs text-slate-200"
                            >
                              <option value="auto">Auto ({providerAll})</option>
                              <option value="trt">TensorRT</option>
                              <option value="cuda">CUDA</option>
                              <option value="cpu">CPU</option>
                            </select>
                          </div>
                          <div className="flex flex-col gap-1">
                            <label className="text-[9px] text-slate-500">Swaper Provider</label>
                            <select
                              value={providerSwaper}
                              onChange={(e) => setProviderSwaper(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-0.5 text-xs text-slate-200"
                            >
                              <option value="auto">Auto ({providerAll})</option>
                              <option value="trt">TensorRT</option>
                              <option value="cuda">CUDA</option>
                              <option value="cpu">CPU</option>
                            </select>
                          </div>
                          <div className="flex flex-col gap-1">
                            <label className="text-[9px] text-slate-500">Restore Provider</label>
                            <select
                              value={providerRestore}
                              onChange={(e) => setProviderRestore(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-0.5 text-xs text-slate-200"
                            >
                              <option value="auto">Auto ({providerAll})</option>
                              <option value="trt">TensorRT</option>
                              <option value="cuda">CUDA</option>
                              <option value="cpu">CPU</option>
                            </select>
                          </div>
                          <div className="flex flex-col gap-1">
                            <label className="text-[9px] text-slate-500">Parser Provider</label>
                            <select
                              value={providerParser}
                              onChange={(e) => setProviderParser(e.target.value)}
                              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-0.5 text-xs text-slate-200"
                            >
                              <option value="auto">Auto ({providerAll})</option>
                              <option value="trt">TensorRT</option>
                              <option value="cuda">CUDA</option>
                              <option value="cpu">CPU</option>
                            </select>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Parser Details */}
                  {enableParser && (
                    <div className="flex flex-col gap-3">
                      <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Parser Details</h4>
                      <label className="flex items-center gap-2 text-[10px] text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={preserveSwapEyes}
                          onChange={(e) => setPreserveSwapEyes(e.target.checked)}
                          className="accent-indigo-500"
                        />
                        <span>Preserve Swapped Eyes</span>
                      </label>

                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-slate-400">Mask Blur (Odd Number)</label>
                        <input
                          type="number"
                          step="2"
                          min="1"
                          max="255"
                          value={parserMaskBlur}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) || 21;
                            setParserMaskBlur(val);
                          }}
                          className={`w-full bg-slate-950 border rounded-lg px-2 py-1 text-xs text-slate-200 ${parserMaskBlur % 2 === 0 ? "border-rose-500" : "border-slate-800"
                            }`}
                        />
                      </div>
                    </div>
                  )}

                  {/* Run Behavior */}
                  <div className="flex flex-col gap-3">
                    <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Run Behavior</h4>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Max Frames (0 = No limit)</label>
                      <input
                        type="number"
                        min="0"
                        value={maxFrames}
                        onChange={(e) => setMaxFrames(parseInt(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Max Retries</label>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={maxRetries}
                        onChange={(e) => setMaxRetries(parseInt(e.target.value) || 2)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      />
                    </div>

                    <label className="flex items-center gap-2 text-[10px] text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={skipExisting}
                        onChange={(e) => setSkipExisting(e.target.checked)}
                        className="accent-indigo-500"
                      />
                      <span>Skip Existing Outputs</span>
                    </label>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Output Suffix</label>
                      <input
                        type="text"
                        placeholder="e.g. _swapped"
                        value={outputSuffix}
                        onChange={(e) => setOutputSuffix(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200 placeholder-slate-600"
                      />
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">File Sorting</label>
                      <select
                        value={fileSorting}
                        onChange={(e) => setFileSorting(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                      >
                        <option value="date_modified_newest">Date Modified (Newest)</option>
                        <option value="date_modified_oldest">Date Modified (Oldest)</option>
                        <option value="name_ascending">Name (Ascending)</option>
                        <option value="name_descending">Name (Descending)</option>
                      </select>
                    </div>
                  </div>

                  {/* Project Settings */}
                  <div className="flex flex-col gap-3">
                    <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Project Settings</h4>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-slate-400">Project Workspace Path</label>
                      <div className="flex gap-1.5">
                        <input
                          type="text"
                          placeholder={isTauri ? "Default project dir..." : "Type PC project workspace path..."}
                          value={projectPath}
                          onChange={(e) => setProjectPath(e.target.value)}
                          className="flex-1 min-w-0 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                        />
                        {isTauri && (
                          <button
                            onClick={handleSelectProjectPath}
                            aria-label="Select workspace folder"
                            title="Select workspace folder"
                            className="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded-lg text-xs font-semibold cursor-pointer flex items-center justify-center shrink-0"
                          >
                            <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                            </svg>
                          </button>
                        )}
                      </div>
                    </div>

                    <label className="flex items-center gap-2 text-[10px] text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preloadModels}
                        onChange={(e) => setPreloadModels(e.target.checked)}
                        className="accent-indigo-500"
                      />
                      <span>Preload Missing Models</span>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Tier 3: Developer collapsible panel */}
            <div className="border border-slate-900 rounded-xl overflow-hidden bg-slate-950/20 mb-4">
              <button
                onClick={() => setDeveloperOpen(!developerOpen)}
                className="w-full p-3 flex justify-between items-center text-xs font-bold text-slate-400 hover:bg-slate-900/50 transition-colors cursor-pointer"
              >
                <span>Developer Options</span>
                <span className="text-slate-600">{developerOpen ? "▼" : "▶"}</span>
              </button>

              {developerOpen && (
                <div className="p-4 border-t border-slate-900 flex flex-col gap-4 bg-slate-900/10">
                  <label className="flex items-center gap-2 text-[10px] text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={dryRun}
                      onChange={(e) => setDryRun(e.target.checked)}
                      className="accent-indigo-500"
                    />
                    <span>Dry Run (Validate Only)</span>
                  </label>

                  <label className="flex items-center gap-2 text-[10px] text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={printEffectiveConfig}
                      onChange={(e) => setPrintEffectiveConfig(e.target.checked)}
                      className="accent-indigo-500"
                    />
                    <span>Print Effective Config</span>
                  </label>

                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-slate-400">Log Level</label>
                    <select
                      value={logLevel}
                      onChange={(e) => setLogLevel(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
                    >
                      <option value="warning">Warning</option>
                      <option value="info">Info</option>
                      <option value="debug">Debug</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons (fixed at bottom of sidebar) */}
          <div className="p-4 bg-slate-950 border-t border-slate-900 flex gap-2 shrink-0">
            <button
              onClick={handleStart}
              disabled={status === "running"}
              className={`flex-1 py-2.5 rounded-lg font-bold text-xs text-white tracking-wide transition-all shadow-md active:scale-[0.98] cursor-pointer ${status === "running"
                ? "bg-slate-800 text-slate-500 cursor-not-allowed shadow-none border border-slate-700/50"
                : "bg-indigo-600 hover:bg-indigo-500"
                }`}
            >
              Start Swap
            </button>
            <button
              onClick={handleCancel}
              disabled={status !== "running"}
              className={`px-4 py-2.5 rounded-lg font-bold text-xs transition-all border active:scale-[0.98] cursor-pointer ${status === "running"
                ? "bg-rose-950/20 text-rose-400 border-rose-900/60 hover:bg-rose-900/40 hover:text-white"
                : "bg-slate-950 text-slate-700 border-slate-900 cursor-not-allowed"
                }`}
            >
              Cancel
            </button>
            <button
              onClick={handleResetDefaults}
              title="Reset Settings to Defaults"
              className="px-3 py-2.5 rounded-lg bg-slate-950 border border-slate-900 text-slate-500 hover:text-slate-200 hover:bg-slate-900 hover:border-slate-800 transition-all cursor-pointer flex items-center justify-center shrink-0"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18" />
              </svg>
            </button>
          </div>
        </aside>

        {/* Right Panel (scrolling preview and output list) */}
        <main className="flex-1 bg-slate-950 p-6 overflow-y-auto flex flex-col gap-6">
          {/* Swarm Engine Diagnostics Header */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl w-full flex flex-col sm:flex-row justify-between gap-4 text-xs">
            <div className="flex flex-col gap-1.5 flex-1 min-w-0">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Swarm Engine Diagnostics</span>
              {tunerStatus ? (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-slate-200">
                  <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    GPU Util: <strong className="text-emerald-400">{tunerStatus.gpu_util}%</strong>
                  </span>
                  <span>
                    Tuner: <strong className="text-indigo-400">{tunerStatus.mode_name}</strong>
                  </span>
                  <span>
                    Hot Stage: <strong className="text-pink-400">{tunerStatus.hot_stage || "None"}</strong>
                  </span>
                  {tunerStatus.ordered_stages && (
                    <span className="text-[11px] text-slate-400 flex flex-wrap gap-1 items-center">
                      Queues: {tunerStatus.ordered_stages.map((stage: string) => (
                        <span key={stage} className="font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-850 text-[10px]">
                          {stage}:<strong className="text-indigo-400">{tunerStatus.sizes[stage] ?? 0}</strong>
                        </span>
                      ))}
                    </span>
                  )}
                </div>
              ) : (
                <span className="text-slate-500 italic">Swarm Engine is offline. Start a job to activate metrics.</span>
              )}
            </div>

            <div className="flex flex-col gap-1 sm:text-right shrink-0 border-t sm:border-t-0 sm:border-l border-slate-800 pt-2 sm:pt-0 sm:pl-4 min-w-[150px]">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Processed Frames / Items</span>
              <span className="text-sm font-extrabold text-white">
                {progress.completed} <span className="text-slate-500 text-xs font-normal">/</span> {progress.total}
              </span>
              {progress.label && (
                <span className="text-[10px] text-indigo-400 truncate max-w-[200px] block" title={progress.label}>
                  {progress.label}
                </span>
              )}
            </div>
          </div>

          {/* Live Preview Monitor Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl w-full">
            <div className="flex justify-between items-center border-b border-slate-800 pb-2 mb-4">
              <h2 className="text-md font-bold text-white tracking-wide">
                Live Monitor
              </h2>
            </div>
            <PreviewPane />
            <div className="mt-5">
              <ProgressBar />
            </div>
          </div>

          {/* Generated Output List Card */}
          <OutputList outputPath={outputPath} />
        </main>
      </div>

      {error && (
        <div className="fixed bottom-6 right-6 bg-slate-900/95 backdrop-blur border border-rose-500/30 text-rose-200 p-4 rounded-xl shadow-2xl z-50 flex items-center gap-3 animate-fade-in max-w-sm">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse shrink-0" />
          <div className="flex-1 text-xs">
            <p className="font-bold text-rose-400 mb-0.5">Pipeline Error (Auto-clears in 5s)</p>
            <p className="opacity-90 leading-relaxed">{error}</p>
          </div>
          <button
            onClick={() => setError("")}
            className="text-slate-400 hover:text-white text-xs font-bold font-mono px-1.5 py-0.5 hover:bg-slate-850 rounded transition-all cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
