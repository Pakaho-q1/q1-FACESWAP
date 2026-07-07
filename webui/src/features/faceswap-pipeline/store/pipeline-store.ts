import { create } from "zustand";

export interface PipelineOutput {
  id: string;
  path: string;
  name: string;
  kind: "image" | "video";
}

export interface PipelineState {
  status: "idle" | "running" | "done" | "error";
  progress: { completed: number; total: number; label: string };
  currentPreviewUrl: string;
  errorMessage: string;
  previewEnabled: boolean;
  outputs: PipelineOutput[];
  tunerStatus: any;

  setStatus: (status: "idle" | "running" | "done" | "error") => void;
  setProgress: (completed: number, total: number, label?: string) => void;
  setPreviewUrl: (url: string) => void;
  setError: (message: string) => void;
  setPreviewEnabled: (enabled: boolean) => void;
  setOutputs: (outputs: PipelineOutput[]) => void;
  addOutput: (path: string, name: string, kind: "image" | "video") => void;
  setTunerStatus: (status: any) => void;
  reset: () => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  status: "idle",
  progress: { completed: 0, total: 0, label: "" },
  currentPreviewUrl: "",
  errorMessage: "",
  previewEnabled: true,
  outputs: [],
  tunerStatus: null,

  setStatus: (status) => set({ status }),
  setProgress: (completed, total, label = "") =>
    set((state) => ({
      progress: { completed, total, label: label || state.progress.label },
    })),
  setPreviewUrl: (currentPreviewUrl) =>
    set((state) => {
      if (!state.previewEnabled) return {};
      return { currentPreviewUrl };
    }),
  setError: (message) => set({ status: "error", errorMessage: message }),
  setPreviewEnabled: (previewEnabled) => set({ previewEnabled }),
  setOutputs: (outputs) => set({ outputs }),
  addOutput: (path, name, kind) =>
    set((state) => {
      // Prevent duplicate outputs by path
      if (state.outputs.some((o) => o.path === path)) return {};
      return {
        outputs: [...state.outputs, { id: `${path}-${Date.now()}`, path, name, kind }],
      };
    }),
  setTunerStatus: (tunerStatus) => set({ tunerStatus }),
  reset: () =>
    set({
      status: "idle",
      progress: { completed: 0, total: 0, label: "" },
      currentPreviewUrl: "",
      errorMessage: "",
      outputs: [],
      tunerStatus: null,
    }),
}));
