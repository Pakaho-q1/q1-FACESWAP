export type PipelineEvent =
  | { type: "progress"; completed: number; total: number }
  | { type: "preview"; data_url: string }
  | { type: "status"; state: "idle" | "running" | "done" | "error"; message?: string }
  | { type: "output"; kind: "image" | "video"; path: string };
