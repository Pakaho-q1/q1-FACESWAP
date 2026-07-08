import { listen } from "@tauri-apps/api/event";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export async function listenToPipelineEvents<T>(
  handler: (payload: T) => void
): Promise<UnlistenFn> {
  // Tauri event listener. The Rust backend emits "pipeline-event"
  return await listen<T>("pipeline-event", (event) => {
    handler(event.payload);
  });
}

export async function startPipeline(config: Record<string, any>): Promise<void> {
  await invoke("start_pipeline_job", { config });
}

export async function openFolder(path: string): Promise<void> {
  await invoke("open_folder", { path });
}

export async function selectFile(filters: string[]): Promise<string | null> {
  return await invoke<string | null>("select_file", { filters });
}

export async function selectDirectory(): Promise<string | null> {
  return await invoke<string | null>("select_directory");
}

export async function cancelPipeline(): Promise<void> {
  await invoke("cancel_pipeline_job");
}
