import { useEffect } from "react";
import { usePipelineStore } from "../store/pipeline-store";

export function usePipelineEvents() {
  const setStatus = usePipelineStore((state) => state.setStatus);
  const setProgress = usePipelineStore((state) => state.setProgress);
  const setPreviewUrl = usePipelineStore((state) => state.setPreviewUrl);
  const setError = usePipelineStore((state) => state.setError);
  const setTunerStatus = usePipelineStore((state) => state.setTunerStatus);

  useEffect(() => {
    const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
    const apiBase = `http://${host}:8234`;
    let timerId: any = null;
    let isActive = true;

    const pollStatus = async () => {
      if (!isActive) return;

      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok && isActive) {
          const data = await res.json();
          setStatus(data.status);
          setProgress(
            data.progress.completed,
            data.progress.total,
            data.progress.label
          );
          if (data.preview_url) {
            setPreviewUrl(data.preview_url);
          }
          if (data.status === "error" && data.error_message) {
            setError(data.error_message);
          }
          setTunerStatus(data.swarm_state);
        }
      } catch (err) {
        // Suppress console spam if server is booting up or offline
      }

      // Schedule next poll: 100ms when running, 1000ms when idle
      const currentStatus = usePipelineStore.getState().status;
      const nextInterval = currentStatus === "running" ? 100 : 1000;
      
      if (isActive) {
        timerId = setTimeout(pollStatus, nextInterval);
      }
    };

    pollStatus();

    return () => {
      isActive = false;
      if (timerId) {
        clearTimeout(timerId);
      }
    };
  }, [setStatus, setProgress, setPreviewUrl, setError, setTunerStatus]);
}
