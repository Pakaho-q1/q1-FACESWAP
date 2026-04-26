from __future__ import annotations

import json

from nicegui import ui


def build_preview_container(preview_dom_id: str) -> None:
    ui.html(
        f"""
        <div id=\"{preview_dom_id}\" style=\"position:relative;width:100%;height:300px;background:#0f172a;border-radius:0;overflow:hidden;\">
          <img id=\"{preview_dom_id}_a\" src=\"\" style=\"position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:1;transition:none;\" />
          <img id=\"{preview_dom_id}_b\" src=\"\" style=\"position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;transition:none;\" />
        </div>
        """
    ).classes("w-full")


def register_preview_bridge_assets() -> None:
    ui.add_body_html(
        """
        <script>
        (function() {
          window.__q1PreviewState = window.__q1PreviewState || {};
          function init(id) {
            if (!window.__q1PreviewState[id]) {
              window.__q1PreviewState[id] = { active: "a", seq: 0 };
            }
            return window.__q1PreviewState[id];
          }
          window.q1PreviewReset = function(id) {
            const st = init(id);
            const a = document.getElementById(id + "_a");
            const b = document.getElementById(id + "_b");
            if (!a || !b) return;
            a.src = "";
            b.src = "";
            a.style.opacity = "1";
            b.style.opacity = "0";
            st.active = "a";
            st.seq += 1;
          };
          window.q1PreviewSwap = function(id, url) {
            const st = init(id);
            const a = document.getElementById(id + "_a");
            const b = document.getElementById(id + "_b");
            if (!a || !b || !url) return;
            const next = st.active === "a" ? "b" : "a";
            const currentLayer = st.active === "a" ? a : b;
            const targetLayer = next === "a" ? a : b;
            const ticket = ++st.seq;
            const img = new Image();
            img.decoding = "async";
            img.onload = function() {
              const currentState = window.__q1PreviewState && window.__q1PreviewState[id];
              if (!currentState || currentState.seq !== ticket) return;
              targetLayer.src = url;
              targetLayer.style.opacity = "1";
              currentLayer.style.opacity = "0";
              currentState.active = next;
            };
            img.src = url;
          };
        })();
        </script>
        """
    )


def preview_reset_script(preview_dom_id: str) -> str:
    return f"window.q1PreviewReset && window.q1PreviewReset('{preview_dom_id}');"


def preview_swap_script(preview_dom_id: str, data_url: str) -> str:
    payload = json.dumps(str(data_url))
    return f"window.q1PreviewSwap && window.q1PreviewSwap('{preview_dom_id}', {payload});"
