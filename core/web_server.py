import http.server
import socketserver
import json
import os
import sys
import threading
import urllib.parse
import mimetypes
import shutil

# Global state mapping telemetry updates
STATE = {
    "status": "idle", # "idle", "running", "done", "error"
    "progress": {"completed": 0, "total": 0, "label": ""},
    "preview_url": "",
    "error_message": "",
    "outputs": [],
    "swarm_state": None # Stores tuner status dictionary
}

import traceback
import datetime

STATE_LOCK = threading.Lock()
ACTIVE_THREAD = None
CANCEL_EVENT = threading.Event()

def log_error_to_file(error_msg, exception=None):
    try:
        import core.config as cfg
        project_path = getattr(cfg, "PROJECT_PATH", "")
        if not project_path:
            project_path = os.getcwd()
            
        log_dir = os.path.join(project_path, "assets", "docs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "log.txt")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {error_msg}\n"
        if exception:
            log_entry += f"Traceback:\n"
            log_entry += "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        log_entry += "-" * 50 + "\n"
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as log_err:
        sys.stderr.write(f"Failed to write to log.txt: {log_err}\n")

def set_cors_headers(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Range")

def serve_file_with_range(handler, filepath):
    if not os.path.isfile(filepath):
        handler.send_response(404)
        handler.end_headers()
        return

    stat = os.stat(filepath)
    size = stat.st_size
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
        mime_type = "application/octet-stream"

    range_header = handler.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        try:
            parts = range_header.strip().split("=")[1].split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else size - 1
        except Exception:
            start = 0
            end = size - 1
        
        if start >= size or end >= size or start > end:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
        
        length = end - start + 1
        handler.send_response(206)
        handler.send_header("Content-Type", mime_type)
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.send_header("Content-Length", str(length))
        set_cors_headers(handler)
        handler.end_headers()
        
        with open(filepath, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                try:
                    handler.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break # Client closed connection abruptly, exit safely
                remaining -= len(chunk)
    else:
        handler.send_response(200)
        handler.send_header("Content-Type", mime_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        set_cors_headers(handler)
        handler.end_headers()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                try:
                    handler.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break # Client closed connection abruptly, exit safely

def serve_static_file(handler):
    path = handler.path.split("?")[0]
    if path == "/":
        path = "/index.html"
    
    dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui", "dist")
    filepath = os.path.abspath(os.path.join(dist_dir, path.lstrip("/")))
    
    if not filepath.startswith(os.path.abspath(dist_dir)):
        handler.send_response(403)
        handler.end_headers()
        return

    if not os.path.isfile(filepath):
        filepath = os.path.join(dist_dir, "index.html")

    if not os.path.isfile(filepath):
        handler.send_response(404)
        handler.end_headers()
        return

    stat = os.stat(filepath)
    size = stat.st_size
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
        mime_type = "application/octet-stream"

    handler.send_response(200)
    handler.send_header("Content-Type", mime_type)
    handler.send_header("Content-Length", str(size))
    set_cors_headers(handler)
    handler.end_headers()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)

class ApiRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging spam in console
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        set_cors_headers(self)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            set_cors_headers(self)
            self.end_headers()
            with STATE_LOCK:
                self.wfile.write(json.dumps(STATE).encode("utf-8"))
        elif path == "/api/outputs":
            query = urllib.parse.parse_qs(parsed_url.query)
            folder_paths = query.get("path")
            folder_path = folder_paths[0] if folder_paths else ""
            if not folder_path.strip():
                folder_path = os.path.join(os.getcwd(), "output")
                
            outputs = []
            if os.path.isdir(folder_path):
                try:
                    for filename in os.listdir(folder_path):
                        filepath = os.path.join(folder_path, filename)
                        if os.path.isfile(filepath):
                            ext = os.path.splitext(filename)[1].lower()
                            if ext in (".mp4", ".mkv", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                                kind = "video" if ext in (".mp4", ".mkv", ".avi", ".mov") else "image"
                                outputs.append({
                                    "id": f"{filepath}-{os.path.getmtime(filepath)}",
                                    "path": filepath,
                                    "name": filename,
                                    "kind": kind
                                })
                    outputs.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
                except Exception as e:
                    print(f"Error scanning folder: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            set_cors_headers(self)
            self.end_headers()
            self.wfile.write(json.dumps(outputs).encode("utf-8"))
        elif path == "/api/file":
            query = urllib.parse.parse_qs(parsed_url.query)
            filepaths = query.get("path")
            if filepaths:
                serve_file_with_range(self, filepaths[0])
            else:
                self.send_response(400)
                self.end_headers()
        else:
            serve_static_file(self)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/start":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                config_data = json.loads(body)
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                return

            # Start pipeline job in background thread
            success = start_pipeline_thread(config_data)
            self.send_response(200 if success else 409)
            self.send_header("Content-Type", "application/json")
            set_cors_headers(self)
            self.end_headers()
            self.wfile.write(json.dumps({"success": success}).encode("utf-8"))

        elif path == "/api/cancel":
            cancel_pipeline_job()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            set_cors_headers(self)
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/file":
            query = urllib.parse.parse_qs(parsed_url.query)
            filepaths = query.get("path")
            if filepaths:
                target_path = filepaths[0]
                success = False
                try:
                    if os.path.exists(target_path):
                        if os.path.isdir(target_path):
                            shutil.rmtree(target_path)
                        else:
                            os.remove(target_path)
                        success = True
                except Exception as e:
                    print(f"Error deleting file {target_path}: {e}")

                # Update state output list
                with STATE_LOCK:
                    STATE["outputs"] = [item for item in STATE["outputs"] if item["path"] != target_path]

                self.send_response(200 if success else 500)
                self.send_header("Content-Type", "application/json")
                set_cors_headers(self)
                self.end_headers()
                self.wfile.write(json.dumps({"success": success}).encode("utf-8"))
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run_pipeline_worker(config_data):
    from core.library_api import run_pipeline
    from core.types import RuntimeContext, build_run_config_from_cfg
    import core.config as cfg

    global CANCEL_EVENT
    CANCEL_EVENT.clear()

    try:
        # Project settings
        project_path = config_data.get("PROJECT_PATH", "").strip()
        if project_path:
            cfg.PROJECT_PATH = cfg.normalize_project_root(project_path)
        else:
            cfg.PROJECT_PATH = cfg._resolve_default_project_root(cfg.BASE_DIR)

        layout = cfg.ensure_project_layout(cfg.PROJECT_PATH, cfg.SOURCE_ASSETS_DIR)
        cfg.ASSETS_HOME = layout.assets_dir
        cfg.MODELS_DIR = layout.models_dir
        cfg.FACES_DIR = layout.faces_dir
        cfg.TEMP_AUDIO_DIR = layout.temp_audio_dir
        cfg.TENSORRT_HOME = layout.tensorrt_home
        cfg.TRT_CACHE_DIR = layout.trt_cache_dir
        
        cfg.TRT_CACHE_DETECT_DIR = os.path.join(cfg.TRT_CACHE_DIR, "trt_cache_detect")
        cfg.TRT_CACHE_SWAP_DIR = os.path.join(cfg.TRT_CACHE_DIR, "trt_cache_swap")
        cfg.TRT_CACHE_RESTORE_DIR = os.path.join(cfg.TRT_CACHE_DIR, "trt_cache_restore")
        cfg.TRT_CACHE_PARSER_DIR = os.path.join(cfg.TRT_CACHE_DIR, "trt_cache_parser")
        cfg.INSIGHTFACE_ROOT = os.path.join(cfg.MODELS_DIR, "insightface_models")

        # Identifiers & Paths
        raw_face = config_data.get("INPUT_FACE", "").strip()
        cfg.FACE_NAME, cfg.SOURCE_FACE_PATH, cfg.FACE_SOURCE_IS_IMAGE = cfg._resolve_input_face(raw_face, cfg.FACES_DIR)

        raw_target = config_data.get("INPUT_TARGET", "").strip()
        cfg.INPUT_PATH, cfg.INPUT_SINGLE_FILE, _fmt_override = cfg._resolve_input_target(raw_target)
        if _fmt_override is not None:
            cfg.FORMAT_IS_IMAGE = _fmt_override
        else:
            cfg.FORMAT_IS_IMAGE = (config_data.get("FORMAT", "video") == "image")

        out_dir = config_data.get("OUTPUT_PATH", "").strip()
        if out_dir:
            cfg.OUTPUT_DIR = os.path.abspath(out_dir)
        else:
            if cfg.FORMAT_IS_IMAGE:
                cfg.OUTPUT_DIR = os.path.join(layout.output_dir, "image", cfg.FACE_NAME)
            else:
                cfg.OUTPUT_DIR = os.path.join(layout.output_dir, "video", cfg.FACE_NAME)

        # Ensure output directory exists
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Switches
        cfg.ENABLE_SWAPPER = config_data.get("USE_SWAPER", True)
        cfg.SWAPPER_BLEND = config_data.get("SWAPER_WEIGH", 0.70)
        cfg.ENABLE_RESTORE = config_data.get("USE_RESTORE", True)
        cfg.RESTORE_CHOICE = config_data.get("RESTORE_CHOICE", "1")
        cfg.RESTORE_WEIGHT = config_data.get("RESTORE_WEIGH", 0.70)
        cfg.RESTORE_BLEND = config_data.get("RESTORE_BLEND", 0.70)
        cfg.ENABLE_PARSER = config_data.get("USE_PARSER", True)
        cfg.PRESERVE_SWAP_EYES = config_data.get("PRESERVE_SWAP_EYES", True)
        cfg.PARSER_MASK_BLUR = config_data.get("PARSER_MASK_BLUR", 21)

        # Models
        cfg.SWAPPER_MODEL = os.path.join(cfg.MODELS_DIR, "inswapper_128.onnx")
        cfg.PARSER_TYPE = "segformer"
        cfg.PARSER_MODEL = os.path.join(cfg.MODELS_DIR, "Segformer_CelebAMask-HQ.onnx")

        if cfg.RESTORE_CHOICE == "2":
            cfg.RESTORE_MODEL_NAME = "GPEN-BFR-512.onnx"
            cfg.RESTORE_SIZE = 512
        elif cfg.RESTORE_CHOICE == "3":
            cfg.RESTORE_MODEL_NAME = "GPEN-BFR-1024.onnx"
            cfg.RESTORE_SIZE = 1024
        elif cfg.RESTORE_CHOICE == "4":
            cfg.RESTORE_MODEL_NAME = "codeformer.onnx"
            cfg.RESTORE_SIZE = 512
        else:
            cfg.RESTORE_MODEL_NAME = "GFPGANv1.4.onnx"
            cfg.RESTORE_SIZE = 512
        cfg.RESTORE_MODEL_PATH = os.path.join(cfg.MODELS_DIR, cfg.RESTORE_MODEL_NAME)

        cfg.FFMPEG_CMD = os.path.join(cfg.MODELS_DIR, cfg._platform_ffmpeg_name())

        # Performance
        cfg.PROVIDER_ALL = config_data.get("PROVIDER_ALL", "trt")
        cfg.PROVIDER_POLICY["detect"] = config_data.get("PROVIDER_DETECT", "auto")
        cfg.PROVIDER_POLICY["swap"] = config_data.get("PROVIDER_SWAPER", "auto")
        cfg.PROVIDER_POLICY["restore"] = config_data.get("PROVIDER_RESTORE", "auto")
        cfg.PROVIDER_POLICY["parse"] = config_data.get("PROVIDER_PARSER", "auto")

        for _stage, _provider in cfg.PROVIDER_POLICY.items():
            if _provider == "auto":
                cfg.PROVIDER_POLICY[_stage] = cfg.PROVIDER_ALL
            else:
                cfg.PROVIDER_POLICY[_stage] = cfg._parse_provider(_provider, f"PROVIDER_{_stage.upper()}")

        cfg.WORKERS_PER_STAGE = config_data.get("WORKERS_PER_STAGE", 8)
        cfg.WORKER_QUEUE_SIZE = config_data.get("WORKER_QUEUE_SIZE", 64)
        cfg.OUT_QUEUE_SIZE = config_data.get("OUT_QUEUE_SIZE", 128)
        cfg.TUNER_MODE = config_data.get("TUNER_MODE", "auto")
        cfg.GPU_TARGET_UTIL = config_data.get("GPU_TARGET_UTIL", 95)
        cfg.HIGH_WATERMARK = config_data.get("HIGH_WATERMARK", 12)
        cfg.LOW_WATERMARK = config_data.get("LOW_WATERMARK", 4)
        cfg.SWITCH_COOLDOWN_S = config_data.get("SWITCH_COOLDOWN_S", 0.35)

        # Run behavior
        cfg.MAX_FRAMES = config_data.get("MAX_FRAMES", 0)
        cfg.MAX_RETRIES = config_data.get("MAX_RETRIES", 2)
        cfg.SKIP_EXISTING = config_data.get("SKIP_EXISTING", True)
        cfg.OUTPUT_SUFFIX = config_data.get("OUTPUT_SUFFIX", "")
        cfg.FILE_SORTING = config_data.get("FILE_SORTING", "date_modified_newest")

        cfg.PRELOAD_MODELS = config_data.get("PRELOAD_MODELS", False)
        cfg.DRY_RUN = config_data.get("DRY_RUN", False)
        cfg.PRINT_EFFECTIVE_CONFIG = config_data.get("PRINT_EFFECTIVE_CONFIG", False)
        cfg.LOG_LEVEL = config_data.get("LOG_LEVEL", "WARNING").upper()

        # Path validations (mirroring the checks in config.py)
        if not cfg.FACE_NAME and cfg.ENABLE_SWAPPER:
            raise ValueError("Source face is required when swapper is enabled.")
        if not cfg.INPUT_PATH:
            raise ValueError("Target input path is required.")

        if cfg.ENABLE_SWAPPER and not os.path.isfile(cfg.SOURCE_FACE_PATH):
            if cfg.FACE_SOURCE_IS_IMAGE:
                raise FileNotFoundError(f"Source face image not found: {cfg.SOURCE_FACE_PATH}")
            else:
                raise FileNotFoundError(f"Source face model (.safetensors) not found: {cfg.SOURCE_FACE_PATH}")

        if cfg.INPUT_SINGLE_FILE:
            single_file_full = os.path.join(cfg.INPUT_PATH, cfg.INPUT_SINGLE_FILE)
            if not os.path.isfile(single_file_full):
                raise FileNotFoundError(f"Input file does not exist: {single_file_full}")
        else:
            if not os.path.isdir(cfg.INPUT_PATH):
                raise FileNotFoundError(f"Input directory does not exist: {cfg.INPUT_PATH}")

        if cfg.ENABLE_SWAPPER and not os.path.isfile(cfg.SWAPPER_MODEL):
            raise FileNotFoundError(f"Swapper model not found: {cfg.SWAPPER_MODEL}")

        if cfg.ENABLE_RESTORE and not os.path.isfile(cfg.RESTORE_MODEL_PATH):
            raise FileNotFoundError(f"Restore model not found: {cfg.RESTORE_MODEL_PATH}")

        if cfg.ENABLE_PARSER and not os.path.isfile(cfg.PARSER_MODEL):
            raise FileNotFoundError(f"Parser model not found: {cfg.PARSER_MODEL}")

        if not os.path.isfile(cfg.FFMPEG_CMD):
            raise FileNotFoundError(f"ffmpeg executable not found: {cfg.FFMPEG_CMD}")

        run_config = build_run_config_from_cfg(cfg)
        runtime_ctx = RuntimeContext(config=run_config)
        
        # Telemetry hooks
        def custom_on_event(name, payload):
            with STATE_LOCK:
                if name == "pipeline_start":
                    STATE["status"] = "running"
                elif name == "pipeline_complete":
                    STATE["status"] = "done"
                elif name == "preview":
                    STATE["preview_url"] = payload.get("data_url", "")
                elif name == "tuner_status":
                    STATE["swarm_state"] = payload
                elif name == "item_completed":
                    item_id = payload.get("item_id", "")
                    kind = payload.get("kind", "image")
                    out_dir = run_config.output_dir or os.path.join(os.getcwd(), "output")
                    out_path = os.path.join(out_dir, item_id)
                    STATE["outputs"].append({
                        "id": f"{out_path}-{len(STATE['outputs'])}",
                        "path": out_path,
                        "name": item_id,
                        "kind": kind
                    })

        def custom_on_progress(label, completed, total):
            with STATE_LOCK:
                STATE["progress"] = {
                    "completed": int(completed),
                    "total": int(total),
                    "label": str(label)
                }

        runtime_ctx.hooks.on_event = custom_on_event
        runtime_ctx.hooks.on_progress = custom_on_progress

        with STATE_LOCK:
            STATE["status"] = "running"
            STATE["error_message"] = ""
            STATE["preview_url"] = ""

        run_pipeline(
            runtime_ctx=runtime_ctx,
            external_stop_event=CANCEL_EVENT
        )
    except Exception as e:
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["error_message"] = str(e)
        log_error_to_file(f"Exception during pipeline execution: {e}", e)
    finally:
        with STATE_LOCK:
            if STATE["status"] == "running":
                STATE["status"] = "done"

def start_pipeline_thread(config_data):
    global ACTIVE_THREAD
    with STATE_LOCK:
        if STATE["status"] == "running":
            return False
    
    ACTIVE_THREAD = threading.Thread(target=run_pipeline_worker, args=(config_data,), daemon=True)
    ACTIVE_THREAD.start()
    return True

def cancel_pipeline_job():
    global CANCEL_EVENT
    CANCEL_EVENT.set()
    with STATE_LOCK:
        STATE["status"] = "idle"
        STATE["error_message"] = "Job canceled by user."

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

def start_api_server(port=8234, bind_address="0.0.0.0"):
    server_address = (bind_address, port)
    httpd = ThreadingHTTPServer(server_address, ApiRequestHandler)
    print(f"Python API/Static Server running on http://{bind_address}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
