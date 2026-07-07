use std::process::{Command, Stdio};
use std::io::{BufReader, BufRead, Read};
use tauri::{AppHandle, Emitter};

static CURRENT_CHILD: std::sync::Mutex<Option<std::process::Child>> = std::sync::Mutex::new(None);

fn find_faceswap_py() -> Option<std::path::PathBuf> {
  let mut current = std::env::current_dir().ok()?;
  loop {
    let candidate = current.join("faceswap.py");
    if candidate.is_file() {
      return Some(candidate);
    }
    if !current.pop() {
      break;
    }
  }
  None
}

#[tauri::command]
fn select_file(filters: Vec<String>) -> Result<Option<String>, String> {
  let mut dialog = rfd::FileDialog::new();
  if !filters.is_empty() {
    let extensions: Vec<&str> = filters.iter().map(|s| s.as_str()).collect();
    dialog = dialog.add_filter("Files", &extensions);
  }
  let path = dialog.pick_file();
  Ok(path.map(|p| p.to_string_lossy().to_string()))
}

#[tauri::command]
fn select_directory() -> Result<Option<String>, String> {
  let path = rfd::FileDialog::new().pick_folder();
  Ok(path.map(|p| p.to_string_lossy().to_string()))
}

#[tauri::command]
fn cancel_pipeline_job() -> Result<(), String> {
  if let Ok(mut lock) = CURRENT_CHILD.lock() {
    if let Some(mut child) = lock.take() {
      let _ = child.kill();
    }
  }
  Ok(())
}

#[tauri::command]
fn open_folder(path: String) -> Result<(), String> {
  let path_to_open = if path.trim().is_empty() {
    let mut current = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
      if current.join("faceswap.py").is_file() {
        break;
      }
      if !current.pop() {
        return Err("Could not locate project root".to_string());
      }
    }
    current.join("output")
  } else {
    std::path::PathBuf::from(path)
  };

  if !path_to_open.exists() {
    return Err(format!("Path does not exist: {}", path_to_open.display()));
  }

  #[cfg(target_os = "windows")]
  {
    Command::new("explorer")
      .arg(&path_to_open)
      .spawn()
      .map_err(|e| e.to_string())?;
  }
  #[cfg(target_os = "macos")]
  {
    Command::new("open")
      .arg(&path_to_open)
      .spawn()
      .map_err(|e| e.to_string())?;
  }
  #[cfg(target_os = "linux")]
  {
    Command::new("xdg-open")
      .arg(&path_to_open)
      .spawn()
      .map_err(|e| e.to_string())?;
  }
  Ok(())
}

#[tauri::command]
fn start_pipeline_job(app: AppHandle, config: std::collections::HashMap<String, serde_json::Value>) -> Result<(), String> {
  // Terminate any existing child process first
  let _ = cancel_pipeline_job();

  let faceswap_path = find_faceswap_py().ok_or_else(|| "Could not locate faceswap.py".to_string())?;
  
  let mut args = vec![
    faceswap_path.to_string_lossy().to_string(),
    "run".to_string(),
  ];

  if let Some(val) = config.get("INPUT_FACE").and_then(|v| v.as_str()) {
    args.push("--input-face".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("INPUT_TARGET").and_then(|v| v.as_str()) {
    args.push("--input-target".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("OUTPUT_PATH").and_then(|v| v.as_str()) {
    if !val.trim().is_empty() {
      args.push("-o".to_string());
      args.push(val.to_string());
    }
  }
  if let Some(val) = config.get("USE_SWAPER").and_then(|v| v.as_bool()) {
    args.push("--use-swaper".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("SWAPER_WEIGH").and_then(|v| v.as_f64()) {
    args.push("--swaper-weigh".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("USE_RESTORE").and_then(|v| v.as_bool()) {
    args.push("--use-restore".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("RESTORE_CHOICE").and_then(|v| v.as_str()) {
    args.push("--restore-choice".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("RESTORE_WEIGH").and_then(|v| v.as_f64()) {
    args.push("--restore-weigh".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("RESTORE_BLEND").and_then(|v| v.as_f64()) {
    args.push("--restore-blend".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("USE_PARSER").and_then(|v| v.as_bool()) {
    args.push("--use-parser".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PRESERVE_SWAP_EYES").and_then(|v| v.as_bool()) {
    args.push("--preserve-swap-eyes".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PARSER_MASK_BLUR").and_then(|v| v.as_i64()) {
    args.push("--parser-mask-blur".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROVIDER_ALL").and_then(|v| v.as_str()) {
    args.push("--provider-all".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROVIDER_SWAPER").and_then(|v| v.as_str()) {
    args.push("--provider-swaper".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROVIDER_RESTORE").and_then(|v| v.as_str()) {
    args.push("--provider-restore".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROVIDER_PARSER").and_then(|v| v.as_str()) {
    args.push("--provider-parser".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROVIDER_DETECT").and_then(|v| v.as_str()) {
    args.push("--provider-detect".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("WORKERS_PER_STAGE").and_then(|v| v.as_i64()) {
    args.push("--workers-per-stage".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("WORKER_QUEUE_SIZE").and_then(|v| v.as_i64()) {
    args.push("--worker-queue-size".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("OUT_QUEUE_SIZE").and_then(|v| v.as_i64()) {
    args.push("--out-queue-size".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("TUNER_MODE").and_then(|v| v.as_str()) {
    args.push("--tuner-mode".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("GPU_TARGET_UTIL").and_then(|v| v.as_i64()) {
    args.push("--gpu-target-util".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("HIGH_WATERMARK").and_then(|v| v.as_i64()) {
    args.push("--high-watermark".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("LOW_WATERMARK").and_then(|v| v.as_i64()) {
    args.push("--low-watermark".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("SWITCH_COOLDOWN_S").and_then(|v| v.as_f64()) {
    args.push("--switch-cooldown-s".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("MAX_FRAMES").and_then(|v| v.as_i64()) {
    args.push("--max-frames".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("MAX_RETRIES").and_then(|v| v.as_i64()) {
    args.push("--max-retries".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("SKIP_EXISTING").and_then(|v| v.as_bool()) {
    args.push("--skip-existing".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("OUTPUT_SUFFIX").and_then(|v| v.as_str()) {
    args.push("--output-suffix".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("FILE_SORTING").and_then(|v| v.as_str()) {
    args.push("--file-sorting".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PROJECT_PATH").and_then(|v| v.as_str()) {
    args.push("--project-path".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PRELOAD_MODELS").and_then(|v| v.as_bool()) {
    args.push("--preload-models".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("DRY_RUN").and_then(|v| v.as_bool()) {
    args.push("--dry-run".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("PRINT_EFFECTIVE_CONFIG").and_then(|v| v.as_bool()) {
    args.push("--print-effective-config".to_string());
    args.push(val.to_string());
  }
  if let Some(val) = config.get("LOG_LEVEL").and_then(|v| v.as_str()) {
    args.push("--log-level".to_string());
    args.push(val.to_string());
  }

  std::thread::spawn(move || {
    let mut child = match Command::new("python")
      .args(&args)
      .env("Q1_GUI_EVENTS", "1")
      .stdout(Stdio::piped())
      .stderr(Stdio::piped())
      .spawn()
    {
      Ok(c) => c,
      Err(e) => {
        let err_msg = format!("Failed to spawn python process: {}", e);
        let _ = app.emit("pipeline-event", serde_json::json!({
          "type": "status",
          "state": "error",
          "message": err_msg
        }));
        return;
      }
    };

    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    // Store child handle globally for cancelation
    if let Ok(mut lock) = CURRENT_CHILD.lock() {
      *lock = Some(child);
    }

    let reader = BufReader::new(stdout);
    for line_res in reader.lines() {
      let line = match line_res {
        Ok(l) => l,
        Err(_) => break,
      };

      if line.starts_with("__Q1_GUI__") {
        let json_part = &line["__Q1_GUI__".len()..];
        if let Ok(parsed_json) = serde_json::from_str::<serde_json::Value>(json_part) {
          let _ = app.emit("pipeline-event", parsed_json);
        }
      }
    }

    // Wait on child inside lock
    let wait_res = {
      let mut lock = CURRENT_CHILD.lock().unwrap();
      if let Some(ref mut child) = *lock {
        Some(child.wait())
      } else {
        None
      }
    };

    // Clear child handle globally
    if let Ok(mut lock) = CURRENT_CHILD.lock() {
      *lock = None;
    }

    match wait_res {
      Some(Ok(status)) => {
        if !status.success() {
          let mut err_string = String::new();
          let _ = stderr.read_to_string(&mut err_string);
          let err_msg = if err_string.trim().is_empty() {
            "Python pipeline process exited with non-zero status".to_string()
          } else {
            err_string
          };

          let _ = app.emit("pipeline-event", serde_json::json!({
            "type": "status",
            "state": "error",
            "message": err_msg
          }));
        } else {
          let _ = app.emit("pipeline-event", serde_json::json!({
            "type": "status",
            "state": "done"
          }));
        }
      }
      Some(Err(e)) => {
        let _ = app.emit("pipeline-event", serde_json::json!({
          "type": "status",
          "state": "error",
          "message": format!("Error waiting for process: {}", e)
        }));
      }
      None => {
        // Child was taken/killed by cancel command
        let _ = app.emit("pipeline-event", serde_json::json!({
          "type": "status",
          "state": "idle",
          "message": "Job canceled by user."
        }));
      }
    }
  });

  Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      start_pipeline_job,
      open_folder,
      select_file,
      select_directory,
      cancel_pipeline_job
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
