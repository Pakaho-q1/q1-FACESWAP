import os
import sys
import subprocess
import time


def print_step(msg):
    print(f"\n{'-'*50}\n🚀 {msg}\n{'-'*50}")


def print_ok(msg):
    print(f"✅ [PASS] {msg}")


def print_err(msg):
    print(f"❌ [FAIL] {msg}")


def run_command(command):
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print_step("STEP 1: Check the Python version")
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 10:
        print_ok(f"Python {py_ver.major}.{py_ver.minor} Supports work")
    else:
        print_err(
            f"Python {py_ver.major}.{py_ver.minor} It may be too old. We recommend using 3.10 or higher."
        )

    print_step("STEP 2: Install the library from requirements.txt")
    if os.path.exists("requirements.txt"):
        print("⏳ Installing package (This may take a while)...")
        if run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        ):
            print_ok("Basic library installed successfully")
        else:
            print_err(
                "Unable to install library. Please check your internet connection."
            )
            return
    else:
        print_err("File requirements.txt not found")
        return

    print_step("STEP 3: Check and fix ONNX Runtime conflicts")
    print("⏳ Checking ONNX Runtime CPU & GPU...")

    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])

    if run_command([sys.executable, "-m", "pip", "show", "onnxruntime-gpu"]):
        print_ok(
            "onnxruntime-gpu Installed correctly and without the CPU bothering me."
        )
    else:
        print_err("onnxruntime-gpu not found Please reinstall")

    print_step("STEP 4: NVIDIA GPU and TensorRT visibility test")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in providers:
            print_ok("ONNX Runtime sees CUDA (NVIDIA GPU)")
        else:
            print_err(
                "ONNX Runtime cannot see CUDA! Please check the graphics card driver."
            )

        if "TensorrtExecutionProvider" in providers:
            print_ok("ONNX Runtime supports TensorRT")
        else:
            print_err("TensorRT Provider not found")

    except Exception as e:
        print_err(f"Error loading ONNX Runtime:{e}")

    print_step("STEP 5: Check the folder structure (Folder Structure Check)")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    critical_paths = {
        "models/folder": os.path.isdir(MODELS_DIR),
        "Inswapper Model": os.path.isfile(
            os.path.join(MODELS_DIR, "inswapper_128.onnx")
        ),
        "Face Restore Model": os.path.isfile(
            os.path.join(MODELS_DIR, "GFPGANv1.4.onnx")
        ),
        "Face Parser Model": os.path.isfile(
            os.path.join(MODELS_DIR, "faceparser_resnet34.onnx")
        ),
        "FFmpeg Executable": os.path.isfile(os.path.join(MODELS_DIR, "ffmpeg.exe")),
        "TensorRT Binaries": os.path.isdir(
            os.path.join(MODELS_DIR, "TensorRT-10.15.1.29", "bin")
        ),
        "InsightFace Models": os.path.isdir(
            os.path.join(MODELS_DIR, "insightface_models", "models", "buffalo_l")
        ),
    }

    all_paths_ok = True
    for name, exists in critical_paths.items():
        if exists:
            print_ok(f"meet {name}")
        else:
            print_err(
                f"Not found {name} -> Please make sure you put the file in the right place."
            )
            all_paths_ok = False

    print_step("🏁 Summary of inspection results (Final Report)")
    if all_paths_ok and ("CUDAExecutionProvider" in locals().get("providers", [])):
        print(
            "🎉 100% complete environment! The system is ready to run face_swap_unified.py."
        )
    else:
        print(
            "⚠️ Found some bugs listed in red above. Please fix it before starting to run Pipeline."
        )


if __name__ == "__main__":
    main()
