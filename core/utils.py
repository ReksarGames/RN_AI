import glob
import os
import platform
import sys
import threading
import time
import traceback
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image
from pynput.keyboard import Key, KeyCode

# Constants
BAR_HEIGHT = 2
SHADOW_OFFSET = 2
VERSION = "v3.1.5"
UPDATE_TIME = "2026-21-06"

# Global variable for TensorRT availability
TENSORRT_AVAILABLE = False

RUN_LOG_FILE = "run_log.txt"

def check_tensorrt_availability():
    """Safely detect if TensorRT environment is available"""
    try:
        # Use torch to initialize CUDA (no C++ compiler needed, unlike pycuda)
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available via torch")
            return False

        import tensorrt as trt

        gpu_name = torch.cuda.get_device_name(0)
        print(f"TensorRT available (GPU: {gpu_name})")
        return True
    except ImportError as e:
        print(f"TensorRT/Torch not installed: {e}")
        return False
    except Exception as e:
        print(f"CUDA/TensorRT not available: {e}")
        return False


# Initialize TensorRT availability
TENSORRT_AVAILABLE = check_tensorrt_availability()


def _append_run_log(lines: Iterable[str]) -> None:
    try:
        with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        pass


def reset_run_log(enabled: bool) -> None:
    if not enabled:
        return
    try:
        with open(RUN_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


def log_run_event(
    title: str, details: Optional[Dict[str, object]] = None, enabled: bool = True
) -> None:
    if not enabled:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{ts}] {title}"]
    if details:
        for key, value in details.items():
            lines.append(f"  {key}: {value}")
    _append_run_log(lines)


def collect_system_info() -> Dict[str, object]:
    info: Dict[str, object] = {}
    info["version"] = VERSION
    info["update_time"] = UPDATE_TIME
    info["platform"] = platform.platform()
    info["platform_release"] = platform.release()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor()
    info["python"] = sys.version.replace("\n", " ")
    info["python_exe"] = sys.executable
    info["cwd"] = os.getcwd()
    info["args"] = " ".join(sys.argv)
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = None
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
    except Exception:
        info["ram_total_gb"] = None
    try:
        import onnxruntime as rt

        info["onnxruntime_providers"] = rt.get_available_providers()
    except Exception:
        info["onnxruntime_providers"] = None
    try:
        path = os.environ.get("PATH", "")
        info["cuda_in_path"] = any(
            ("CUDA" in p.upper() and ("12" in p or "11" in p)) for p in path.split(os.pathsep)
        )
    except Exception:
        info["cuda_in_path"] = None
    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device"] = torch.cuda.get_device_name(0)
            try:
                info["cuda_runtime_version"] = torch.version.cuda
            except Exception:
                info["cuda_runtime_version"] = None
        else:
            info["cuda_available"] = False
            info["cuda_device"] = None
    except Exception:
        info["cuda_available"] = None
        info["cuda_device"] = None
        info["torch_version"] = None
        info["cuda_runtime_version"] = None
    try:
        import tensorrt  # noqa: F401
        info["tensorrt_installed"] = True
    except Exception:
        info["tensorrt_installed"] = False
    try:
        providers = info.get("onnxruntime_providers") or []
        info["onnx_gpu_available"] = (
            "CUDAExecutionProvider" in providers
            or "TensorrtExecutionProvider" in providers
        )
        info["onnx_dml_available"] = "DmlExecutionProvider" in providers
    except Exception:
        info["onnx_gpu_available"] = None
        info["onnx_dml_available"] = None
    info["status_cuda"] = (
        "OK" if info.get("cuda_in_path") else "Not detected (may need restart)"
    )
    info["status_pytorch_cuda"] = (
        "OK" if info.get("cuda_available") else "Not working"
    )
    info["status_tensorrt"] = (
        "OK" if info.get("tensorrt_installed") else "Not installed"
    )
    info["status_onnx_gpu"] = (
        "OK" if info.get("onnx_gpu_available") else "Not detected"
    )
    info["status_onnx_dml"] = (
        "OK" if info.get("onnx_dml_available") else "Not detected"
    )
    return info


def log_startup_info(
    enabled: bool = True, reset: bool = False, extra: Optional[Dict[str, object]] = None
) -> None:
    if reset:
        reset_run_log(enabled)
    details = collect_system_info()
    if extra:
        details.update(extra)
    log_run_event("Program start", details, enabled=enabled)


def log_model_info(
    model_path: str,
    group: str,
    is_trt: bool,
    yolo_format: str,
    yolo_version: str,
    input_shape: object,
    output_shape: object,
    providers: object,
    engine_type: str,
    extra: Optional[Dict[str, object]] = None,
    enabled: bool = True,
) -> None:
    details = {
        "group": group,
        "model_path": model_path,
        "is_trt": is_trt,
        "engine": engine_type,
        "yolo_format": yolo_format,
        "yolo_version": yolo_version,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "providers": providers,
    }
    if extra:
        details.update(extra)
    log_run_event("Model loaded", details, enabled=enabled)


def create_gradient_image(width, height):
    gradient = np.zeros((height, width, 4), dtype=np.uint8)
    colors = [(55, 177, 218), (204, 91, 184), (204, 227, 53)]
    for x in range(width):
        t = x / width
        r = int(colors[0][0] * (1 - t) + colors[2][0] * t)
        g = int(colors[0][1] * (1 - t) + colors[2][1] * t)
        b = int(colors[0][2] * (1 - t) + colors[2][2] * t)
        gradient[:, x] = (r, g, b, 255)
    img = Image.fromarray(gradient, "RGBA")
    img.save("skeet_gradient.png")
    return "skeet_gradient.png"


# Key alias mapping for pynput keys
_KEY_ALIAS = {
    Key.space: "space",
    Key.enter: "enter",
    Key.tab: "tab",
    Key.backspace: "backspace",
    Key.esc: "esc",
    Key.shift: "shift",
    Key.shift_l: "shift",
    Key.shift_r: "shift",
    Key.ctrl: "ctrl",
    Key.ctrl_l: "ctrl",
    Key.ctrl_r: "ctrl",
    Key.alt: "alt",
    Key.alt_l: "alt",
    Key.alt_r: "alt",
    Key.caps_lock: "caps_lock",
    Key.cmd: "cmd",
    Key.cmd_l: "cmd",
    Key.cmd_r: "cmd",
    Key.up: "up",
    Key.down: "down",
    Key.left: "left",
    Key.right: "right",
    Key.delete: "delete",
    Key.home: "home",
    Key.end: "end",
    Key.page_up: "page_up",
    Key.page_down: "page_down",
    Key.insert: "insert",
}


def key2str(key) -> str:
    """Convert pynput key object to unified string representation"""
    # Letters/numbers and other printable keys
    if isinstance(key, KeyCode):
        if key.char:  # Normal character
            return key.char
        # No char, fallback to virtual key code
        if getattr(key, "vk", None) is not None:
            vk = key.vk
            # Numpad 0-9
            if 96 <= vk <= 105:
                return f"kp_{vk - 96}"
            return f"vk_{vk}"
        return str(key)

    # Function keys / special keys
    if isinstance(key, Key):
        if key in _KEY_ALIAS:
            return _KEY_ALIAS[key]
        # F1~F24
        name = getattr(key, "name", None) or str(key)
        if name.startswith("f") and name[1:].isdigit():
            return name  # e.g. 'f1'
        # Fallback
        return name.replace("Key.", "") if name.startswith("Key.") else name

    # Fallback
    return str(key)


def auto_convert_engine(onnx_path):
    """
    Enhanced auto-conversion function that checks TensorRT environment availability first

    Args:
        onnx_path: Path to ONNX model

    Returns:
        bool: Whether conversion was successful
    """
    if not TENSORRT_AVAILABLE:
        print("TensorRT environment not available, cannot convert to TRT engine")
        return False
    from src.inference_engine import auto_convert_engine as original_auto_convert_engine

    return original_auto_convert_engine(onnx_path)


def global_exception_hook(exctype, value, tb):
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[Global Exception] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("".join(traceback.format_exception(exctype, value, tb)))
        f.write("\n")
    print(
        "Program encountered uncaught exception, details written to error_log.txt. Please report this file to developers."
    )


# Set up global exception hooks
sys.excepthook = global_exception_hook
if hasattr(threading, "excepthook"):

    def thread_exception_hook(args):
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[Thread Exception] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
            )
            f.write("\n")
        print(
            "Child thread encountered uncaught exception, details written to error_log.txt. Please report this file to developers."
        )

    threading.excepthook = thread_exception_hook
