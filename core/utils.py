import glob
import os
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image
from pynput.keyboard import Key, KeyCode

# Constants
BAR_HEIGHT = 2
SHADOW_OFFSET = 2
VERSION = "v2.1.5"
UPDATE_TIME = "2025-11-04"

# Global variable for TensorRT availability
TENSORRT_AVAILABLE = False

# Lazy-loaded kmNet module
_kmNet = None


def get_kmnet():
    """Lazily import and return kmNet module. Returns None if import fails."""
    global _kmNet
    if _kmNet is None:
        try:
            import kmNet
            _kmNet = kmNet
        except ImportError as e:
            print(f"kmNet import failed: {e}")
            return None
    return _kmNet


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
    from inference_engine import auto_convert_engine as original_auto_convert_engine

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
