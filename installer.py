"""
ZTXAI Installer - Console wizard for installing dependencies

Usage: python installer.py
"""

import os
import subprocess
import sys
import urllib.request

# URLs
CUDA_URL = "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe"
CUDA_FILENAME = "cuda_12.8.0_571.96_windows.exe"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def print_header():
    print("\n" + "=" * 50)
    print("         ZTXAI Installer")
    print("=" * 50 + "\n")


def print_step(step_num, total, name):
    print(f"\n[Step {step_num}/{total}] {name}")
    print("-" * 40)


# =============================================================================
# Check functions (via import)
# =============================================================================


def check_requirements() -> bool:
    """Check if base requirements are installed"""
    try:
        import cv2
        import dearpygui
        import numpy
        import PIL
        import pynput

        return True
    except ImportError:
        return False


def find_cuda_in_path(version: str = "12") -> bool:
    """Check if CUDA is installed by looking at PATH (like helper.py)"""
    path = os.environ.get("PATH", "")
    for p in path.split(os.pathsep):
        if "CUDA" in p.upper() and version in p:
            return True
    return False


def check_torch_cuda() -> bool:
    """Check if PyTorch with CUDA is installed and working"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_tensorrt_installed() -> bool:
    """Check if TensorRT is installed"""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def check_onnx_gpu() -> bool:
    """Check if ONNX Runtime has GPU support (init CUDA via torch first)"""
    try:
        # Initialize CUDA via torch first so ONNX can detect providers
        import torch
        _ = torch.cuda.is_available()

        import onnxruntime as ort
        providers = ort.get_available_providers()
        return (
            "CUDAExecutionProvider" in providers
            or "TensorrtExecutionProvider" in providers
        )
    except:
        return False


def check_onnx_directml() -> bool:
    """Check if ONNX Runtime has DirectML support"""
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except:
        return False


# =============================================================================
# Install functions
# =============================================================================


def run_pip(args: list, show_output: bool = True):
    """Run pip command"""
    cmd = [sys.executable, "-m", "pip"] + args
    if show_output:
        subprocess.run(cmd)
    else:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def install_requirements():
    """Install base requirements from requirements.txt"""
    print("Installing base requirements...")
    run_pip(["install", "-r", "requirements.txt"])
    print("Done!")


def download_file(url: str, filename: str):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    print("This may take a while (file is ~2.5 GB)")
    print()

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def download_and_install_cuda():
    """Download CUDA installer and wait for user to install"""
    # Check if installer already downloaded
    if os.path.exists(CUDA_FILENAME):
        print(f"CUDA installer already downloaded: {CUDA_FILENAME}")
    else:
        if not download_file(CUDA_URL, CUDA_FILENAME):
            print("\nFailed to download CUDA installer.")
            print("Please download manually from:")
            print(CUDA_URL)
            input("\nPress ENTER when you have CUDA installed...")
            return

    print("\n" + "=" * 50)
    print("IMPORTANT: Please run the CUDA installer now!")
    print(f"File: {os.path.abspath(CUDA_FILENAME)}")
    print("=" * 50)
    print("\nRecommended installation options:")
    print("- Express installation is fine")
    print("- Make sure CUDA Toolkit is selected")
    print()

    # Open installer
    try:
        os.startfile(CUDA_FILENAME)
        print("Installer launched...")
    except:
        print(f"Could not auto-launch installer. Please run manually:")
        print(os.path.abspath(CUDA_FILENAME))

    input("\nPress ENTER when CUDA installation is complete...")

    # Verify installation
    print("\nVerifying CUDA installation...")
    if find_cuda_in_path():
        print("CUDA found in PATH!")
    else:
        print("Warning: CUDA not detected in PATH.")
        print("You may need to restart your computer for changes to take effect.")
        input("Press ENTER to continue anyway...")


def install_torch_cuda():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA support...")
    print(f"Using index: {TORCH_INDEX}")
    # Uninstall existing torch first
    run_pip(["uninstall", "torch", "torchvision", "torchaudio", "-y"], show_output=False)
    # Install CUDA version
    run_pip(["install", "torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX])

    # Verify
    if check_torch_cuda():
        import torch
        print(f"PyTorch {torch.__version__} with CUDA installed successfully!")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: PyTorch CUDA not working.")
        print("Make sure CUDA Toolkit is installed and restart may be needed.")


def install_tensorrt():
    """Install TensorRT via pip"""
    print("Installing TensorRT...")
    # Uninstall any existing versions first
    run_pip([
        "uninstall", "tensorrt", "tensorrt-bindings", "tensorrt-cu12",
        "tensorrt-cu12_bindings", "tensorrt-cu12_libs", "tensorrt-libs", "-y"
    ], show_output=False)
    run_pip(["install", "tensorrt"])

    # Verify
    if check_tensorrt_installed():
        import tensorrt
        print(f"TensorRT {tensorrt.__version__} installed successfully!")
    else:
        print("Warning: TensorRT installation may have failed.")
        print("Try: pip install tensorrt-cu12")


def install_onnx_gpu():
    """Install ONNX Runtime with GPU support"""
    print("Installing ONNX Runtime GPU...")
    # First uninstall any existing onnxruntime
    run_pip(
        ["uninstall", "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "-y"],
        show_output=False,
    )
    # Install GPU version
    run_pip(["install", "onnxruntime-gpu"])

    # Verify
    if check_onnx_gpu():
        print("ONNX Runtime GPU installed successfully!")
    else:
        print("Warning: ONNX Runtime GPU providers not detected.")
        print("This might be normal if CUDA is not properly configured.")


def install_onnx_directml():
    """Install ONNX Runtime with DirectML support"""
    print("Installing ONNX Runtime DirectML...")
    # First uninstall any existing onnxruntime
    run_pip(
        ["uninstall", "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "-y"],
        show_output=False,
    )
    # Install DirectML version
    run_pip(["install", "onnxruntime-directml"])

    # Verify
    if check_onnx_directml():
        print("ONNX Runtime DirectML installed successfully!")
    else:
        print("Warning: DirectML provider not detected.")


# =============================================================================
# Main wizard
# =============================================================================


def select_mode() -> int:
    """Let user select installation mode"""
    print("Select installation mode:\n")
    print("[1] GPU (CUDA + TensorRT) - Maximum performance")
    print("    Requires NVIDIA GPU with CUDA support")
    print()
    print("[2] CPU-only (DirectML) - Universal compatibility")
    print("    Works on any GPU (NVIDIA, AMD, Intel)")
    print()

    while True:
        choice = input("Enter choice (1/2): ").strip()
        if choice == "1":
            print("\n--- GPU Mode Selected ---")
            return 1
        elif choice == "2":
            print("\n--- CPU Mode Selected ---")
            return 2
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    print_header()

    mode = select_mode()

    if mode == 1:  # GPU mode
        steps = [
            ("Base Requirements", check_requirements, install_requirements),
            ("CUDA Toolkit", find_cuda_in_path, download_and_install_cuda),
            ("PyTorch CUDA", check_torch_cuda, install_torch_cuda),
            ("TensorRT", check_tensorrt_installed, install_tensorrt),
            ("ONNX Runtime GPU", check_onnx_gpu, install_onnx_gpu),
        ]
    else:  # CPU mode
        steps = [
            ("Base Requirements", check_requirements, install_requirements),
            ("ONNX Runtime DirectML", check_onnx_directml, install_onnx_directml),
        ]

    total_steps = len(steps)

    for i, (name, check_fn, install_fn) in enumerate(steps, 1):
        print_step(i, total_steps, name)

        print("Checking...", end=" ")
        if check_fn():
            print("Already installed, skipping")
        else:
            print("Not found")
            install_fn()

    # Final summary
    print("\n" + "=" * 50)
    print("         Installation Complete!")
    print("=" * 50)
    print("\nTo start ZTXAI, run:")
    print("  python main.py")
    print("  or double-click run.bat")
    print()

    # Show status
    print("Status:")
    if mode == 1:
        print(f"  CUDA:       {'OK' if find_cuda_in_path() else 'Not detected (may need restart)'}")
        print(f"  PyTorch:    {'OK' if check_torch_cuda() else 'Not working'}")
        print(f"  TensorRT:   {'OK' if check_tensorrt_installed() else 'Not installed'}")
        print(f"  ONNX GPU:   {'OK' if check_onnx_gpu() else 'Not detected'}")
    else:
        print(f"  ONNX DirectML: {'OK' if check_onnx_directml() else 'Not detected'}")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
