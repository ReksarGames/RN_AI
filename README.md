<div align="center">

# 🎯 RN_AI - YOLO Based AI Assistant

**Language:** [Русский](README_RU.md) | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.12%2B-FFD43B?logo=python)](#)
[![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?logo=windows)](#)
[![GUI](https://img.shields.io/badge/GUI-DearPyGui-2ea44f)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

</div>

## 🧠 Overview 

**RN_AI** is a high-performance YOLO-based AI assistant with an intuitive **DearPyGui** interface. 
Built with **Kalman filtering**, **SunOner algorithms**, and optimized for competitive gaming.

![RN_AI Demo](docs/demo.gif)

- **Entry point:** `main.py`
- **Core logic:** `src/` and `core/`
- **Inspired by:** [SunOner's sunone_aimbot_cpp](https://github.com/SunOner/sunone_aimbot_cpp)

> 💡 *The SunOner project is solid and well-designed, but this Python pipeline delivers even higher FPS than C++.*

---

## 📊 Performance Comparison (RTX 2060)

All benchmarks performed on **NVIDIA RTX 2060** - screen capture excluded.

<table>
<tr>
<th align="center">Metric</th>
<th align="center">SunOner (C++)</th>
<th align="center">RN_AI (Python)</th>
</tr>
<tr>
<td><strong>Average FPS</strong></td>
<td align="center">90 – 140 FPS</td>
<td align="center"><strong>~240 FPS</strong> ⚡</td>
</tr>
<tr>
<td><strong>Pipeline Latency</strong></td>
<td align="center">~2 ms</td>
<td align="center">~3.37 ms</td>
</tr>
<tr>
<td><strong>Pipeline Overhead</strong></td>
<td align="center">Very Low</td>
<td align="center">Minimal</td>
</tr>
</table>

### 📈 RN_AI Profiling Results

```
┌─────────────────────────────┐
│ [PROFILE] RTX 2060          │
├─────────────────────────────┤
│ Capture  : 0.09 ms          │
│ Preproc  : 0.00 ms          │
│ Infer    : 3.12 ms          │
│ Post     : 0.16 ms          │
├─────────────────────────────┤
│ Total    : 3.37 ms ✓        │
└─────────────────────────────┘
```

### 🎯 Key Benefits

While RN_AI has slightly higher end-to-end latency, it delivers:
- ✅ **Significantly higher FPS** (~240 vs ~140)
- ✅ Smoother tracking and motion
- ✅ Higher temporal resolution
- ✅ Flexible experimentation in Python
- ✅ Rapid iteration capability

### ⚠️ Important Warnings

| ⚠️ Warning | ℹ️ Note |
|-----------|---------|
| **Use at your own risk.** You may be banned by VAC or Vanguard. | Optimized for **RTX 20xx** and newer GPUs |

---

## ⚡ Quick Start 

### Installation & Setup

<details open>
<summary><b>Step 1: Install Dependencies</b></summary>

```batch
install.bat
```
</details>

<details open>
<summary><b>Step 2: Run the Application</b></summary>

```batch
run.bat
```
</details>

---

## 🖥️ GUI Preview 

Intuitive interface with real-time monitoring and configuration:

| System Panel | Aim System | Class Detection |
| :---: | :---: | :---: |
| ![System](docs/gui/system.png) | ![Aim](docs/gui/aim.png) | ![Classes](docs/gui/classes.png) |

| Driver Control | Bypass Settings | Strafe Config | Config Manager |
| :---: | :---: | :---: | :---: |
| ![Driver](docs/gui/driver.png) | ![Bypass](docs/gui/bypass.png) | ![Strafe](docs/gui/strafe.png) | ![Config](docs/gui/config.png) |

---


## 📁 Project Structure 

```
RN_AI/
├── main.py                  # Entry point - Start here
├── cfg.json                 # Main configuration file
├── requirements.txt         # Python dependencies
├── *.bat                    # Helper scripts (build, install, run)
│
├── src/                     # Inference & Utilities
│   ├── app.py              # Application core
│   ├── inference_engine.py # YOLO inference pipeline
│   ├── screenshot_manager.py # Screen capture
│   ├── pid.py              # PID controller
│   ├── profiler.py         # Performance profiling
│   └── ...
│
├── core/                    # GUI & Control Logic
│   ├── gui.py              # DearPyGui interface
│   ├── aiming.py           # Aiming algorithms
│   ├── recoil.py           # Recoil compensation
│   ├── config.py           # Config management
│   ├── input.py            # Input handling
│   └── ...
│
└── docs/                    # Documentation & Assets
    ├── demo.gif            # Demo animation
    ├── gui/                # GUI screenshots
    └── HELP.md             # Help documentation
```

---

## 🧪 Profiling & Performance Analysis

### CPU Profiling

Analyze CPU usage patterns with `py-spy`:

```batch
profile.bat
```

This generates detailed CPU profiling data to identify bottlenecks and optimization opportunities.

---

## 🤖 Model Support 

RN_AI supports multiple YOLO model formats and versions:

| Format | Support | Preference | Notes |
|--------|---------|-----------|-------|
| **ONNX** | ✅ Full | Standard | Cross-platform compatibility |
| **TensorRT** (.engine) | ✅ Full | **Preferred** ⭐ | Best performance on NVIDIA GPUs |

### Supported YOLO Versions
- ✅ YOLO v8 through v12
- 📁 Example models available in `.weight/`
- 🎛️ Fully configurable from GUI

---

## 📝 License & Credits

- **Inspired by:** [SunOner/sunone_aimbot_cpp](https://github.com/SunOner/sunone_aimbot_cpp)
- **Framework:** DearPyGui
- **AI Model:** YOLO Object Detection

---

<div align="center">

**Made with ❤️ for the gaming community**

*High-performance, flexible, and Pythonic*

</div>

## ✨ Features 
- Makcu support with optional input masking
- Kalman filtering & prediction
- Dead-zone logic (no target jumping)
- Multiple capture sources:
    * Screen
    * OBS
    * Capture Card
- Class filtering:
    * Enable / disable
    * Custom class sets
    * Per-class aim positions

## 🎥 Capture 
- Sources: Standard (screen), OBS, Capture Card.
- Capture offsets can be set in the GUI.

## 🧷 Config & Classes 
- Stored in `cfg.json`. Auto-generated if missing.
- Class names can be loaded or entered manually in the GUI.
- Per-class confidence/IOU is supported.

## Documentation 📘
[Full GUI and parameter help](docs/HELP_EN.md)
