# 📘 RN_AI Configuration Guide

## 📋 Settings Structure

All parameters are divided into three levels:

| Type | Description | Scope |
|------|-------------|-------|
| **Global** | Interface, capture, debug | Applies to entire project |
| **Groups** | Profiles with model and parameters | Independent profiles |
| **Buttons** | Aiming/trigger settings | Applies to selected button |

### ⚠️ Important Notes

- ℹ️ Parameters in the **Aim** tab apply to the active aiming button (**Targeting Button**).
- 🔄 If Triggerbot is on a different button — temporarily assign it as Targeting Button, configure it, then change back.
- 🎯 If the class list is empty — system uses ALL classes.

---

## 1️⃣ System ⚙️

### 🎨 Interface

| Parameter | Description | Note |
|-----------|-------------|------|
| **GUI DPI Scale** | Window scale | Higher = larger elements |
| **Auto Detect** | Auto DPI detection | Based on system settings |
| **UI Language** | Interface language | English / Русский |
| **Font Scale** | Font size scale | Independent of DPI |
| **UI Width Scale** | Window width | ⚠️ Requires restart |
| **UI Scale Detail** | Current DPI | Informational |

> **💡 Tip:** After changing UI Width Scale, restart the application to apply changes.

### 🐛 Overlay & Debugging

| Parameter | Action |
|-----------|--------|
| **Inference Window** | 🤖 Debug window with boxes and preview |
| **Print FPS** | 📊 Print FPS to console |
| **Show Motion Speed** | 📈 Target movement speed |
| **Show Curve** | 📉 Movement curve points |
| **Show Infer Time** | ⏱️ Inference execution time |
| **Screenshot Separation** | 🔄 Multi-threaded capture/inference |

### 🎯 Small Target Enhancement

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **Enable Small Target Enhancement** | ✅ Boost priority of small targets | Higher weight |
| **Enable Small Target Smoothing** | 🔄 Smooth coordinates | Smoother movement |
| **Adaptive NMS** | 📖 Adaptive NMS for small objects | Better separation |
| **Small Target Boost** | 📊 Weight multiplier | Amplification |
| **Smooth History Frames** | 📉 Frames for averaging | Stability |
| **Small Target Threshold** | 📏 Size threshold for "small" target | px |
| **Medium Target Threshold** | 📏 Threshold for "medium" target | px |

> ⚠️ **Note:** Improves stability but may reduce FPS

### ⚡ Performance

| Parameter | Action |
|-----------|--------|
| **Turbo Mode** | 🚀 Aggressive optimizations |
| **Skip Frame Processing** | ⏭️ Skip frames (faster, less frequent updates) |

### 🔍 Debug & Visualization

| Parameter | Action |
|-----------|--------|
| **Recoil Debug** | 🔫 Recoil compensation debugging |
| **Class Priority Debug** | 🎯 Class priority debugging |
| **Show Aim Scope** | ⭕ Aiming area circle |

### 📹 Capture

| Parameter | Description | Values |
|-----------|-------------|--------|
| **Capture Source** | Capture type | Standard / OBS / Capture Card |
| **Capture Offset X/Y** | Center offset | Pixels (Standard only) |
| **OBS IP/Port/FPS** | Network capture | IP:Port, FPS |
| **Capture Device** | Device ID | OS dependent |
| **Capture FPS** | Source FPS | 60/120/144 etc. |
| **Capture Resolution** | Stream resolution | WxH (e.g.: 1920x1080) |
| **Capture Crop Size** | Crop size | Pixels |
| **Video Codec** | FourCC | NV12, MJPG, YUYV etc. |
| **Current Capture** | Active source | Informational |

---

## 2️⃣ Aim (Aiming) 🎯

### 🔄 Algorithm Flow

```
┌────────────────────────────────────────────────────┐
│ 1️⃣  YOLO Model                         │
│    Object detection → Bounding boxes    │
└────────────────────────────────────────────────────┘
                       │
┌────────────────────────────────────────────────────┐
│ 2️⃣  Select config by button            │
│    Targeting Button → Config             │
└────────────────────────────────────────────────────┘
                       │
┌────────────────────────────────────────────────────┐
│ 3️⃣  Target selection                   │
│    Priorities + Aim Scope → Best target  │
└────────────────────────────────────────────────────┘
                       │
┌────────────────────────────────────────────────────┐
│ 4️⃣  Aim point                          │
│    Aim Position (inside box)            │
└────────────────────────────────────────────────────┘
                       │
┌────────────────────────────────────────────────────┐
│ 5️⃣  Movement controller                │
│    PID / Sunone → Mouse movement        │
└────────────────────────────────────────────────────┘
                       │
┌────────────────────────────────────────────────────┐
│ 6️⃣  Trigger (optional)                 │
│    If target in trigger zone → Fire     │
└────────────────────────────────────────────────────┘
```

### 🎮 Aim Controller (Algorithm Selection)

| Controller | Description | Best For |
|-----------|-------------|----------|
| **PID** | Classical proportional controller | Stable, predictable movements |
| **Sunone** | Advanced algorithm (Kalman + prediction) | Smoothness, human-like movements |

#### 💡 Recommendations

- 🟢 **Beginners:** Try **PID** for stability
- 🟡 **Advanced:** Experiment with **Sunone** for smoothness

### 📊 Sunone Settings (Basic Smoothing)

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **Enable Smoothing** | Enable smoothing | On/Off |
| **Tracking Smooth** | Tracking smoothing | Smoother target following |
| **Smoothness** | Smoothing strength | Higher = Smoother + slower |

### 🔬 Kalman Filter (Sunone)

| Parameter | Purpose | Guidance |
|-----------|---------|----------|
| **Enable Kalman** | Enable filter | Smooths model noise |
| **Q** (Process Noise) | Change speed | ⬆️ Q = faster + jittery |
| **R** (Measurement Noise) | Model trust | ⬆️ R = smoother + slower |
| **Kalman Speed X/Y** | Axis multiplier | Speed adjustment |
| **Reset Threshold** | Filter reset | On sudden position jumps |

### 🎯 Prediction (Sunone)

| Parameter | Description | Note |
|-----------|-------------|------|
| **Prediction Mode** | Prediction type | Standard / Kalman / Kalman+Standard |
| **Prediction Interval** | Update step | Lower = more frequent updates |
| **Kalman Lead (ms)** | Target lead | Lag compensation |
| **Kalman Max Lead (ms)** | Max lead | Limiter |
| **Velocity Smoothing** | Velocity smoothing | Reduces jitter |
| **Velocity Scale** | Velocity scale | Amplifies/weakens lead |
| **Prediction Kalman Q/R** | Filter parameters | For prediction block |
| **Future Positions** | Future points | How many to calculate |
| **Draw Future Positions** | Show positions | Debug visualization |
| **Prediction Preview** | Demo | See algorithm work |

### 📈 Speed Curve (Sunone)

Controls mouse movement speed based on distance to target.

| Parameter | Purpose | Effect |
|-----------|---------|--------|
| **Min Speed** | Minimum speed | At close range |
| **Max Speed** | Maximum speed | At far range |
| **Snap Radius** | "Snap" zone | Accelerates target approach |
| **Near Radius** | Speed change area | Distance in pixels |
| **Curve Exponent** | Curve steepness | Sharpness of speed change |
| **Snap Boost** | Snap amplification | Effect multiplier |
| **Speed Curve Preview** | Visualization | 📊 Curve graph |

### 🐛 Sunone Debug

| Parameter | Action |
|-----------|--------|
| **Show Predicted Target** | 🎯 Target prediction point |
| **Show Step** | 📍 Next movement step |
| **Show Future Points** | 📊 Chain of future positions |

### 👆 Long Press

| Parameter | Description |
|-----------|-------------|
| **Long Press No Lock Y** | Don't lock Y when holding |
| **Long Press Threshold** | Hold duration for "long" press (ms) |

### 🔄 Target Switching

| Parameter | Description |
|-----------|-------------|
| **Target Switch Delay** | Protection from instant switching |
| **Target Reference Class** | Class for tracking |

### 🎯 Aim Scope & Offset

| Parameter | Description |
|-----------|-------------|
| **Min Offset** | Min vertical shift |
| **Aim Scope** | Target selection radius (px) |

### 🔐 Smart Target Lock (Dynamic Scope)

| Parameter | Description |
|-----------|-------------|
| **Smart Target Lock** | Hold current target |
| **Min Scope** | Min area size when locked |
| **Shrink Duration** | Area shrink time (ms) |
| **Recover Duration** | Recovery time (ms) |
| **Lock Distance** | Max tracking distance (px) |

### ⭕ Move Deadzone

| Parameter | Description |
|-----------|-------------|
| **Move Deadzone** | Min amplitude for movement (px) |

### ⚖️ Aim Weights (Weight Coefficients)

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **Distance Weight** | Distance weight | Closer = higher priority |
| **Center Weight** | Screen center weight | Toward center |
| **Size Weight** | Target size weight | Larger = higher priority |

### 🎮 PID Controller Parameters

| Parameter | Purpose | Recommendation |
|-----------|---------|-----------------|
| **Kp** | Response speed | Higher = faster, less stable |
| **Ki** | Steady-state error | Caution! Can cause oscillation |
| **Kd** | Damping | Reduces jitter, don't overdo it |
| **X/Y Limit** | Integral limit | Prevents overshoot |
| **X/Y Smooth** | Axis smoothing | Individual per axis |
| **Smooth Algorithm** | Filter model | Filtering algorithm choice |
| **Smooth Deadzone** | No smoothing zone | Very small movements |

### 🔫 Trigger Config

| Parameter | Description | Applied |
|-----------|-------------|---------|
| **Auto Trigger** | Auto fire when target in zone | On trigger activation |
| **Continuous Trigger** | Hold fire button | While target in zone |
| **Trigger Recoil** | Recoil compensation | During trigger |
| **Trigger Delay** | Pre-fire delay | ms |
| **Press Duration** | Button hold duration | ms |
| **Trigger Cooldown** | Post-fire pause | ms |
| **Random Delay** | Delay randomization | Obfuscation |
| **X/Y Trigger Scope** | Zone size | Relative to box |
| **X/Y Trigger Offset** | Zone offset | Pixels |
| **Preview** | Visualization | 🔴 Red rectangle |

### ⚠️ Important About Triggerbot

- **Works by holding its button**
- 🔘 If **separate Triggerbot Button** → fires independently
- 🔘 If **matches Targeting Button** → works together with aiming

---

## 3️⃣ Classes 🧩

### 📝 Class Management

| Parameter | Action | Note |
|-----------|--------|------|
| **Class Names File** | Load from .txt file | One per line |
| **Load Names** | 📂 Import from file | Overwrites current list |
| **Class Names (Manual)** | Manual input | Each line = class |
| **Apply Names** | ✅ Apply list | Saves changes |

### 🎯 Class Configuration

| Parameter | Description | Values |
|-----------|-------------|--------|
| **Class Priority** | Priority order | Example: `7-0-4` |
| **Inference Classes** | Enable/disable | Checkboxes for each |
| **Select Class** | Select to edit | Current class |
| **Confidence Threshold** | Confidence threshold | 0.0 — 1.0 |
| **IOU** | NMS threshold | 0.0 — 1.0 |

### 🎯 Aim Points

| Parameter | Description | Range | Note |
|-----------|-------------|-------|------|
| **Aim Position** | Main point | 0.0 — 1.0 | 0=top, 1=bottom |
| **Aim Position 2** | Second point | 0.0 — 1.0 | Two-stage behavior |
| **Aim Preview** | 📊 Visualization | — | Shows position in box |

> 💡 **If nothing selected in Inference Classes** → uses ALL classes

---

## 4️⃣ Driver 🧠

### 📋 Movement Curves

| Parameter | Description | Application |
|-----------|-------------|-------------|
| **Move Curve** | Human movement curve | Anti-cheat obfuscation |
| **Compensation Curve** | Compensation profile | Movement correction |
| **Control Points** | Curve control points | More = smoother |
| **Path Points Total** | Trajectory points | Path detail |

### 📏 Boundaries & Deviation

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **Horizontal Boundary** | X deviation range | Width variation |
| **Vertical Boundary** | Y deviation range | Height variation |
| **Distortion Mean** | Average distortion | Position shift |
| **Distortion StdDev** | Distortion variance | Variability |
| **Distortion Frequency** | Distortion frequency | How often to change |

### 🔧 Technical Parameters

| Parameter | Purpose |
|-----------|---------|
| **Move Method** | Movement send method (makcu) |

---

## 5️⃣ Bypass 🛡️

This tab obfuscates mouse clicks and movement while driver is active.

### 🖱️ Mouse Button Masking

| Parameter | Action |
|-----------|--------|
| **Mask Left** | Block LMB |
| **Mask Right** | Block RMB |
| **Mask Middle** | Block middle button |
| **Mask Side1** | Block side button 1 |
| **Mask Side2** | Block side button 2 |
| **Mask Wheel** | Block scroll wheel |

### 📍 Movement Masking

| Parameter | Action |
|-----------|--------|
| **Mask X Axis** | Block horizontal movement |
| **Mask Y Axis** | Block vertical movement |
| **Aim Mask X** | X mask for aiming only |
| **Aim Mask Y** | Y mask for aiming only |

---

## 6️⃣ Strafe (Recoil) 🔫

### 🎮 Recoil Profiles

| Parameter | Description | Action |
|-----------|-------------|--------|
| **Check Right Button** | Trigger condition | Only when RMB held |
| **Game / Gun** | Game/weapon list | Profile hierarchy |
| **Add/Delete Game** | Manage game | ➕ New / ➖ Delete |
| **Add/Delete Gun** | Manage weapon | ➕ New / ➖ Delete |

### 📊 Recoil Stages

| Parameter | Description | Values |
|-----------|-------------|--------|
| **Index** | Stage number | 1, 2, 3... |
| **Count** | Steps in stage | Integer |
| **X** | Horizontal offset | Pixels |
| **Y** | Vertical offset | Pixels |
| **Add/Delete Index** | Manage | ➕ / ➖ |

### 🎬 Trajectory Recoil (Mouse_re)

| Parameter | Description | Application |
|-----------|-------------|-------------|
| **Enable mouse_re Trajectory Recoil** | Use trajectories | Enable/disable |
| **Replay Speed** | Playback speed | Trajectory acceleration |
| **Pixel Enhancement Ratio** | Pixel amplification | Offset multiplier |
| **Import Trajectory File** | Import JSON | 📂 Trajectory file |
| **Clear Mapping** | Reset mappings | Clear all |
| **Current Status** | Status info | 📊 File and points |

---

## 7️⃣ Config ⚙️

### 🎮 Button Assignment

| Parameter | Purpose | Note |
|-----------|---------|------|
| **Targeting Buttons** | Aiming button | Main aim button |
| **Triggerbot Buttons** | Trigger button | Separate from aim |
| **Disable Headshot Buttons** | Disable head | Alt button |
| **Disable Headshot Class ID** | Class to disable | Class number |
| **Disable headshot (status)** | Disable status | Informational |

### 🤖 Model Settings

| Parameter | Description | Options |
|-----------|-------------|---------|
| **Group Name** | Profile name | Any string |
| **Add/Delete Group** | Manage profiles | ➕ / ➖ |
| **TRT** | TensorRT (if .engine exists) | ✅ On / ❌ Off |
| **Use Sunone Processing** | Sunone post-processing | ⚠️ May reduce FPS |
| **YOLO Version** | Decoder version | v8, v9, v10, v11, v12 |
| **Inference Model** | Model path | ONNX or .engine |
| **Select Model** | File selection | 📂 Browse |

### 🖥️ Control

| Button | Action |
|--------|--------|
| **Start** | ▶️ Start system |
| **Stop** | ⏹️ Stop system |
| **Save Config** | 💾 Save to cfg.json |

---

## 📚 Useful Notes ✅

### 🔄 Model Conversion

> 💡 If **TRT** is enabled and `.engine` not found → automatic ONNX conversion at startup

### 📹 Video Capture

> ⚠️ For **OBS** and **Capture Card** you need correct:
> - IP address and Port for OBS
> - Resolution, FPS, codec parameters

### ⚡ FPS Optimization

```
If FPS drops, try in order:

1. 🔴 Disable Sunone Processing
2. 📉 Lower model resolution
3. 🎯 Reduce Capture Resolution
4. ⏭️ Enable Skip Frame Processing
5. 🚀 Enable Turbo Mode
```

---

## 📖 Additional Help

- 📂 All parameters saved in **cfg.json**
- 🖥️ Use **System** tab for debugging (Inference Window, Show Infer Time)
- 🧪 Test settings gradually, don't change everything at once

<div align="center">

**Good luck with setup! 🎯**

</div>
