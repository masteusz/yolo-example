```
██╗   ██╗ ██████╗ ██╗      ██████╗
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗
 ╚████╔╝ ██║   ██║██║     ██║   ██║
  ╚██╔╝  ██║   ██║██║     ██║   ██║
   ██║   ╚██████╔╝███████╗╚██████╔╝
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝
```

# 🎯 YOLO Webcam Detection

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-Metal-000000?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

Real-time object detection using YOLOv11 with Mac hardware acceleration.

---

## 🚀 Tech Stack

<div align="center">

| Technology | Purpose |
|:----------:|:-------:|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core Language |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep Learning Framework |
| ![Ultralytics](https://img.shields.io/badge/Ultralytics-00FFFF?style=flat) | YOLOv11 Implementation |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | Computer Vision |
| ![Metal](https://img.shields.io/badge/Metal_MPS-000000?style=flat&logo=apple&logoColor=white) | GPU Acceleration |

</div>

## ✨ Features

- 🎯 **YOLOv11** - Latest YOLO model for state-of-the-art object detection
- ⚡ **Mac GPU Acceleration** - Uses Metal Performance Shaders (MPS) for faster inference
- 🖥️ **Fullscreen Display** - Immersive real-time detection experience
- 📹 **Webcam Support** - Live detection from your Mac's camera

## 📋 Requirements

- Python 3.13+
- macOS with Metal support
- Webcam

## 📦 Installation

```bash
# Install dependencies using uv
uv sync
```

## 🎮 Usage

```bash
# Run the application
uv run python main.py
```

The first run will automatically download the YOLOv11 model (~6MB).

**Controls:**
- Press `q` or `ESC` to quit

## 🔧 Model Options

You can switch to different YOLOv11 models in `main.py` for better accuracy:

- `yolo11n.pt` - Nano (fastest, default)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge (most accurate)
