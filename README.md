# 🛡️ RoadGuard AI: Real-Time Road Guardrail & Barrier Damage Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An End-to-End Deep Learning Architecture and Modern Web Application for Autonomous Driving Road Boundary Detection, Highway Guardrail Inspection, Structural Damage Assessment, and Real-Time Collision Risk Telemetry.**

[📖 Description](#-project-description) •
[✨ Key Features](#-key-features) •
[🧠 Model Architecture](#-deep-learning-architecture) •
[🖥️ Web Application](#-interactive-web-application-ui) •
[🚀 Quick Start](#-quick-start) •
[📊 Benchmark & Metrics](#-benchmark--metrics) •
[📂 Project Structure](#-project-structure)

</div>

---

## 📖 Project Description

**RoadGuard AI** (also known as *GuardRailNet*) is an end-to-end computer vision and deep learning system built from scratch to detect, monitor, and assess road safety barriers, crash guardrails, and roadside infrastructure in real time.

Highway guardrails—such as steel W-beam barriers, modular concrete Jersey barriers, and high-tension cable fences—are safety-critical assets designed to prevent fatal roadside departures, guide traffic, and absorb vehicle collision energy. However, keeping highway barrier networks inspected across thousands of kilometers is a massive operational challenge. Manual inspections are hazardous, slow, and expensive, leaving collision-damaged or missing barrier sections undetected for weeks or months.

**RoadGuard AI solves this problem by combining:**
1. **Custom Deep Learning Architecture (`GuardNet`)**: A purpose-built PyTorch object detection and condition classification network that features **Convolutional Block Attention (CBAM)** to specifically track elongated horizontal barrier lines along road edges.
2. **Multi-Task Detection & Condition Classification**: Simultaneously predicts precise bounding box coordinates, object presence, barrier asset types (W-beam, concrete, cable, delineator posts), and structural health (Intact vs. Crushed / Deformed / Breached).
3. **Safety Corridor & Collision Risk Telemetry**: Analyzes barrier continuity and road boundary clearance to issue real-time hazard warnings (Normal Clearance, Unguarded Shoulder Caution, or Critical Barrier Breach).
4. **Interactive Modern Web Application (`app.py`)**: A Streamlit dashboard designed with modern dark glassmorphism, offering dashcam image analysis, full video processing, an integrated model training studio, and automated Department of Transportation (DOT) inspection report generation.

---

### 🎯 Detected Infrastructure Classes & Conditions

| Class ID | Class Name | Description | Overlay Color |
| :---: | :--- | :--- | :---: |
| `0` | **`metal_w_beam`** | Galvanized corrugated steel highway W-beam guardrail | Cyan `[0, 210, 255]` |
| `1` | **`concrete_jersey_barrier`** | High-strength modular concrete highway median barrier | Amber `[255, 170, 0]` |
| `2` | **`cable_barrier`** | High-tension steel wire rope barrier system | Magenta `[170, 0, 255]` |
| `3` | **`damaged_guardrail`** | Crushed, dented, deformed, or breached barrier section | **Neon Red `[255, 50, 50]`** |
| `4` | **`hazard_post`** | Retro-reflective delineator post & terminal guide marker | Emerald `[50, 255, 120]` |

---

## ✨ Key Features

- **Built Completely from Scratch in PyTorch**:
  - No opaque third-party black boxes: custom residual backbone, spatial attention blocks, feature pyramid network, decoupled regression/classification heads, and vectorized Non-Maximum Suppression (NMS).
- **Spatial & Channel Attention (CBAM)**:
  - Specially tailored for highway scenes: channel attention focuses on reflective material textures, while 7x7 spatial attention captures elongated horizontal guardrail boundaries.
- **Multi-Task Loss Formulation**:
  - **Complete IoU (CIoU) Loss**: Accurately handles bounding box aspect ratio and center-point distances.
  - **Focal Loss**: Solves extreme foreground-background class imbalance between vast road/sky backgrounds and thin guardrail assets.
  - **Label-Smoothed Cross-Entropy**: Stabilizes multi-class barrier condition classification.
- **Procedural Synthetic Highway Generator**:
  - Generates realistic highway scenes under diverse lighting (Sunny, Sunset, Night, Foggy) with procedural vanishing perspectives, lane markings, barrier types, and damage deformation patterns.
- **Dashcam Video Stream Inspector**:
  - Processes MP4/AVI driving footage frame-by-frame with real-time barrier presence %, damage frame counters, and HUD telemetry.
- **DOT Safety Audit Exporter**:
  - Generates downloadable Department of Transportation (DOT) standardized inspection reports in JSON and CSV.
- **Hardware Agnostic Acceleration**:
  - Automatically leverages **Apple Silicon GPU (MPS)**, **NVIDIA CUDA**, or CPU.

---

## 🧠 Deep Learning Architecture

```
                    ┌─────────────────────────────────────────┐
                    │      Input Road Image (512 x 512 x 3)   │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │     GuardBackbone (ResNet + CBAM)       │
                    │  - Channel Attention (SE Block)         │
                    │  - Spatial Attention (Linear Edge Conv) │
                    └──────┬─────────────┬─────────────┬──────┘
                           │ P3 (1/8)    │ P4 (1/16)   │ P5 (1/32)
                           ▼             ▼             ▼
                    ┌─────────────────────────────────────────┐
                    │    PANet Neck (Top-Down + Bottom-Up)    │
                    └──────┬─────────────┬─────────────┬──────┘
                           ▼             ▼             ▼
                    ┌─────────────────────────────────────────┐
                    │       Decoupled Detection Heads         │
                    │  - Box Regression Branch [dx,dy,dw,dh]  │
                    │  - Objectness Branch (Focal Loss)       │
                    │  - Condition Classification Branch      │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │ Vectorized Non-Maximum Suppression (NMS)│
                    │   & Highway Safety Telemetry HUD Engine │
                    └─────────────────────────────────────────┘
```

1. **Backbone (`models/backbone.py`)**:
   - 4-stage residual convolutional feature extractor yielding multi-scale feature maps: P3 (stride 8 for fine delineators and post brackets), P4 (stride 16 for standard barrier segments), and P5 (stride 32 for global road horizon context).
   - Each residual block integrates CBAM (Channel Attention + Spatial Attention).
2. **Neck (`models/guardnet.py`)**:
   - Path Aggregation Network (PANet) combining top-down feature pyramids with bottom-up lateral connections to preserve both semantic and low-level edge features.
3. **Detection Heads**:
   - Decoupled anchor-free heads predicting bounding box offsets, objectness presence probabilities, and condition class logits.
4. **Loss Function (`models/loss.py`)**:
   - Total Loss = `5.0 * CIoU_Loss + 2.0 * Focal_Loss + 1.0 * Class_Loss`.

---

## 🖥️ Interactive Web Application (UI)

The project includes an interactive web interface built with **Streamlit** (`app.py`), organized into four modules:

1. **📸 Image & Dashcam Inspector**:
   - Drag-and-drop custom road imagery or select from 5 pre-loaded scenarios (Sunny Highway, Crash Impact Breach, Sunset Concrete Barrier, Night Inspection, Foggy Mountain Pass).
   - Side-by-side visualization: Raw scene vs. AI Vision overlay with bounding boxes, risk zones, and HUD banner.
   - Real-time telemetry cards: **Barriers Detected**, **Damage Alerts**, **Safety Status**, and **Latency (ms)**.
   - Interactive breakdown table with coordinates, confidence scores, and dimensions.
2. **🎥 Dashcam Video Stream Processor**:
   - Upload continuous dashcam video files or generate a synthetic 60-frame highway driving video with dynamic perspective.
   - Real-time video processing with detection overlays and damage frame logging.
3. **🧪 Live Training & Evaluation Studio**:
   - Train or fine-tune the model directly from the UI with custom epoch and batch size controls.
   - Live loss curves, validation mAP tracker, and Confusion Matrix heatmap.
4. **📑 DOT Highway Asset Safety Report**:
   - Formats the inspection results into official Department of Transportation (DOT) audit reports downloadable in JSON or CSV.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/SiddeGnaneswarReddy/-RoadGuard-AI-Real-Time-Road-Damage-Detection.git
cd -RoadGuard-AI-Real-Time-Road-Damage-Detection
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Studio
```bash
streamlit run app.py
```
> The application will open automatically at **`http://localhost:8501`**.

---

## 💻 CLI Commands

### 1. Run 1-Click Terminal Demo
```bash
python run_demo.py
```
*Processes all 5 test scenarios and saves annotated overlays to the `outputs/` folder.*

### 2. Train the Model from Scratch
```bash
python pipeline/train.py --epochs 15 --batch-size 8
```

### 3. Benchmark Evaluation
```bash
python pipeline/evaluate.py --weights checkpoints/guardnet_best.pt
```

### 4. Run Unit Test Suite
```bash
python -m unittest discover tests
```

---

## 📂 Project Structure

```
.
├── .gitignore                      # Git ignore file (optimized for PyTorch & GitHub limits)
├── LICENSE                         # MIT Open Source License
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Package dependencies
├── setup.py                        # Python package setup
├── config.json                     # System configuration & hyperparameters
├── config.py                       # Configuration loader
├── app.py                          # Streamlit Web Application
├── run_demo.py                     # 1-Click Terminal Demo Script
├── checkpoints/
│   ├── guardnet_best.pt            # Pre-trained deep learning model weights (~65 MB)
│   ├── confusion_matrix.png        # Benchmark evaluation confusion matrix plot
│   ├── evaluation_report.json      # Evaluation metrics log
│   └── train_history.json          # Training epoch loss and mAP log
├── data/
│   ├── dataset_generator.py        # Procedural synthetic road scene & YOLO annotation generator
│   ├── dataset.py                  # PyTorch Dataset, DataLoaders & Augmentations
│   ├── sample_media/               # 5 Pre-rendered test scenarios
│   ├── train/                      # Training images and YOLO formatted labels
│   └── val/                        # Validation images and YOLO formatted labels
├── models/
│   ├── __init__.py
│   ├── backbone.py                 # ResNet backbone with CBAM spatial & channel attention
│   ├── guardnet.py                 # GuardNet architecture with PANet & decoupled heads
│   └── loss.py                     # Multi-task loss (CIoU + Focal Loss + Cross-Entropy)
├── pipeline/
│   ├── __init__.py
│   ├── train.py                    # Multi-scale training pipeline with validation & scheduler
│   ├── evaluate.py                 # Benchmark evaluation suite (mAP@50, PR curves, Confusion Matrix)
│   └── infer.py                    # Inference engine for images, directories & videos
├── tests/
│   ├── __init__.py
│   ├── test_model.py               # Unit tests for architecture shapes, forward pass & loss
│   └── test_pipeline.py            # Unit tests for dataset, transforms & metrics
└── utils/
    ├── __init__.py
    ├── metrics.py                  # Vectorized IoU, AP, mAP@50 and confusion matrix
    └── visualizer.py               # Telemetry HUD renderer, safety corridor overlays & plotting
```

---

## 📊 Benchmark & Metrics

GuardRailNet was evaluated across diverse road conditions and barrier asset types:

```
=================================================================
🚦 GUARDRAILNET BENCHMARK EVALUATION RESULTS
=================================================================
Overall mAP@50: 0.2059

Class Name                   | Precision  | Recall     | F1-Score  
-----------------------------------------------------------------
metal_w_beam                 | 0.7317     | 0.5556     | 0.6316    
concrete_jersey_barrier      | 0.0455     | 0.1579     | 0.0706    
cable_barrier                | 0.0000     | 0.0000     | 0.0000    
damaged_guardrail            | 0.0094     | 0.0455     | 0.0156    
hazard_post                  | 0.1411     | 0.8718     | 0.2429    
=================================================================
```

- **Inference Latency**: Real-time throughput (~15–30 ms per frame on Apple Silicon MPS / NVIDIA CUDA GPUs).
- **Pre-trained Weights Included**: The repository includes `checkpoints/guardnet_best.pt` so you can run inference immediately without needing to train first.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome!
1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/NewCapability`)
3. Commit your Changes (`git commit -m 'feat: Add NewCapability'`)
4. Push to the Branch (`git push origin feature/NewCapability`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
