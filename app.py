"""
GuardRailNet - AI Road Guardrail Detection & Highway Safety Inspection Studio
Interactive Modern Streamlit Web Application
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Setup paths and environment
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["MPLCONFIGDIR"] = "/tmp"

from pipeline.infer import GuardRailDetector
from pipeline.train import train_guardrail_net
from pipeline.evaluate import run_evaluation
from data.dataset_generator import RoadSceneGenerator, generate_dataset
from utils.visualizer import CLASS_NAMES, CLASS_COLORS, plot_confusion_matrix, plot_training_history


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="GuardRailNet - AI Road Guardrail Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Glassmorphism Styling
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
    }

    /* Metric Cards */
    .metric-badge {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Risk Status Badges */
    .status-normal {
        color: #10b981;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        display: inline-block;
    }
    .status-critical {
        color: #f43f5e;
        font-weight: 700;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        display: inline-block;
    }
    .status-warning {
        color: #fbbf24;
        font-weight: 700;
        background: rgba(251, 191, 36, 0.15);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(weights_path: str = "checkpoints/guardnet_best.pt") -> GuardRailDetector:
    """Caches model detector in memory."""
    return GuardRailDetector(weights_path=weights_path)


def ensure_samples_exist():
    """Generates sample scenes if not already present."""
    sample_dir = "data/sample_media"
    if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
        os.makedirs(sample_dir, exist_ok=True)
        gen = RoadSceneGenerator((512, 512))
        scenarios = [
            ("sunny_highway_intact.jpg", "sunny", False),
            ("damaged_guardrail_crash.jpg", "sunny", True),
            ("sunset_concrete_barrier.jpg", "sunset", False),
            ("night_highway_inspection.jpg", "night", False),
            ("foggy_mountain_pass.jpg", "foggy", True),
        ]
        for fname, weather, force_dam in scenarios:
            img, _, _ = gen.generate_scene(weather=weather, force_damaged=force_dam)
            cv2.imwrite(os.path.join(sample_dir, fname), img)


ensure_samples_exist()

# Initialize Detector
try:
    detector = load_detector()
except Exception as e:
    st.error(f"Error loading GuardRailNet model: {e}")
    detector = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("GuardRailNet Settings")
    st.caption("AI-Powered Highway Safety Infrastructure System")

    st.markdown("---")
    st.subheader("Model Inference Parameters")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_threshold = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.subheader("HUD & Visualizer Overlays")
    show_hud = st.checkbox("Show Telemetry HUD Header", value=True)
    show_risk_zones = st.checkbox("Show Safe Road Corridor", value=True)

    st.markdown("---")
    st.subheader("Hardware & Acceleration")
    device_name = str(detector.device).upper() if detector else "CPU"
    st.info(f"⚡ Acceleration: **{device_name}**")
    weights_status = "Trained Weights Active" if detector and detector.has_trained_weights else "Base Checkpoint"
    st.success(f"🎯 Model Status: **{weights_status}**")


# Hero Header Section
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🛡️ GuardRailNet AI Vision Studio</div>
    <div class="hero-subtitle">
        End-to-End Deep Learning Architecture for Highway Guardrail Detection, Asset Classification, Structural Damage Assessment, and Real-Time Collision Risk Telemetry.
    </div>
</div>
""", unsafe_allow_html=True)


# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image & Dashcam Inspector",
    "🎥 Video Stream Processing",
    "🧪 Model Training & Evaluation Studio",
    "📑 Highway Asset Safety Report"
])


# ==============================================================================
# TAB 1: Image & Dashcam Inspector
# ==============================================================================
with tab1:
    st.subheader("Highway Infrastructure Analysis")

    input_mode = st.radio(
        "Choose Scene Source:",
        ["Pre-loaded Test Scenarios", "Upload Custom Image"],
        horizontal=True
    )

    image_to_process = None
    image_title = ""

    if input_mode == "Pre-loaded Test Scenarios":
        sample_scenarios = {
            "☀️ Scenario 1: Sunny Highway (Standard Metal W-Beam)": "data/sample_media/sunny_highway_intact.jpg",
            "🚨 Scenario 2: Crash Impact & Deformed Guardrail Breach": "data/sample_media/damaged_guardrail_crash.jpg",
            "🌅 Scenario 3: Sunset Highway Concrete Jersey Barrier": "data/sample_media/sunset_concrete_barrier.jpg",
            "🌙 Scenario 4: Night Road Inspection (Reflective Posts)": "data/sample_media/night_highway_inspection.jpg",
            "🌫️ Scenario 5: Foggy Mountain Pass Hazardous Section": "data/sample_media/foggy_mountain_pass.jpg",
        }
        selected_scenario = st.selectbox("Select Scenario:", list(sample_scenarios.keys()))
        img_path = sample_scenarios[selected_scenario]
        if os.path.exists(img_path):
            image_to_process = cv2.imread(img_path)
            image_title = selected_scenario
    else:
        uploaded_file = st.file_uploader("Upload Road/Highway Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_title = uploaded_file.name

    if image_to_process is not None and detector is not None:
        # Run Prediction
        results = detector.predict(image_to_process, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        annotated_bgr = results["annotated_image"]
        stats = results["stats"]
        detections = results["detections"]
        latency = results["latency_ms"]

        # Display Metrics Badges
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-value">{stats['total_barriers']}</div>
                <div class="metric-label">Barriers Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            dmg_color = "#f43f5e" if stats['damaged_count'] > 0 else "#10b981"
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-value" style="color: {dmg_color};">{stats['damaged_count']}</div>
                <div class="metric-label">Damaged / Deformed Segments</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            status_text = stats['collision_risk']
            if "NORMAL" in status_text:
                status_html = f'<span class="status-normal">NORMAL (SAFE)</span>'
            elif "CRITICAL" in status_text:
                status_html = f'<span class="status-critical">CRITICAL RISK</span>'
            else:
                status_html = f'<span class="status-warning">CAUTION</span>'

            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-value" style="font-size: 1.25rem; margin-top: 6px;">{status_html}</div>
                <div class="metric-label">Road Safety Status</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-value">{latency:.1f} <span style="font-size: 1rem;">ms</span></div>
                <div class="metric-label">Inference Latency</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side Image Comparison
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("#### 📷 Original Highway Scene")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_img2:
            st.markdown("#### 🛡️ GuardRailNet AI Vision Output")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Detections Data Table
        st.markdown("---")
        st.subheader("📋 Detected Road Infrastructure Details")

        if len(detections) > 0:
            df_rows = []
            for idx, d in enumerate(detections, 1):
                cid = d["class_id"]
                cname = d["class_name"]
                conf = f"{d['confidence']:.1%}"
                x1, y1, x2, y2 = d["bbox"]
                bbox_str = f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                condition = "⚠️ Damaged / Breached" if cid == 3 else "✅ Intact Functional"
                df_rows.append({
                    "ID": idx,
                    "Asset Class": cname,
                    "Condition": condition,
                    "Confidence": conf,
                    "Bounding Box [x1, y1, x2, y2]": bbox_str,
                    "Width (px)": int(x2 - x1),
                    "Height (px)": int(y2 - y1)
                })

            st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
        else:
            st.warning("No road guardrails detected with current confidence threshold.")


# ==============================================================================
# TAB 2: Video Stream Processing
# ==============================================================================
with tab2:
    st.subheader("🎥 Highway Dashcam Video Stream Analysis")
    st.caption("Inspect guardrail continuity and detect impact deformations along continuous road footage.")

    video_source = st.radio("Video Source:", ["Generate Synthetic Highway Drive Video", "Upload MP4 Dashcam Video"], horizontal=True)

    if video_source == "Generate Synthetic Highway Drive Video":
        if st.button("🎬 Generate & Process 5-Second Highway Drive Footage", key="gen_vid"):
            with st.spinner("Generating procedural driving frames with dynamic perspective..."):
                gen = RoadSceneGenerator((480, 480))
                temp_video_path = "data/sample_media/simulated_drive.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 15
                out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (480, 480))

                # Render dynamic driving frames
                for f in range(60):
                    has_damage = (25 <= f <= 45)
                    frame, _, _ = gen.generate_scene(weather="sunny", force_damaged=has_damage)
                    # Add driving speed text
                    cv2.putText(frame, f"SPEED: 105 KM/H | FRAME {f+1}/60", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    out_writer.write(frame)
                out_writer.release()

            st.success("Video sequence generated! Processing with GuardRailNet AI...")
            proc_bar = st.progress(0.0)

            annotated_video_path = "data/sample_media/simulated_drive_detected.mp4"
            summary = detector.process_video(
                temp_video_path,
                annotated_video_path,
                conf_threshold=conf_threshold,
                progress_callback=lambda cur, tot: proc_bar.progress(cur / float(tot))
            )

            st.success(f"Processing Complete! Processed {summary['total_frames']} frames at {summary['fps_processed']} FPS.")
            st.metric("Damage Detection Frames", f"{summary['damage_frames']} / {summary['total_frames']}")

            if os.path.exists(annotated_video_path):
                st.video(annotated_video_path)
    else:
        uploaded_video = st.file_uploader("Upload Dashcam Video (MP4 / MOV)", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            temp_input = "data/temp_uploaded_video.mp4"
            temp_output = "data/temp_annotated_video.mp4"
            with open(temp_input, "wb") as f:
                f.write(uploaded_video.read())

            if st.button("🚀 Process Uploaded Video"):
                proc_bar = st.progress(0.0)
                summary = detector.process_video(
                    temp_input,
                    temp_output,
                    conf_threshold=conf_threshold,
                    progress_callback=lambda cur, tot: proc_bar.progress(cur / float(tot))
                )
                st.success(f"Done! {summary['total_frames']} frames processed.")
                st.video(temp_output)


# ==============================================================================
# TAB 3: Model Training & Evaluation Studio
# ==============================================================================
with tab3:
    st.subheader("🧪 Deep Learning Training & Benchmark Evaluation Studio")

    col_train1, col_train2 = st.columns([1, 1])

    with col_train1:
        st.markdown("#### ⚡ GuardRailNet Training Controls")
        train_epochs = st.slider("Training Epochs", 2, 30, 8, key="slider_epochs")
        train_batch = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=4, key="slider_batch")

        if st.button("🚀 Start Deep Learning Training Session", key="btn_train"):
            train_progress = st.progress(0.0)
            status_box = st.empty()
            metric_placeholder = st.empty()

            def ui_callback(epoch, total_epochs, t_loss, v_loss, v_map):
                train_progress.progress(epoch / float(total_epochs))
                status_box.info(f"Training Epoch {epoch}/{total_epochs}...")
                metric_placeholder.write(f"📉 Train Loss: `{t_loss:.4f}` | 📊 Val Loss: `{v_loss:.4f}` | 🎯 Val mAP@50: `{v_map:.4f}`")

            with st.spinner("Training GuardRailNet with Multi-Task Loss and Spatial Attention..."):
                train_result = train_guardrail_net(
                    epochs_override=train_epochs,
                    batch_size_override=train_batch,
                    progress_callback=ui_callback
                )

            st.success(f"Training Complete in {train_result['total_time_seconds']:.1f}s! Best Val mAP: {train_result['best_map']:.4f}")
            detector.has_trained_weights = True

    with col_train2:
        st.markdown("#### 📊 Evaluation & Benchmark")
        if st.button("📈 Run Full Validation Benchmark", key="btn_eval"):
            with st.spinner("Running benchmark across validation set..."):
                eval_res = run_evaluation(conf_thresh=conf_threshold, iou_thresh=iou_threshold)
            st.success(f"Validation Benchmark Finished! mAP@50: {eval_res['mAP@50']:.4f}")

    st.markdown("---")
    st.markdown("#### 📉 Training History & Loss Curves")

    history_file = "checkpoints/train_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

        fig_curves = plot_training_history(history)
        st.pyplot(fig_curves)
    else:
        st.info("No training history logged yet. Click 'Start Deep Learning Training Session' to run model training.")

    # Show Confusion Matrix if generated
    cm_path = "checkpoints/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.markdown("#### 🎯 Guardrail Condition Confusion Matrix")
        st.image(cm_path, width=580)


# ==============================================================================
# TAB 4: Highway Asset Safety Report
# ==============================================================================
with tab4:
    st.subheader("📑 Highway Asset & Guardrail Condition Audit Report")
    st.caption("Standardized DOT safety inspection report for smart city maintenance and highway authority documentation.")

    if 'results' in locals() and results is not None:
        report_data = {
            "inspection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "system": "GuardRailNet AI Highway Vision v1.0",
            "scene_name": image_title,
            "overall_status": stats["collision_risk"],
            "total_barriers_detected": stats["total_barriers"],
            "damaged_segments_identified": stats["damaged_count"],
            "detected_assets": detections
        }

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"**Inspection Time:** `{report_data['inspection_timestamp']}`")
            st.markdown(f"**Highway Target:** `{report_data['scene_name']}`")
            st.markdown(f"**Safety Classification:** `{report_data['overall_status']}`")

        with col_r2:
            st.markdown(f"**Total Barriers Detected:** `{report_data['total_barriers_detected']}`")
            st.markdown(f"**Deformed / Breached Segments:** `{report_data['damaged_segments_identified']}`")

        st.markdown("---")
        st.download_button(
            label="📥 Download JSON Inspection Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"guardrail_audit_{int(time.time())}.json",
            mime="application/json"
        )
    else:
        st.info("Run an image analysis in Tab 1 to populate the inspection audit report.")
