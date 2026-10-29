"""
Quick terminal demo script for GuardRailNet.
Runs inference across test highway scenes and saves visual inspection outputs.
"""

import os
import glob
import cv2
from pipeline.infer import GuardRailDetector

def main():
    print("=" * 60)
    print("🛡️  GUARDRAILNET AI HIGHWAY VISION DEMO")
    print("=" * 60)

    detector = GuardRailDetector(weights_path="checkpoints/guardnet_best.pt")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    samples = sorted(glob.glob("data/sample_media/*.jpg"))
    if not samples:
        print("No sample images found in data/sample_media/")
        return

    print(f"Found {len(samples)} sample scenarios. Running detection...\n")

    for img_path in samples:
        base_name = os.path.basename(img_path)
        print(f"🔍 Analyzing: {base_name}...")
        results = detector.predict(img_path, conf_threshold=0.35)

        stats = results["stats"]
        print(f"   ↳ Barriers Detected: {stats['total_barriers']}")
        print(f"   ↳ Damage Alerts:     {stats['damaged_count']}")
        print(f"   ↳ Safety Status:     {stats['collision_risk']}")
        print(f"   ↳ Latency:           {results['latency_ms']} ms")

        out_path = os.path.join(output_dir, f"detected_{base_name}")
        cv2.imwrite(out_path, results["annotated_image"])
        print(f"   ↳ Saved overlay to:  {out_path}\n")

    print("=" * 60)
    print(f"✨ Demo complete! Annotated images saved in '{output_dir}/'")
    print("To launch the web studio, run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
