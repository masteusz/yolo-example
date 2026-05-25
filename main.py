#!/usr/bin/env python3
"""
YOLO Webcam Detection with Mac Hardware Acceleration
Runs YOLOv11 (latest) in fullscreen mode using Mac GPU acceleration via MPS
Features: Temporal smoothing, object tracking, and confidence filtering
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class BoxSmoother:
    """Exponential Moving Average smoothing for bounding boxes"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor (0-1, lower = smoother)
        self.boxes = {}  # track_id -> smoothed box

    def update(self, track_id, box):
        """Update and return smoothed box coordinates"""
        if track_id not in self.boxes:
            self.boxes[track_id] = np.array(box, dtype=np.float32)
        else:
            # EMA: new_value = alpha * current + (1 - alpha) * previous
            self.boxes[track_id] = (self.alpha * np.array(box) +
                                   (1 - self.alpha) * self.boxes[track_id])
        return self.boxes[track_id]

    def cleanup(self, active_ids):
        """Remove boxes for tracks that are no longer active"""
        self.boxes = {k: v for k, v in self.boxes.items() if k in active_ids}


def main():
    # Configuration
    CONFIDENCE_THRESHOLD = 0.5  # Only show detections above this confidence
    SMOOTHING_ALPHA = 0.3  # Lower = smoother (0.1-0.5 recommended)

    # Check for Mac GPU acceleration (MPS)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Mac GPU acceleration (MPS)")
    else:
        device = "cpu"
        print("MPS not available, using CPU")

    # Load YOLOv11 (newest model)
    # You can use: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium),
    # yolo11l.pt (large), yolo11x.pt (xlarge)
    print("Loading YOLOv11 model...")
    model = YOLO("yolo11s.pt")  # Using small for better accuracy
    model.to(device)

    # Initialize smoother
    smoother = BoxSmoother(alpha=SMOOTHING_ALPHA)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create fullscreen window
    window_name = "YOLO Webcam Detection"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Starting detection... Press 'q' or ESC to quit")
    print(f"Smoothing: alpha={SMOOTHING_ALPHA}, Confidence threshold: {CONFIDENCE_THRESHOLD}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Run YOLO tracking (smoother than regular detection)
            # persist=True maintains track IDs across frames
            results = model.track(
                frame,
                device=device,
                verbose=False,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml"  # Fast and accurate tracker
            )

            # Get detections with tracking IDs
            annotated_frame = frame.copy()

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                active_ids = []

                for box in boxes:
                    # Get tracking ID (if available)
                    track_id = int(box.id[0]) if box.id is not None else -1

                    if track_id != -1:
                        active_ids.append(track_id)

                        # Get box coordinates and smooth them
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        smoothed = smoother.update(track_id, [x1, y1, x2, y2])
                        x1, y1, x2, y2 = smoothed.astype(int)

                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{model.names[cls]} {conf:.2f} ID:{track_id}"

                        # Draw smoothed bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label with background
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Cleanup old tracks
                smoother.cleanup(active_ids)

            # Display FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow(window_name, annotated_frame)

            # Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
