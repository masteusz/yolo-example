#!/usr/bin/env python3
"""
YOLO Webcam Detection with Mac Hardware Acceleration
Runs YOLOv11 (latest) in fullscreen mode using Mac GPU acceleration via MPS
"""

import cv2
import torch
from ultralytics import YOLO


def main():
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
    model = YOLO("yolo11s.pt")  # Using nano for faster inference
    model.to(device)

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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Run YOLO inference
            results = model(frame, device=device, verbose=False)

            # Visualize results on frame
            annotated_frame = results[0].plot()

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
