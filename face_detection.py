
import cv2
import mediapipe as mp
import time
import csv
import os
from datetime import datetime

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="face_detector.task"),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

# Camera settings
requested_width = 1280
requested_height = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera resolution:", actual_width, "x", actual_height)

# Performance variables
total_inference_time = 0
total_frames = 0
fps = 0

fps_start_time = time.perf_counter()
fps_frame_count = 0
run_start_time = time.perf_counter()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Measure inference time
        start_time = time.perf_counter()
        result = detector.detect(mp_image)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        total_frames += 1

        average_inference_time = total_inference_time / total_frames

        # Count faces
        face_count = len(result.detections)

        if face_count == 0:
            text = "No Face Detected"
        elif face_count == 1:
            text = "1 Face Detected"
        else:
            text = f"{face_count} Faces Detected"

        # Draw face boxes
        if result.detections:
            for detection in result.detections:
                box = detection.bounding_box

                x = box.origin_x
                y = box.origin_y
                w = box.width
                h = box.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        # Calculate FPS
        fps_frame_count += 1
        elapsed = time.perf_counter() - fps_start_time

        if elapsed >= 1:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.perf_counter()

        # Display information
        cv2.putText(
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Resolution: {frame_width}x{frame_height}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Inference: {inference_time:.2f} ms",
            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Average: {average_inference_time:.2f} ms",
            (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("SmartRecruit Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    run_elapsed_time = time.perf_counter() - run_start_time

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if total_frames > 0:
        average_inference_time = total_inference_time / total_frames
        overall_fps = total_frames / run_elapsed_time

        # Save results beside this Python file
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "performance_results.csv"
        )

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Date and Time",
                    "Requested Width",
                    "Requested Height",
                    "Actual Width",
                    "Actual Height",
                    "Total Frames",
                    "Average Inference (ms)",
                    "Overall FPS"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                requested_width,
                requested_height,
                actual_width,
                actual_height,
                total_frames,
                round(average_inference_time, 2),
                round(overall_fps, 2)
            ])

        print("\nPerformance results saved!")
        print("File:", csv_path)
        print("Total frames:", total_frames)
        print("Average inference:", round(average_inference_time, 2), "ms")
        print("Overall FPS:", round(overall_fps, 2))