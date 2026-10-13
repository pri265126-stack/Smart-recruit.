
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

# Face filtering settings
min_confidence = 0.6
min_face_area_ratio = 0.02

# Inference time tracking
total_inference_time = 0
total_frames = 0
average_inference_time = 0
inference_time = 0

# FPS tracking
fps_start_time = time.perf_counter()
fps_frame_count = 0
fps = 0

# Overall test time
run_start_time = time.perf_counter()

# CSV file will be saved beside this Python script
csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "performance_results.csv"
)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame could not be read.")
            break

        frame_height, frame_width = frame.shape[:2]

        # Convert frame to RGB
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

        average_inference_time = (
            total_inference_time / total_frames
        )

        # Calculate live FPS
        fps_frame_count += 1
        elapsed_time = time.perf_counter() - fps_start_time

        if elapsed_time >= 1:
            fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            fps_start_time = time.perf_counter()

        # Filter faces by confidence and size
        valid_detections = []
        frame_area = frame_width * frame_height

        for detection in result.detections:
            box = detection.bounding_box

            if detection.categories:
                confidence = detection.categories[0].score
            else:
                confidence = 0

            face_area = box.width * box.height
            face_area_ratio = face_area / frame_area

            if (confidence >= min_confidence and
                    face_area_ratio >= min_face_area_ratio):
                valid_detections.append(detection)

        # Count detected faces
        face_count = len(valid_detections)

        if face_count == 0:
            text = "No Face Detected"
        elif face_count == 1:
            text = "1 Face Detected"
        else:
            text = f"{face_count} Faces Detected"

        # Draw face boxes
        for detection in valid_detections:
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

        # Display face count
        cv2.putText(
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2
        )

        # Display resolution
        resolution_text = f"Resolution: {frame_width}x{frame_height}"

        cv2.putText(
            frame, resolution_text, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

        # Display current inference time
        time_text = f"Inference Time: {inference_time:.2f} ms"

        cv2.putText(
            frame, time_text, (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        # Display average inference time
        avg_text = f"Average Time: {average_inference_time:.2f} ms"

        cv2.putText(
            frame, avg_text, (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        # Display FPS
        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(
            frame, fps_text, (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )

        cv2.imshow("SmartRecruit Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Calculate overall test performance
    run_elapsed_time = time.perf_counter() - run_start_time

    if total_frames > 0 and run_elapsed_time > 0:
        overall_fps = total_frames / run_elapsed_time
    else:
        overall_fps = 0

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Save performance results to CSV
    try:
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
                    "Average Inference Time (ms)",
                    "Overall FPS"
                ])

            if total_frames > 0:
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

        if total_frames > 0:
            print("Total frames processed:", total_frames)
            print(
                "Average inference time:",
                round(average_inference_time, 2), "ms"
            )
            print("Overall FPS:", round(overall_fps, 2))
            print("Results saved to:", csv_path)

    except OSError as error:
        print("Could not save CSV:", error)