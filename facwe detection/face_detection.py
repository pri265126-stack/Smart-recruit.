import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="face_detector.task"),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    # Count faces
    face_count = len(result.detections)

    # Face status
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

    # Show face count
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SmartRecruit Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
