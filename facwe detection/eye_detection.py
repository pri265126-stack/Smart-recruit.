import cv2
import mediapipe as mp
import time
import csv
import os


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=2
)

landmarker = FaceLandmarker.create_from_options(options)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()



session_id = "S001"

total_frames = 0

# Face presence
face_frames = 0

# Maximum number of faces
max_faces_detected = 0

# Face absence
face_absence_start = None
longest_face_absence = 0


# Multiple face duration
multiple_face_start = None
multiple_face_duration = 0


# Looking-away events
looking_away_events = 0
looking_away_start = None
event_counted = False


# Gaze frame counts
center_frames = 0
left_frames = 0
right_frames = 0




while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    total_frames += 1

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert image for MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    # Number of detected faces
    face_count = len(result.face_landmarks)


    
    if face_count > 0:

        face_frames += 1

        # Face has returned
        if face_absence_start is not None:

            absence_time = (
                time.time() - face_absence_start
            )

            if absence_time > longest_face_absence:
                longest_face_absence = absence_time

            face_absence_start = None

    else:

        # Start counting face absence
        if face_absence_start is None:
            face_absence_start = time.time()


    

    if face_count > max_faces_detected:

        max_faces_detected = face_count


    

    if face_count > 1:

        if multiple_face_start is None:

            multiple_face_start = time.time()

    else:

        if multiple_face_start is not None:

            duration = (
                time.time() - multiple_face_start
            )

            multiple_face_duration += duration

            multiple_face_start = None


    

    gaze = "No Face"


    

    if result.face_landmarks:

        # Use first detected face
        face_landmarks = result.face_landmarks[0]

        # Important facial landmarks
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        nose = face_landmarks[1]

        # Calculate eye center
        eye_center = (
            left_eye.x + right_eye.x
        ) / 2

        # Difference between eye center and nose
        difference = (
            eye_center - nose.x
        )


        # Determine gaze direction
        if difference < -0.015:

            gaze = "Looking Left"

            left_frames += 1

        elif difference > 0.015:

            gaze = "Looking Right"

            right_frames += 1

        else:

            gaze = "Looking Center"

            center_frames += 1


        

        if gaze == "Looking Center":

            looking_away_start = None
            event_counted = False

        else:

            if looking_away_start is None:

                looking_away_start = time.time()

            away_time = (
                time.time() - looking_away_start
            )

            # Count event after 1.5 seconds
            if (
                away_time >= 1.5
                and not event_counted
            ):

                looking_away_events += 1

                event_counted = True


    

    face_presence = (
        face_frames / total_frames
    ) * 100


    

    if face_count == 0:

        face_text = "No Face Detected"

    elif face_count == 1:

        face_text = "1 Face Detected"

    else:

        face_text = (
            f"{face_count} Faces Detected"
        )


    

    cv2.putText(
        frame,
        face_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Face Presence: {face_presence:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Gaze: {gaze}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Looking Away Events: {looking_away_events}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Max Faces: {max_faces_detected}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    

    cv2.imshow(
        "SmartRecruit Eye Detection",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break





# If face is still absent when user presses Q
if face_absence_start is not None:

    absence_time = (
        time.time() - face_absence_start
    )

    if absence_time > longest_face_absence:

        longest_face_absence = absence_time


# If multiple faces are still present when user presses Q
if multiple_face_start is not None:

    duration = (
        time.time() - multiple_face_start
    )

    multiple_face_duration += duration


# Face presence percentage
if total_frames > 0:

    face_presence_percentage = (
        face_frames / total_frames
    ) * 100

else:

    face_presence_percentage = 0



total_gaze_frames = (
    center_frames
    + left_frames
    + right_frames
)


if total_gaze_frames > 0:

    looking_center_percentage = (
        center_frames / total_gaze_frames
    ) * 100

    looking_left_percentage = (
        left_frames / total_gaze_frames
    ) * 100

    looking_right_percentage = (
        right_frames / total_gaze_frames
    ) * 100

else:

    looking_center_percentage = 0
    looking_left_percentage = 0
    looking_right_percentage = 0




csv_file = "session_metrics.csv"

file_exists = os.path.exists(csv_file)

with open(
    csv_file,
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    # Write header only for new/empty file
    if not file_exists or os.path.getsize(csv_file) == 0:

        writer.writerow([
            "Session ID",
            "Face Presence (%)",
            "Maximum Faces Detected",
            "Longest Face Absence (sec)",
            "Multiple Face Duration (sec)",
            "Looking Away Events",
            "Looking Center (%)",
            "Looking Left (%)",
            "Looking Right (%)"
        ])


    # Write session data
    writer.writerow([
        session_id,
        round(face_presence_percentage, 2),
        max_faces_detected,
        round(longest_face_absence, 2),
        round(multiple_face_duration, 2),
        looking_away_events,
        round(looking_center_percentage, 2),
        round(looking_left_percentage, 2),
        round(looking_right_percentage, 2)
    ])




cap.release()

cv2.destroyAllWindows()

landmarker.close()




print("\n===================================")
print("       SESSION COMPLETED")
print("===================================")

print(f"Session ID: {session_id}")

print(
    f"Face Presence: "
    f"{face_presence_percentage:.2f}%"
)

print(
    f"Maximum Faces Detected: "
    f"{max_faces_detected}"
)

print(
    f"Longest Face Absence: "
    f"{longest_face_absence:.2f} sec"
)

print(
    f"Multiple Face Duration: "
    f"{multiple_face_duration:.2f} sec"
)

print(
    f"Looking Away Events: "
    f"{looking_away_events}"
)

print(
    f"Looking Center: "
    f"{looking_center_percentage:.2f}%"
)

print(
    f"Looking Left: "
    f"{looking_left_percentage:.2f}%"
)

print(
    f"Looking Right: "
    f"{looking_right_percentage:.2f}%"
)

print("\nData saved to:")
print("session_metrics.csv")
