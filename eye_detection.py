import numpy as np
import cv2
import mediapipe as mp
import time
import csv
import os
import joblib
 

MODEL_FILE = "best_face_activity_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

if not os.path.exists(MODEL_FILE):
    print(f"Model file not found: {MODEL_FILE}")
    exit()

if not os.path.exists(LABEL_ENCODER_FILE):
    print(f"Label encoder not found: {LABEL_ENCODER_FILE}")
    exit()

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(LABEL_ENCODER_FILE)

print("AI model loaded successfully.")




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
    landmarker.close()
    exit()




csv_file = "session_metrics.csv"



if os.path.exists(csv_file):

    try:
        with open(csv_file, "r", newline="") as file:
            rows = list(csv.reader(file))

        # First row is header
        existing_sessions = max(0, len(rows) - 1)

    except Exception:
        existing_sessions = 0

else:
    existing_sessions = 0

session_id = f"S{existing_sessions + 1:03d}"

print(f"Session ID: {session_id}")



total_frames = 0

# Face presence
face_frames = 0

# Maximum faces
max_faces_detected = 0

# Face absence
face_absence_start = None
longest_face_absence = 0.0

# Multiple face duration
multiple_face_start = None
multiple_face_duration = 0.0

# Looking away events
looking_away_events = 0
looking_away_start = None
event_counted = False

# Gaze frames
center_frames = 0
left_frames = 0
right_frames = 0



while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    total_frames += 1

    

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    

    result = landmarker.detect(mp_image)

    face_count = len(result.face_landmarks)


    

    if face_count > 0:

        face_frames += 1

        # Face returned after absence
        if face_absence_start is not None:

            absence_time = (
                time.time() - face_absence_start
            )

            if absence_time > longest_face_absence:
                longest_face_absence = absence_time

            face_absence_start = None

    else:

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

        # First detected face
        face_landmarks = result.face_landmarks[0]

        # Important landmarks
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        nose = face_landmarks[1]

        # Eye center
        eye_center = (
            left_eye.x + right_eye.x
        ) / 2

        # Difference between eye center and nose
        difference = eye_center - nose.x

        

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


    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




if face_absence_start is not None:

    absence_time = (
        time.time() - face_absence_start
    )

    if absence_time > longest_face_absence:
        longest_face_absence = absence_time




if multiple_face_start is not None:

    duration = (
        time.time() - multiple_face_start
    )

    multiple_face_duration += duration




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




print("\n")
print("=" * 50)
print("       MODEL INPUT FEATURES")
print("=" * 50)

print(
    f"Face Presence        : "
    f"{face_presence_percentage:.2f}%"
)

print(
    f"Maximum Faces        : "
    f"{max_faces_detected}"
)

print(
    f"Longest Face Absence : "
    f"{longest_face_absence:.2f} sec"
)

print(
    f"Multiple Face Time   : "
    f"{multiple_face_duration:.2f} sec"
)

print(
    f"Looking Away Events  : "
    f"{looking_away_events}"
)

print(
    f"Looking Center       : "
    f"{looking_center_percentage:.2f}%"
)

print(
    f"Looking Left         : "
    f"{looking_left_percentage:.2f}%"
)

print(
    f"Looking Right        : "
    f"{looking_right_percentage:.2f}%"
)

print("=" * 50)



features = [[

    face_presence_percentage,
    max_faces_detected,
    longest_face_absence,
    multiple_face_duration,
    looking_away_events,
    looking_center_percentage,
    looking_left_percentage,
    looking_right_percentage

]]




prediction_encoded = model.predict(
    features
)

prediction = label_encoder.inverse_transform(
    prediction_encoded.astype(int)
)[0]



confidence = None

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        features
    )

    confidence = float(
        np.max(probabilities)
    ) * 100




flag = False

flag_reason = (
    "No visual activity flagged."
)




if max_faces_detected > 1:

    flag = True

    flag_reason = (
        "Multiple faces detected during session."
    )




elif longest_face_absence >= 3:

    flag = True

    flag_reason = (
        "Face was absent for 3 or more seconds."
    )



elif (
    prediction == "LOOKING_AWAY"
    and looking_away_events >= 2
):

    flag = True

    flag_reason = (
        "Repeated looking-away activity detected."
    )




elif prediction == "LOOKING_LEFT":

    if looking_left_percentage >= 60:

        flag = True

        flag_reason = (
            "Prolonged left-looking activity detected."
        )

    else:

        flag = False

        flag_reason = (
            "Short or limited left-looking activity."
        )




elif prediction == "LOOKING_RIGHT":

    if looking_right_percentage >= 60:

        flag = True

        flag_reason = (
            "Prolonged right-looking activity detected."
        )

    else:

        flag = False

        flag_reason = (
            "Short or limited right-looking activity."
        )



elif prediction == "NORMAL":

    flag = False

    flag_reason = (
        "Normal face activity detected."
    )




elif prediction == "NO_FACE":

    flag = True

    flag_reason = (
        "Face was absent for a prolonged duration."
    )



else:

    flag = False

    flag_reason = (
        "No significant visual activity detected."
    )




csv_file = "session_metrics.csv"

file_exists = os.path.exists(csv_file)



header = [
    "Session ID",
    "Face Presence (%)",
    "Maximum Faces Detected",
    "Longest Face Absence (sec)",
    "Multiple Face Duration (sec)",
    "Looking Away Events",
    "Looking Center (%)",
    "Looking Left (%)",
    "Looking Right (%)",
    "Predicted Activity",
    "Model Confidence (%)",
    "Visual Activity Flag",
    "Flag Reason"
]


with open(
    csv_file,
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    # Write header if file is new/empty
    if (
        not file_exists
        or os.path.getsize(csv_file) == 0
    ):

        writer.writerow(header)


    

    writer.writerow([

        session_id,

        round(
            face_presence_percentage,
            2
        ),

        max_faces_detected,

        round(
            longest_face_absence,
            2
        ),

        round(
            multiple_face_duration,
            2
        ),

        looking_away_events,

        round(
            looking_center_percentage,
            2
        ),

        round(
            looking_left_percentage,
            2
        ),

        round(
            looking_right_percentage,
            2
        ),

        prediction,

        round(
            confidence,
            2
        ) if confidence is not None else "",

        "YES" if flag else "NO",

        flag_reason
    ])



cap.release()

cv2.destroyAllWindows()

landmarker.close()



print("\n")
print("=" * 50)
print("       SESSION COMPLETED")
print("=" * 50)

print(
    f"\nSession ID: {session_id}"
)

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


print("\n")
print("-" * 50)
print("       AI ACTIVITY ANALYSIS")
print("-" * 50)

print(
    f"\nPredicted Activity: "
    f"{prediction}"
)

if confidence is not None:

    print(
        f"Model Confidence: "
        f"{confidence:.2f}%"
    )

print(
    f"\nVisual Activity Flag: "
    f"{'YES' if flag else 'NO'}"
)

print(
    f"Reason: {flag_reason}"
)

print("\nData saved to:")
print("session_metrics.csv")

print("\n")