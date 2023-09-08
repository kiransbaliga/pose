import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

bicep_curl = True
deadlift = False

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

cap = cv2.VideoCapture(0)

counter_curl = 0
stage_curl = None
counter_deadlift = 0
stage_deadlift = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks for Bicep Curl and Deadlift
        if bicep_curl:
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for bicep curl
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle for bicep curl
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle for bicep curl
                cv2.putText(image, f'Bicep Curl Angle: {angle:.2f} degrees',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic for bicep curl
                if angle > 160:
                    stage_curl = "down"
                if angle < 30 and stage_curl == 'down':
                    stage_curl = "up"
                    counter_curl += 1
                    print(f'Bicep Curl Count: {counter_curl}')

            except:
                bicep_curl = False
                pass

        if deadlift:
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for deadlift
                hips = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knees = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankles = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle for deadlift
                angle = calculate_angle(hips, knees, ankles)

                # Visualize angle for deadlift
                cv2.putText(image, f'Deadlift Angle: {angle:.2f} degrees',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Deadlift counter logic
                if angle > 120:
                    stage_deadlift = "down"
                if angle < 60 and stage_deadlift == 'down':
                    stage_deadlift = "up"
                    counter_deadlift += 1
                    print(f'Deadlift Count: {counter_deadlift}')

            except:
                deadlift = False
                pass

        # Render curl and deadlift counters
        cv2.putText(image, 'Exercise: Bicep Curl', (10, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'Exercise: Deadlift', (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Rep data for bicep curl
        cv2.putText(image, 'Bicep Curl Reps', (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_curl),
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep data for deadlift
        cv2.putText(image, 'Deadlift Reps', (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_deadlift),
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data for bicep curl
        cv2.putText(image, 'Bicep Curl Stage', (120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_curl,
                    (110, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data for deadlift
        cv2.putText(image, 'Deadlift Stage', (120, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_deadlift,
                    (110, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
