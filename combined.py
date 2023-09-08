# Has errrors


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define landmarks for both exercises
exercise_landmarks = {
    "Bicep Curl": {
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    },
    "Leg Raise": {
        "hip": mp_pose.PoseLandmark.LEFT_HIP,
        "knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    },
}

angle_thresholds = {
    "Bicep Curl": {"up": 30, "down": 160},
    "Leg Raise": {"up": 60, "down": 10},
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def classify_exercise(landmarks):
    # Calculate the angles for both exercises
    bicep_curl_angle = calculate_angle(
        [landmarks["Bicep Curl"]["shoulder"].x, landmarks["Bicep Curl"]["shoulder"].y],
        [landmarks["Bicep Curl"]["elbow"].x, landmarks["Bicep Curl"]["elbow"].y],
        [landmarks["Bicep Curl"]["wrist"].x, landmarks["Bicep Curl"]["wrist"].y]
    )

    leg_raise_angle = calculate_angle(
        [landmarks["Leg Raise"]["hip"].x, landmarks["Leg Raise"]["hip"].y],
        [landmarks["Leg Raise"]["knee"].x, landmarks["Leg Raise"]["knee"].y],
        [landmarks["Leg Raise"]["ankle"].x, landmarks["Leg Raise"]["ankle"].y]
    )

    # Determine the exercise based on angles
    if bicep_curl_angle > angle_thresholds["Bicep Curl"]["down"]:
        return "Bicep Curl"
    elif leg_raise_angle > angle_thresholds["Leg Raise"]["down"]:
        return "Leg Raise"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    exercise_type = None
    counter = 0
    stage = None

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
        
        # Extract landmarks for both exercises
        landmarks = {}
        for exercise, points in exercise_landmarks.items():
            landmarks[exercise] = {}
            for point, landmark in points.items():
                landmarks[exercise][point] = results.pose_landmarks.landmark[landmark.value]

        # Dynamically classify the exercise
        exercise_type = classify_exercise(landmarks)
        
        # Visualize exercise type
        cv2.putText(image, f"Exercise: {exercise_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if exercise_type != "Unknown":
            # Calculate and count reps for the detected exercise
            try:
                points = exercise_landmarks[exercise_type]
                a = [landmarks[exercise_type][points["shoulder"]].x, landmarks[exercise_type][points["shoulder"]].y]
                b = [landmarks[exercise_type][points["elbow"]].x, landmarks[exercise_type][points["elbow"]].y]
                c = [landmarks[exercise_type][points["wrist"]].x, landmarks[exercise_type][points["wrist"]].y]

                # Calculate angle
                angle = calculate_angle(a, b, c)

                # Visualize angle
                cv2.putText(image, str(angle), 
                           tuple(np.multiply(b, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic"
                if angle > angle_thresholds[exercise_type]["down"]:
                    stage = "down"
                if angle < angle_thresholds[exercise_type]["up"] and stage =='down':
                    stage = "up"
                    counter += 1
                    print(f"Exercise: {exercise_type}, Reps: {counter}")

            except:
                pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
