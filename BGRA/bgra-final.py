# Existing imports 
import cv2
from datetime import datetime
import mediapipe as mp
import numpy as np
from scipy.stats import zscore

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialise counters and flags for sudden movement detection
count_elb, count_shou = 0, 0
sm_elb, sm_shou = False, False

# Video feed
cap = cv2.VideoCapture(0)

# define angle calculation
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # subtract y and x values from mid and end point minus y and x values from first to mid point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)  # convert radian to degree

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialise a list to store angle measurements for sudden movement detection
angle_measurements = []

# Define a function to check for sudden movements
def check_for_sudden_movement(new_angle, window_size=10, threshold=2):
    global angle_measurements
    if len(angle_measurements) > window_size:
        angle_measurements.pop(0)  # Remove the oldest measurement
    angle_measurements.append(new_angle)  # Add the new measurement

    if len(angle_measurements) < window_size:
        return False  # Not enough data to compute z-score

    # Calculate the rate of change of angles
    rates = [current - prev for prev, current in zip(angle_measurements[:-1], angle_measurements[1:])]

    # Compute z-scores
    z_scores = zscore(rates)

    if np.abs(z_scores[-1]) > threshold:
        return True  # Sudden movement detected
    return False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
       
        # Mirror image
        frame = cv2.flip(frame, 1)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates to calculate angle at desired joint
            # For ELbow use shoulder, elbow, wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Include Hip for Shoulder so it will be hip, shoulder, elbow
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate angles
            # Elbow
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            # Shoulder
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            # Check for sudden movements and show on camera feed   
            # Show at Camera Feed
            sudden_movement_elbow = ""
            sudden_movement_shoulder = ""
            if check_for_sudden_movement(elbow_angle):
                sm_elb = True
                count_elb = 0
            if check_for_sudden_movement(shoulder_angle):
                sm_shou = True
                count_shou = 0
            
            if sm_elb:
                if count_elb < 20:
                    sudden_movement_elbow = "Elbow: Sudden movement detected!"
                    count_elb += 1
                else:
                    count_elb = 0
                    sm_elb = False
            
            if sm_shou:
                if count_shou < 20:
                    sudden_movement_shoulder = "Shoulder: Sudden movement detected!"
                    count_shou += 1
                else:
                    count_shou = 0
                    sm_shou = False

            if sudden_movement_elbow or sudden_movement_shoulder:
                # Display message on both terminal and the camera feed
                print(sudden_movement_elbow)
                print(sudden_movement_shoulder)
                cv2.putText(image, sudden_movement_elbow, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, sudden_movement_shoulder, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
            # Visualize angles
            # Elbow
            elbow_position = tuple(np.multiply(elbow, [640, 480]).astype(int))
            cv2.putText(image, f"{elbow_angle:.3f}",
                        elbow_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # Shoulder
            shoulder_position = tuple(np.multiply(shoulder, [640, 480]).astype(int))
            cv2.putText(image, f"{shoulder_angle:.3f}",
                        shoulder_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get current time
            current_time = datetime.now().strftime("%H:%M:%S")

            # Get fram dimensions
            height, width, channels = image.shape
            # Display current time on the video feed at the bottom-left corner
            cv2.putText(image, current_time, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


            print(landmarks)
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Biomechanical Gesture Recognition and Analysis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()