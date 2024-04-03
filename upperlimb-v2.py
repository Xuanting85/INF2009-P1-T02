import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to calculate jerk
def calculate_jerk(positions):
    velocity = np.gradient(positions, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    jerk = np.gradient(acceleration, axis=0)
    return np.linalg.norm(jerk, axis=1)  # Return the magnitude of the jerk

# Function to draw limb and calculate jerk
def draw_limb_and_calculate_jerk(landmark1, landmark2, landmark3, image_width, image_height):
    
    #scale landmarks to pixel values
    def scale_landmark(landmark):
        return np.array([landmark.x * image_width, landmark.y * image_height])
    scaled_landmark1 = scale_landmark(landmark1)
    scaled_landmark2 = scale_landmark(landmark2)
    scaled_landmark3 = scale_landmark(landmark3)
    
    # Draw circles at the landmarks
    def draw_circle(position):
        cv2.circle(annotated_image, tuple(position.astype(int)), 5, (0, 255, 0), -1)

    draw_circle(scaled_landmark1)
    draw_circle(scaled_landmark2)
    draw_circle(scaled_landmark3)

    # Draw lines connecting the landmarks
    def draw_line(pos1, pos2):
        cv2.line(annotated_image, tuple(pos1.astype(int)), tuple(pos2.astype(int)), (255, 0, 0), 2)

    draw_line(scaled_landmark1, scaled_landmark2)
    draw_line(scaled_landmark2, scaled_landmark3)

    # Calculate jerk for the limb
    positions = np.array([scaled_landmark1, scaled_landmark2, scaled_landmark3])
    limb_jerk = calculate_jerk(positions)

    # Apply Savitzky-Golay filter if there are enough data points
    if len(limb_jerk) >= 5:
        smoothed_jerk = savgol_filter(limb_jerk, window_length=5, polyorder=2)
        print(f"Average smoothed jerk magnitude for limb: {np.mean(smoothed_jerk)}")
    else:
        print(f"Average jerk magnitude for limb: {np.mean(limb_jerk)}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose
    results = pose.process(image)

    # Draw the pose annotations on the image
    annotated_image = frame.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get the image dimensions
        image_width, image_height = frame.shape[1], frame.shape[0]

        # Right upper limb
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        draw_limb_and_calculate_jerk(right_shoulder, right_elbow, right_wrist, image_width, image_height)

        # Left upper limb
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        draw_limb_and_calculate_jerk(left_shoulder, left_elbow, left_wrist, image_width, image_height)

    # Display the annotated image
    cv2.imshow('Upper Limb Motion Detection', annotated_image)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
