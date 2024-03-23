import cv2
import mediapipe as mp

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

        def draw_limb(landmark1, landmark2, landmark3):
            # Draw circles at the landmarks
            def draw_circle(landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

            draw_circle(landmark1)
            draw_circle(landmark2)
            draw_circle(landmark3)

            # Draw lines connecting the landmarks
            def draw_line(landmark1, landmark2):
                x1 = int(landmark1.x * frame.shape[1])
                y1 = int(landmark1.y * frame.shape[0])
                x2 = int(landmark2.x * frame.shape[1])
                y2 = int(landmark2.y * frame.shape[0])
                cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            draw_line(landmark1, landmark2)
            draw_line(landmark2, landmark3)

        # Right upper limb
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        draw_limb(right_shoulder, right_elbow, right_wrist)

        # Left upper limb
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        draw_limb(left_shoulder, left_elbow, left_wrist)

    # Display the annotated image
    cv2.imshow('Upper Limb Motion Detection', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
