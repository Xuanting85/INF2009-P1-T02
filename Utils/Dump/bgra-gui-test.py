import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import zscore

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Define function to check for sudden movements
def check_for_sudden_movement(new_angle, window_size=10, threshold=2):
    global angle_measurements
    if len(angle_measurements) > window_size:
        angle_measurements.pop(0)
    angle_measurements.append(new_angle)
    if len(angle_measurements) < window_size:
        return False
    rates = [current - prev for prev, current in zip(angle_measurements[:-1], angle_measurements[1:])]
    z_scores = zscore(rates)
    if np.abs(z_scores[-1]) > threshold:
        return True
    return False

# Initialize a list to store angle measurements for sudden movement detection
angle_measurements = []

# Tkinter GUI class
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Biomechanical Gesture Recognition and Analysis')
        self.geometry('800x600')

        # Style configuration
        style = ttk.Style(self)
        style.configure('TLabel', font=('Helvetica', 12), foreground='red')

        # Camera feed label
        self.image_label = ttk.Label(self)
        self.image_label.pack(pady=20)

        # Alert label
        self.alert_label = ttk.Label(self, text='', style='TLabel')
        self.alert_label.pack()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        
        # Pose detection setup
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Update frame in the GUI
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image, alert_message = self.process_frame(frame)
            self.display_image(image)
            self.alert_label.config(text=alert_message)
        self.after(30, self.update_frame)

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sudden_movement_message = ""
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            if check_for_sudden_movement(elbow_angle):
                sudden_movement_message += "Elbow: Sudden movement detected! "
            if check_for_sudden_movement(shoulder_angle):
                sudden_movement_message += "Shoulder: Sudden movement detected! "

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, sudden_movement_message

    def display_image(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == '__main__':
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
