import cv2
import multiprocessing
import mediapipe as mp
import numpy as np
from datetime import datetime
from scipy.stats import zscore
import paho.mqtt.client as mqtt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to store accelerometer and gyroscope data for both devices
data_device_1 = {
    'timestamps_acc': [],
    'accelerometer_data': {'x': [], 'y': [], 'z': []},
    'timestamps_gyro': [],
    'gyroscope_data': {'x': [], 'y': [], 'z': []}
}

data_device_2 = {
    'timestamps_acc': [],
    'accelerometer_data': {'x': [], 'y': [], 'z': []},
    'timestamps_gyro': [],
    'gyroscope_data': {'x': [], 'y': [], 'z': []}
}

# Initialize a rolling window for both devices (accelerometer and gyroscope)
window_size = 12
device_windows = {
    'device_1': {
        'acc_x_window': np.zeros(window_size),
        'acc_y_window': np.zeros(window_size),
        'acc_z_window': np.zeros(window_size),
        'gyro_x_window': np.zeros(window_size),
        'gyro_y_window': np.zeros(window_size),
        'gyro_z_window': np.zeros(window_size)
    },
    'device_2': {
        'acc_x_window': np.zeros(window_size),
        'acc_y_window': np.zeros(window_size),
        'acc_z_window': np.zeros(window_size),
        'gyro_x_window': np.zeros(window_size),
        'gyro_y_window': np.zeros(window_size),
        'gyro_z_window': np.zeros(window_size)
    }
}
window_index = 0

spike_threshold_d1_acc = 3.135
spike_threshold_d1_gyro = 3.135
spike_threshold_d2_acc = 3.135
spike_threshold_d2_gyro = 3.135

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
def check_for_sudden_movement(new_angle, threshold=1.8):
    global angle_measurements, window_size
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


def detectAccSpikes(device_id, curr_acc_x, curr_acc_y, curr_acc_z):
    global spike_detected_in_acc, window_index
    # Update the rolling window with the current values
    device_windows[device_id]['acc_x_window'][window_index] = curr_acc_x
    device_windows[device_id]['acc_y_window'][window_index] = curr_acc_y
    device_windows[device_id]['acc_z_window'][window_index] = curr_acc_z
    window_index = (window_index + 1) % window_size
    # Calculate z-scores for each axis
    z_scores_x = zscore(device_windows[device_id]['acc_x_window'])
    z_scores_y = zscore(device_windows[device_id]['acc_y_window'])
    z_scores_z = zscore(device_windows[device_id]['acc_z_window'])
    # Check if any z-score exceeds the threshold
    if device_id == 'device_1':
        spike_threshold = spike_threshold_d1_acc
    else:
        spike_threshold = spike_threshold_d2_acc
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        print(f"Spikes detected in Accel for {device_id}!")
        publish_message()
        


def detectGyroSpikes(device_id, curr_gyro_x, curr_gyro_y, curr_gyro_z):
    global spike_detected_in_gyro, window_index
    # Update the rolling window with the current values
    device_windows[device_id]['gyro_x_window'][window_index] = curr_gyro_x
    device_windows[device_id]['gyro_y_window'][window_index] = curr_gyro_y
    device_windows[device_id]['gyro_z_window'][window_index] = curr_gyro_z
    window_index = (window_index + 1) % window_size
    # Calculate z-scores for each axis
    z_scores_x = zscore(device_windows[device_id]['gyro_x_window'])
    z_scores_y = zscore(device_windows[device_id]['gyro_y_window'])
    z_scores_z = zscore(device_windows[device_id]['gyro_z_window'])
    # Check if any z-score exceeds the threshold
    if device_id == 'device_1':
        spike_threshold = spike_threshold_d1_gyro
    else:
        spike_threshold = spike_threshold_d2_gyro
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        print(f"Spikes detected in Gyro for {device_id}!")
        publish_message()

def on_message(client, userdata, message):
    topic = message.topic
    print(f"Received message on topic '{topic}': {message.payload.decode('utf-8')}")
    data = message.payload.decode('utf-8').split(' ')  # Split by spaces
    acc_data = [float(val.split('=')[1].rstrip(',')) for val in data[1:4]]  # Extract accelerometer data
    gyro_data = [float(val.split('=')[1].rstrip(',')) for val in data[5:8]]  # Extract gyroscope data
    
    if topic == 'imu_data_1':
        detectAccSpikes('device_1', acc_data[0], acc_data[1], acc_data[2])
        detectGyroSpikes('device_1', gyro_data[0], gyro_data[1], gyro_data[2])
        
        # Update data for plotting (device 1)
        data_device_1['timestamps_acc'].append(len(data_device_1['timestamps_acc']))
        data_device_1['accelerometer_data']['x'].append(acc_data[0])
        data_device_1['accelerometer_data']['y'].append(acc_data[1])
        data_device_1['accelerometer_data']['z'].append(acc_data[2])

        data_device_1['timestamps_gyro'].append(len(data_device_1['timestamps_gyro']))
        data_device_1['gyroscope_data']['x'].append(gyro_data[0])
        data_device_1['gyroscope_data']['y'].append(gyro_data[1])
        data_device_1['gyroscope_data']['z'].append(gyro_data[2])
        
    elif topic == 'imu_data_2':
        detectAccSpikes('device_2', acc_data[0], acc_data[1], acc_data[2])
        detectGyroSpikes('device_2', gyro_data[0], gyro_data[1], gyro_data[2])
        
        # Update data for plotting (device 2)
        data_device_2['timestamps_acc'].append(len(data_device_2['timestamps_acc']))
        data_device_2['accelerometer_data']['x'].append(acc_data[0])
        data_device_2['accelerometer_data']['y'].append(acc_data[1])
        data_device_2['accelerometer_data']['z'].append(acc_data[2])

        data_device_2['timestamps_gyro'].append(len(data_device_2['timestamps_gyro']))
        data_device_2['gyroscope_data']['x'].append(gyro_data[0])
        data_device_2['gyroscope_data']['y'].append(gyro_data[1])
        data_device_2['gyroscope_data']['z'].append(gyro_data[2])

# Create a figure with 4 subplots (2 for each device's accelerometer and gyroscope data)
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
fig.tight_layout(pad=4.0)

# Set titles, labels, and grid for each subplot
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Value')
        axs[i, j].grid(True)
axs[0, 0].set_title('Device 1 - Accelerometer Data')
axs[0, 1].set_title('Device 1 - Gyroscope Data')
axs[1, 0].set_title('Device 2 - Accelerometer Data')
axs[1, 1].set_title('Device 2 - Gyroscope Data')

# Initialize line objects for each subplot (device 1)
lines_device1_acc = [axs[0, 0].plot([], [], label=f'{axis}')[0] for axis in ['x', 'y', 'z']]
lines_device1_gyro = [axs[0, 1].plot([], [], label=f'{axis}')[0] for axis in ['x', 'y', 'z']]

# Initialize line objects for each subplot (device 2)
lines_device2_acc = [axs[1, 0].plot([], [], label=f'{axis}')[0] for axis in ['x', 'y', 'z']]
lines_device2_gyro = [axs[1, 1].plot([], [], label=f'{axis}')[0] for axis in ['x', 'y', 'z']]

def update_plot(frame):
    # Update line data for device 1 accelerometer and gyroscope
    for i, axis in enumerate(['x', 'y', 'z']):
        lines_device1_acc[i].set_data(data_device_1['timestamps_acc'], data_device_1['accelerometer_data'][axis])
        lines_device1_gyro[i].set_data(data_device_1['timestamps_gyro'], data_device_1['gyroscope_data'][axis])
    
    # Update line data for device 2 accelerometer and gyroscope     
    for i, axis in enumerate(['x', 'y', 'z']):
        lines_device2_acc[i].set_data(data_device_2['timestamps_acc'], data_device_2['accelerometer_data'][axis])
        lines_device2_gyro[i].set_data(data_device_2['timestamps_gyro'], data_device_2['gyroscope_data'][axis])
    
    # Rescale and update the x-axis limits for a scrolling effect
    for row in axs:
        for ax in row:
            ax.relim()
            ax.autoscale_view(True, True, True)
            current_len = max(len(data_device_1['timestamps_acc']), len(data_device_2['timestamps_acc']))
            ax.set_xlim(left=max(0, current_len - 40), right=current_len)
    
    # Return all line objects
    return lines_device1_acc + lines_device1_gyro + lines_device2_acc + lines_device2_gyro

def video_feed():
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
                sudden_movement_message = ""
                if check_for_sudden_movement(elbow_angle):
                    sm_elb = True
                    publish_message()
                    count_elb = 0
                if check_for_sudden_movement(shoulder_angle):
                    sm_shou = True
                    publish_message()
                    count_shou = 0
                
                if sm_elb:
                    if count_elb < 20:
                        sudden_movement_message += "Elbow: Sudden movement detected! "
                        count_elb += 1
                    else:
                        count_elb = 0
                        sm_elb = False
                
                if sm_shou:
                    if count_shou < 20:
                        sudden_movement_message += "Shoulder: Sudden movement detected! "
                        count_shou += 1
                    else:
                        count_shou = 0
                        sm_shou = False

                if sudden_movement_message:
                    # Display message on both terminal and the camera feed
                    print(sudden_movement_message)
                    cv2.putText(image, sudden_movement_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

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
client = mqtt.Client("Subscriber")

def plot_data():
    global client
    ani = FuncAnimation(fig, update_plot, blit=True, interval=100)
    client.on_message = on_message
    client.connect("localhost", 1883)
    client.subscribe("imu_data_1")
    client.subscribe("imu_data_2")
    client.loop_start()
    plt.show()

def publish_message():
    global client
    client.publish("flagged_results", f"Flag raised at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    
    # Create separate processes for the video feed and plotting
    video_process = multiprocessing.Process(target=video_feed)
    plot_process = multiprocessing.Process(target=plot_data)

    # Start both processes
    video_process.start()
    plot_process.start()

    # Wait for both processes to complete
    video_process.join()
    plot_process.join()

