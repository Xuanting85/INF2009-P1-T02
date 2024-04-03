import paho.mqtt.client as mqtt
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Global variables to store accelerometer data
timestamps_acc = []
accelerometer_data = {'x': [], 'y': [], 'z': []}
spike_detected_in_acc = False

# Global variables to store gyroscope data
timestamps_gyro = []
gyroscope_data = {'x': [], 'y': [], 'z': []}
spike_detected_in_gyro = False

# Initialize a rolling window for accelerometer
window_size = 15
acc_x_window = np.zeros(window_size)
acc_y_window = np.zeros(window_size)
acc_z_window = np.zeros(window_size)
gyro_x_window = np.zeros(window_size)
gyro_y_window = np.zeros(window_size)
gyro_z_window = np.zeros(window_size)
window_index = 0

def detectAccSpikes(curr_acc_x, curr_acc_y, curr_acc_z):
    global spike_detected_in_acc, window_index
    # Update the rolling window with the current values
    acc_x_window[window_index] = curr_acc_x
    acc_y_window[window_index] = curr_acc_y
    acc_z_window[window_index] = curr_acc_z
    window_index = (window_index + 1) % window_size
    # Calculate z-scores for each axis
    z_scores_x = zscore(acc_x_window)
    z_scores_y = zscore(acc_y_window)
    z_scores_z = zscore(acc_z_window)
    # Check if any z-score exceeds a threshold (e.g., 3)
    spike_threshold = 2.775
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        spike_detected_in_acc = True
        print("Spikes detected in Accel!")
    else:
        spike_detected_in_acc = False

def detectGyroSpikes(curr_gyro_x, curr_gyro_y, curr_gyro_z):
    global spike_detected_in_gyro, window_index
    # Update the rolling window with the current values
    gyro_x_window[window_index] = curr_gyro_x
    gyro_y_window[window_index] = curr_gyro_y
    gyro_z_window[window_index] = curr_gyro_z
    window_index = (window_index + 1) % window_size
    # Calculate z-scores for each axis
    z_scores_x = zscore(gyro_x_window)
    z_scores_y = zscore(gyro_y_window)
    z_scores_z = zscore(gyro_z_window)
    # Check if any z-score exceeds a threshold (e.g., 3)
    spike_threshold = 2.8
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        spike_detected_in_gyro = True
        print("Spikes detected in Gyro!")
    else:
        spike_detected_in_gyro = False

def on_message(client, userdata, message):
    print(f"Received message on topic '{message.topic}': {message.payload.decode('utf-8')}")
    data = message.payload.decode('utf-8').split(' ')  # Split by spaces
    acc_data = [float(val.split('=')[1].rstrip(',')) for val in data[1:4]]  # Extract accelerometer data and remove trailing commas
    gyro_data = [float(val.split('=')[1].rstrip(',')) for val in data[5:8]]  # Extract gyroscope data
    detectAccSpikes(acc_data[0], acc_data[1], acc_data[2])
    detectGyroSpikes(gyro_data[0], gyro_data[1], gyro_data[2])
    
    # Update data for plotting
    timestamps_acc.append(len(timestamps_acc))
    accelerometer_data['x'].append(acc_data[0])
    accelerometer_data['y'].append(acc_data[1])
    accelerometer_data['z'].append(acc_data[2])

    timestamps_gyro.append(len(timestamps_gyro))
    gyroscope_data['x'].append(gyro_data[0])
    gyroscope_data['y'].append(gyro_data[1])
    gyroscope_data['z'].append(gyro_data[2])

# Create a figure and axis for the plots
fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(12, 10))
fig.tight_layout(pad=3.0)
ax_acc.set_xlabel('Time')
ax_acc.set_ylabel('Acceleration')
ax_acc.set_title('Real-time Accelerometer Data')
ax_acc.grid(True)
ax_gyro.set_xlabel('Time')
ax_gyro.set_ylabel('Gyroscope')
ax_gyro.set_title('Real-time Gyroscope Data')
ax_gyro.grid(True)

# Initialize empty line objects for accelerometer plot
line_acc_x, = ax_acc.plot([], [], label='Acc X')
line_acc_y, = ax_acc.plot([], [], label='Acc Y')
line_acc_z, = ax_acc.plot([], [], label='Acc Z')
# Initialize empty line objects for gyroscope plot
line_gyro_x, = ax_gyro.plot([], [], label='Gyro X')
line_gyro_y, = ax_gyro.plot([], [], label='Gyro Y')
line_gyro_z, = ax_gyro.plot([], [], label='Gyro Z')

def update_plot_acc(frame):
    global spike_detected_in_acc
    line_acc_x.set_data(timestamps_acc, accelerometer_data['x'])
    line_acc_y.set_data(timestamps_acc, accelerometer_data['y'])
    line_acc_z.set_data(timestamps_acc, accelerometer_data['z'])
    
    ax_acc.relim()  
    ax_acc.autoscale_view(scalex=False, scaley=True)  # Disable autoscaling for x-axis

    # Update x-axis limits to create scrolling effect
    ax_acc.set_xlim(left=max(0, len(timestamps_acc) - 300), right=len(timestamps_acc))

    return line_acc_x, line_acc_y, line_acc_z

def update_plot_gyro(frame):
    global spike_detected_in_gyro
    line_gyro_x.set_data(timestamps_gyro, gyroscope_data['x'])
    line_gyro_y.set_data(timestamps_gyro, gyroscope_data['y'])
    line_gyro_z.set_data(timestamps_gyro, gyroscope_data['z'])
    
    ax_gyro.relim()  
    ax_gyro.autoscale_view(scalex=False, scaley=True)  # Disable autoscaling for x-axis

    # Update x-axis limits to create scrolling effect
    ax_gyro.set_xlim(left=max(0, len(timestamps_gyro) - 300), right=len(timestamps_gyro))

    return line_gyro_x, line_gyro_y, line_gyro_z


# Create animations to update the plots
ani_acc = FuncAnimation(fig, update_plot_acc, frames=None, blit=True, interval=100)
ani_gyro = FuncAnimation(fig, update_plot_gyro, frames=None, blit=True, interval=100)

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("imu_data")
client.loop_start()

plt.show()
