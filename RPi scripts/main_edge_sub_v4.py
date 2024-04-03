import paho.mqtt.client as mqtt
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to store previous gyroscope values
prev_gyro_x = 0
prev_gyro_y = 0
prev_gyro_z = 0

# Initialize a rolling window for each axis
window_size = 10
gyro_x_window = np.zeros(window_size)
gyro_y_window = np.zeros(window_size)
gyro_z_window = np.zeros(window_size)
window_index = 0

# Initialize empty lists to store gyroscope data for plotting
gyro_timestamps = []
gyroscope_data = {'x': [], 'y': [], 'z': []}

def detect_spikes(curr_gyro_x, curr_gyro_y, curr_gyro_z):
    global window_index
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
    spike_threshold = 2.755
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        print("Spikes detected!")

def on_message(client, userdata, message):
    print(f"Received message on topic '{message.topic}': {message.payload.decode('utf-8')}")
    data = message.payload.decode('utf-8').split(' ')  # Split by spaces
    acc_data = [float(val.split('=')[1].rstrip(',')) for val in data[1:4]]  # Extract accelerometer data and remove trailing commas
    gyro_data = [float(val.split('=')[1].rstrip(',')) for val in data[5:8]]  # Extract gyroscope data
    detect_spikes(gyro_data[0], gyro_data[1], gyro_data[2])
    
    # Update data for plotting
    gyro_timestamps.append(len(gyro_timestamps))
    gyroscope_data['x'].append(gyro_data[0])
    gyroscope_data['y'].append(gyro_data[1])
    gyroscope_data['z'].append(gyro_data[2])

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
ax.set_title('Real-time Gyroscope Data')
ax.grid(True)

# Initialize empty line objects for each axis
line_x, = ax.plot([], [], label='Gyro X')
line_y, = ax.plot([], [], label='Gyro Y')
line_z, = ax.plot([], [], label='Gyro Z')

# Function to update the plot
def update_plot(frame):
    line_x.set_data(gyro_timestamps, gyroscope_data['x'])
    line_y.set_data(gyro_timestamps, gyroscope_data['y'])
    line_z.set_data(gyro_timestamps, gyroscope_data['z'])
    ax.relim()  # Update the limits of the axes
    ax.autoscale_view()  # Autoscale the view
    return line_x, line_y, line_z

# Create an animation to update the plot
ani = FuncAnimation(fig, update_plot, frames=None, blit=True, interval=100)

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("imu_data")
client.loop_start()

plt.show()
