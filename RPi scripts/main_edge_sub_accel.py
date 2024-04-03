import paho.mqtt.client as mqtt
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to store previous accelerometer values
prev_acc_x = 0
prev_acc_y = 0
prev_acc_z = 0

# Initialize a rolling window for each axis
window_size = 10
acc_x_window = np.zeros(window_size)
acc_y_window = np.zeros(window_size)
acc_z_window = np.zeros(window_size)
window_index = 0

# Initialize empty lists to store data for plotting
timestamps = []
accelerometer_data = {'x': [], 'y': [], 'z': []}

def detect_spikes(curr_acc_x, curr_acc_y, curr_acc_z):
    global window_index
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
    spike_threshold = 2.755
    if np.any(np.abs(z_scores_x) > spike_threshold) or np.any(np.abs(z_scores_y) > spike_threshold) or np.any(np.abs(z_scores_z) > spike_threshold):
        print("Spikes detected!")

def on_message(client, userdata, message):
    print(f"Received message on topic '{message.topic}': {message.payload.decode('utf-8')}")
    data = message.payload.decode('utf-8').split(' ')  # Split by spaces
    acc_data = [float(val.split('=')[1].rstrip(',')) for val in data[1:4]]  # Extract accelerometer data and remove trailing commas
    gyro_data = [float(val.split('=')[1].rstrip(',')) for val in data[5:8]]  # Extract gyroscope data
    detect_spikes(acc_data[0], acc_data[1], acc_data[2])
    
    # Update data for plotting
    timestamps.append(len(timestamps))
    accelerometer_data['x'].append(acc_data[0])
    accelerometer_data['y'].append(acc_data[1])
    accelerometer_data['z'].append(acc_data[2])

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Acceleration')
ax.set_title('Real-time Accelerometer Data')
ax.grid(True)

# Initialize empty line objects for each axis
line_x, = ax.plot([], [], label='Acc X')
line_y, = ax.plot([], [], label='Acc Y')
line_z, = ax.plot([], [], label='Acc Z')

# Function to update the plot
def update_plot(frame):
    line_x.set_data(timestamps, accelerometer_data['x'])
    line_y.set_data(timestamps, accelerometer_data['y'])
    line_z.set_data(timestamps, accelerometer_data['z'])
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
