import paho.mqtt.client as mqtt
import pandas as pd
import matplotlib.pyplot as plt
import time

# Initialize an empty DataFrame to store the sensor data
columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
sensor_data = pd.DataFrame(columns=columns)

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
lines = [ax.plot([], [])[0] for _ in range(6)]  # Create six lines, one for each sensor axis
ax.set_xlim(0, 100)  # Set the x-axis limits (adjust as needed)
ax.set_ylim(-10, 10)  # Set the y-axis limits (adjust as needed)

def on_message(client, userdata, message):
    print(f"Received message on topic '{message.topic}': {message.payload.decode('utf-8')}")
    data = message.payload.decode('utf-8').split(' ')  # Split by spaces
    acc_data = [float(val.split('=')[1].rstrip(',')) for val in data[1:4]]  # Extract accelerometer data and remove trailing commas
    gyro_data = [float(val.split('=')[1].rstrip(',')) for val in data[5:8]]  # Extract gyroscope data
    detect_anomalies(acc_data, gyro_data)

def detect_anomalies(acc_data, gyro_data):
    global sensor_data
    # Add the new data to the DataFrame
    new_data = acc_data + gyro_data
    sensor_data = sensor_data._append(pd.Series(new_data, index=columns), ignore_index=True)
    # Update the plot
    update_plot()

def update_plot():
    global sensor_data, lines
    for i in range(6):  # Update each line with the new data
        lines[i].set_data(range(len(sensor_data)), sensor_data.iloc[:, i])
    plt.draw()
    plt.pause(0.01)  # Pause to update the plot

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("imu_data")
client.loop_start()

time.sleep(3)  # Add a 3-second delay before starting the plot

# Keep the script running and plot the data
try:
    while True:
        if not sensor_data.empty:  # Check if the DataFrame is not empty
            update_plot()  # Update the plot
        plt.pause(1)  # Pause for 1 second before updating the plot again
except KeyboardInterrupt:
    client.loop_stop()
    plt.close()  # Close the plot window
