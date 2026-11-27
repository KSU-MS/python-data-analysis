import serial
import matplotlib.pyplot as plt
from collections import deque

# Open the serial port
ser = serial.Serial('COM10', 9600)  # Change 'COM1' to your serial port and 9600 to your baud rate

# Initialize deque to store the data with a maximum length of 100000
data = deque(maxlen=100000)

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Serial Data Plot')

# Read and plot the data
try:
    while True:
        line_data = ser.readline().decode().strip()  # Read one line of data
        try:
            value = float(line_data)  # Convert the data to float
            data.append(value)  # Append the value to the data deque

            # Update the plot
            line.set_xdata(range(len(data)))
            line.set_ydata(data)
            ax.relim()
            ax.autoscale_view()

            # Pause to allow the plot to update
            plt.pause(0.0001)
        except ValueError:
            pass  # Ignore any lines that can't be converted to float
except KeyboardInterrupt:
    ser.close()
    plt.close()
