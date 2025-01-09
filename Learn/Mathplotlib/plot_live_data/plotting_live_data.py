import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time

# Set the plot style
plt.style.use("seaborn-v0_8-paper")

# Path to the CSV file
data_path = '../data/data.csv'

# Create a figure and axis object
fig, ax = plt.subplots()

def animate(i):
    # Clear the current axes
    ax.cla()

    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return  # Skip this frame

    try:
        # Read the data from the CSV file
        data = pd.read_csv(data_path)

        # Ensure the data is not empty
        if data.empty:
            print("Data is empty.")
            return  # Skip this frame

        # Ensure required columns are present
        required_columns = {'x_value', 'total_1', 'total_2'}
        if not required_columns.issubset(data.columns):
            print(f"Missing columns in data. Required: {required_columns}")
            return  # Skip this frame

        # Extract data for plotting
        x = data['x_value']
        y1 = data['total_1']
        y2 = data['total_2']

        # Plot the data
        ax.plot(x, y1, label='Channel_1')
        ax.plot(x, y2, label='Channel_2')

        # Add legend
        ax.legend()

        # Add labels and title
        ax.set_xlabel('X Value')
        ax.set_ylabel('Total')
        ax.set_title('Live Data Plot')

        # Optional: Adjust plot limits or formatting
        ax.relim()
        ax.autoscale_view()

    except pd.errors.EmptyDataError:
        print('No columns to parse from file.')
        return  # Skip this frame
    except Exception as e:
        print(f"An error occurred: {e}")
        return  # Skip this frame


# Create the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=200,  # Number of frames to save
    interval=1000,
    cache_frame_data=True,
)

# Display the plot
plt.show()
