#%%
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use("seaborn-v0_8-paper") # set style to xkcd
#%%
import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn-v0_8-paper")

x_vals = []
y_vals = []

index = count()

def animate(i):
    data = pd.read_csv('data.csv')
    x = data['x_value']
    y1 = data['total1']
    y2 = data['total2']
    
    plt.cla()
    plt.plot(x_vals, y_vals)
    plt.title("Live Plotting")
    plt.tight_layout()

anim = FuncAnimation(plt.gcf(), animate, frames=100, interval=1000)

anim.save("live_plot.gif", writer="pillow")
# If you want to display the plot as well
plt.show()

#%%

#%%
