import matplotlib.pyplot as plt
import numpy as np

time = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
position = [0.0, 1.1,3.2,5.7,8.9,12.7,17,22,27.4]
position_poly = np.polyfit(time, position, 2)

def function(x):
    return f"{x[0]:.2f}x^2 + {x[1]:.2f}x + {x[2]:.2f}"


plt.plot(time, position, label = function(position_poly), color = 'green')
plt.xlabel('time')
plt.ylabel('position')

plt.legend()

plt.title("Position and Velocity of a Fletcher Trolley Cart\n"
    "Accelerating across a plane at 59.6 cm/s^2")

plt.show()

print(function(position_poly))


