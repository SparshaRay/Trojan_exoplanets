import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
# ax = fig.add_subplot(111, projection='3d')
ax1.set_xlim(-120, 120)
ax1.set_ylim(-120, 120)
ax1.set_zlim(-25, 25)
ax2.set_xlim(-120, 120)
ax2.set_ylim(-120, 120)
ax2.set_zlim(-25, 25)

def plot_trajectory(masses, positions, velocities, num_steps, dt):
    num_bodies = len(masses)
    num_dimensions = len(positions[0])

    # Define the system of differential equations
    def equations_of_motion(state, t):
        positions = state[:num_bodies*num_dimensions].reshape((num_bodies, num_dimensions))
        velocities = state[num_bodies*num_dimensions:].reshape((num_bodies, num_dimensions))
        
        acceleration = np.zeros((num_bodies, num_dimensions))
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i != j:
                    r = positions[j] - positions[i]
                    acceleration[i] += masses[j] * r / np.linalg.norm(r)**3
        
        return np.concatenate((velocities.flatten(), acceleration.flatten()))

    # Set up the initial state
    initial_state = np.concatenate((positions.flatten(), velocities.flatten()))

    # Integrate the equations of motion using odeint
    t = np.linspace(0, num_steps*dt, num_steps)
    states = odeint(equations_of_motion, initial_state, t)

    # Extract positions from the states
    positions_over_time = states[:, :num_bodies*num_dimensions].reshape((num_steps, num_bodies, num_dimensions))[::10]

    # Plot the trajectory in 3D

    ax2.plot(0, 0, 0, 'ro')
    # ax2.plot(positions_over_time[:, 0, 0] - positions_over_time[:, 0, 0], positions_over_time[:, 0, 1] - positions_over_time[:, 0, 1], positions_over_time[:, 0, 2] - positions_over_time[:, 0, 2], 'r-', alpha=0.75, lw=0.75)
    ax2.plot(positions_over_time[:, 1, 0] - positions_over_time[:, 0, 0], positions_over_time[:, 1, 1] - positions_over_time[:, 0, 1], positions_over_time[:, 1, 2] - positions_over_time[:, 0, 2], 'g-', alpha=0.75, lw=0.75)
    ax2.plot(positions_over_time[:, 2, 0] - positions_over_time[:, 0, 0], positions_over_time[:, 2, 1] - positions_over_time[:, 0, 1], positions_over_time[:, 2, 2] - positions_over_time[:, 0, 2], 'b-', alpha=0.75, lw=0.75)

    print(positions_over_time.shape)
    plotting_array = np.zeros((1, 3))

    for data_point in positions_over_time :
        sun = data_point[0]
        jupiter = data_point[1]
        trojan = data_point[2]

        offset_vec = sun - jupiter
        azm = np.arctan2(offset_vec[0], offset_vec[1])
        alt = np.arctan2(offset_vec[2], np.linalg.norm(offset_vec[:2]))
        rotation_matrix_3d = np.array([[np.cos(azm)*np.cos(alt), -np.sin(azm), np.cos(azm)*np.sin(alt)],
                                       [np.sin(azm)*np.cos(alt), np.cos(azm), np.sin(azm)*np.sin(alt)],
                                       [-np.sin(alt), 0, np.cos(alt)]], dtype=float)
        
        heading = trojan - jupiter
        header_corrected = np.dot(rotation_matrix_3d, heading) - np.array([0, 100, 0])
        plotting_array = np.vstack((plotting_array, header_corrected))
    
    print('appended')
    
    ax1.plot(plotting_array[1:, 0], plotting_array[1:, 1], plotting_array[1:, 2], alpha=0.75, lw=0.75)
    ax1.plot(0, -100, 0, 'go')
    ax1.plot(0, 0, 0, 'ro')

    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

# Example usage
init_posrad = 100 # np.random.uniform(low=90, high=110)
init_posang = np.pi/4
init_velmag = 10 # np.random.uniform(low=9, high=10)
init_velang = np.pi/4


masses = [10000.0, 25.0, 25.0]  # Masses of the bodies (Sun, Jupiter, Trojan asteroid)
positions = np.array([[0.0, 0.0, 0.0], [100, 0.0, 0.0], [init_posrad*np.cos(init_posang), init_posrad*np.sin(init_posang), 5]])  # Initial positions (Sun, Jupiter, Trojan asteroid)
velocities = np.array([[0.0, 0.0, 0.0], [0.0, init_velmag, 0.0], [-init_velmag*np.sin(init_velang), init_velmag*np.cos(init_velang), 0.0]])  # Initial velocities (Sun, Jupiter, Trojan asteroid)
num_steps = 75000  # Number of simulation steps
dt = 0.01  # Time step

plot_trajectory(masses, positions, velocities, num_steps, dt)
plt.show()

