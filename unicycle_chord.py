import numpy as np
import matplotlib.pyplot as plt

def unicycle_motion(x0, y0, theta0, v, omega, dt, steps):
    """
    Simulate unicycle motion using the chord approximation.
    
    Parameters:
    x0, y0: Initial position
    theta0: Initial orientation (radians)
    v: Linear velocity (m/s)
    omega: Angular velocity (rad/s)
    dt: Time step (s)
    steps: Number of simulation steps
    
    Returns:
    x_traj, y_traj: Lists of x, y positions over time
    """
    x, y, theta = x0, y0, theta0
    x_traj, y_traj = [x], [y]
    
    for _ in range(steps):
        delta_theta = omega * dt  # Change in orientation
        
        if abs(delta_theta) > 1e-6:
            rm = v / omega  # Midpoint radius
            delta_s = 2 * rm * np.sin(delta_theta / 2)  # Chord approximation
        else:
            delta_s = v * dt  # Straight-line motion when omega ≈ 0
        
        theta_mid = theta + delta_theta / 2  # Midpoint orientation
        
        dx = delta_s * np.cos(theta_mid)
        dy = delta_s * np.sin(theta_mid)
        
        x += dx
        y += dy
        theta += delta_theta
        
        x_traj.append(x)
        y_traj.append(y)
    
    return x_traj, y_traj

# Parameters
x0, y0, theta0 = 0, 0, np.pi / 4  # Start at (0,0) with 45-degree orientation
v = 1.0  # 1 m/s linear velocity
omega = 0.5  # 0.5 rad/s angular velocity
dt = 0.1  # 0.1 second time step
steps = 100  # Run for 100 time steps

# Simulate motion
x_traj, y_traj = unicycle_motion(x0, y0, theta0, v, omega, dt, steps)

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, 'bo-', markersize=3, label='Unicycle Path')
plt.quiver(x_traj, y_traj, np.cos(np.linspace(theta0, theta0 + omega * dt * steps, steps+1)),
           np.sin(np.linspace(theta0, theta0 + omega * dt * steps, steps+1)),
           scale=10, color='r', alpha=0.5, label='Heading')
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Unicycle Motion with Chord Approximation")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
