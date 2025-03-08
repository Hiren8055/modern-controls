import numpy as np
import matplotlib.pyplot as plt

def unicycle_motion(x0, y0, theta0, v, R, dt, steps):
    """
    Simulate unicycle motion using the chord approximation.
    
    Parameters:
    x0, y0: Initial position
    theta0: Initial orientation (radians)
    v: Linear velocity (m/s)
    R: Turning radius (m)
    dt: Time step (s)
    steps: Number of simulation steps
    
    Returns:
    x_traj, y_traj: Lists of x, y positions over time
    """
    omega = v / R  # Adjust omega to maintain the same radius
    x, y, theta = x0, y0, theta0
    x_traj, y_traj = [x], [y]
    
    for _ in range(steps):
        delta_theta = omega * dt  # Change in orientation
        
        if abs(delta_theta) > 1e-6:
            delta_s = 2 * R * np.sin(delta_theta / 2)  # Chord approximation
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
v = 1.0  # Both models have the same velocity
R = 2.0  # Fixed turning radius
dt = 0.1  # 0.1 second time step
steps = 100  # Run for 100 time steps

# Simulate motion
x_traj1, y_traj1 = unicycle_motion(x0, y0, theta0, v, R, dt, steps)
x_traj2, y_traj2 = unicycle_motion(x0, y0, theta0, 3*v, R, dt, steps)  # Adjusted omega

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_traj1, y_traj1, 'bo-', markersize=3, label='Unicycle Path (1 m/s)')
plt.plot(x_traj2, y_traj2, 'go-', markersize=3, label='Unicycle Path (3 m/s)')

# plt.quiver(x_traj1, y_traj1, np.cos(np.linspace(theta0, theta0 + (v/R) * dt * steps, steps+1)),
#            np.sin(np.linspace(theta0, theta0 + (v/R) * dt * steps, steps+1)),
#            scale=10, color='r', alpha=0.5, label='Heading (1 m/s)')
# plt.quiver(x_traj2, y_traj2, np.cos(np.linspace(theta0, theta0 + (3*v/R) * dt * steps, steps+1)),
#            np.sin(np.linspace(theta0, theta0 + (3*v/R) * dt * steps, steps+1)),
#            scale=10, color='purple', alpha=0.5, label='Heading (3 m/s)')

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Unicycle Motion with Chord Approximation (Same Radius)")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
