import numpy as np
import matplotlib.pyplot as plt

def rk4_step(x, y, theta, v, omega, dt):
    """
    Perform one step of RK4 integration for the unicycle model.
    """
    def f(state):
        x, y, theta = state
        return np.array([v * np.cos(theta), v * np.sin(theta), omega])
    
    k1 = f((x, y, theta))
    k2 = f((x + dt/2 * k1[0], y + dt/2 * k1[1], theta + dt/2 * k1[2]))
    k3 = f((x + dt/2 * k2[0], y + dt/2 * k2[1], theta + dt/2 * k2[2]))
    k4 = f((x + dt * k3[0], y + dt * k3[1], theta + dt * k3[2]))
    
    new_state = np.array([x, y, theta]) + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

def unicycle_motion_rk4(x0, y0, theta0, v, R, dt, steps):
    """
    Simulate unicycle motion using RK4 integration.
    """
    omega = v / R  # Adjust omega to maintain the same radius
    x, y, theta = x0, y0, theta0
    x_traj, y_traj = [x], [y]
    
    for _ in range(steps):
        x, y, theta = rk4_step(x, y, theta, v, omega, dt)
        x_traj.append(x)
        y_traj.append(y)
    
    return x_traj, y_traj

# Parameters
x0, y0, theta0 = 0, 0, np.pi / 4  # Start at (0,0) with 45-degree orientation
v = 1.0  # Both models have the same velocity
R = 2.0  # Fixed turning radius
dt = 0.1  # 0.1 second time step
steps = 100  # Run for 100 time steps

# Simulate motion using RK4
x_traj1, y_traj1 = unicycle_motion_rk4(x0, y0, theta0, v, R, dt, steps)
x_traj2, y_traj2 = unicycle_motion_rk4(x0, y0, theta0, 3*v, R, dt, steps)  # Adjusted omega

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_traj1, y_traj1, 'bo-', markersize=3, label='Unicycle Path (1 m/s)')
plt.plot(x_traj2, y_traj2, 'go-', markersize=3, label='Unicycle Path (3 m/s)')

plt.quiver(x_traj1, y_traj1, np.cos(np.linspace(theta0, theta0 + (v/R) * dt * steps, steps+1)),
           np.sin(np.linspace(theta0, theta0 + (v/R) * dt * steps, steps+1)),
           scale=10, color='r', alpha=0.5, label='Heading (1 m/s)')
plt.quiver(x_traj2, y_traj2, np.cos(np.linspace(theta0, theta0 + (3*v/R) * dt * steps, steps+1)),
           np.sin(np.linspace(theta0, theta0 + (3*v/R) * dt * steps, steps+1)),
           scale=10, color='purple', alpha=0.5, label='Heading (3 m/s)')

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Unicycle Motion with RK4 Integration (Same Radius)")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()