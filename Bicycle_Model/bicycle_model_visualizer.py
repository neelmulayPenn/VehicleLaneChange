import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class VehicleBicycleVisualizer:
    def __init__(self, Lf=1.2, Lr=1.6, width=1.0, show_velocity=True, show_trail=True):
        """
        Args:
            Lf: float - distance from CG to front axle [m]
            Lr: float - distance from CG to rear axle [m]
            width: float - vehicle width [m]
            show_velocity: bool - whether to display velocity vectors
            show_trail: bool - whether to display trajectory trail
        """
        self.Lf = Lf
        self.Lr = Lr
        self.width = width
        self.show_velocity = show_velocity
        self.show_trail = show_trail

        # Setup figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.title = self.ax.set_title("Vehicle Visualization")

        # Define geometry
        length = Lf + Lr
        half_w = width / 2
        self.body = np.array([
            [-Lr,  half_w],
            [ Lf,  half_w],
            [ Lf, -half_w],
            [-Lr, -half_w],
            [-Lr,  half_w]
        ]).T  # shape (2,5)

        self.front_axle = np.array([
            [Lf, Lf],
            [half_w, -half_w]
        ])
        self.rear_axle = np.array([
            [-Lr, -Lr],
            [half_w, -half_w]
        ])

        # Front wheel rectangle (short segment)
        wheel_half = 0.3
        wheel_len = 0.5
        self.front_wheel = np.array([
            [-wheel_len / 2, -wheel_len / 2, wheel_len / 2, wheel_len / 2, -wheel_len / 2],
            [ wheel_half, -wheel_half, -wheel_half,  wheel_half,  wheel_half]
        ])

        # Draw static components
        self.body_patch = self.ax.fill(
            self.body[0, :], self.body[1, :],
            facecolor=[0.6, 0.6, 0.6], edgecolor='k', zorder=2
        )
        self.front_axle_line, = self.ax.plot([], [], 'k-', lw=2, zorder=1)
        self.rear_axle_line, = self.ax.plot([], [], 'k-', lw=2, zorder=1)
        # self.front_wheel_patch = self.ax.fill([], [], 'b', edgecolor='k', zorder=3)
        self.front_wheel_patch = self.ax.fill(
            self.front_wheel[0, :],
            self.front_wheel[1, :],
            facecolor='b',
            edgecolor='k',
            zorder=3
        )


        # Optional elements
        if self.show_velocity:
            self.vel_arrow = self.ax.quiver([], [], [], [], color='r', scale=10, width=0.004)
        else:
            self.vel_arrow = None

        if self.show_trail:
            self.trail_line, = self.ax.plot([], [], 'g--', lw=1, alpha=0.7, zorder=0)
            self.trail_x, self.trail_y = [], []
        else:
            self.trail_line = None

    def draw(self, x, u, t=0.0):
        """
        Draw vehicle at state x with input u.
        Args:
            x: [X, Y, psi, vx, vy, w]
            u: [delta, ax]
            t: float, time
        """
        X, Y, psi = x[0], x[1], x[2]
        vx, vy = x[3], x[4]
        delta = u[0]

        # Body rotation
        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
        # Transform shapes
        body_xy = R @ self.body + np.array([[X], [Y]])
        front_axle_xy = R @ self.front_axle + np.array([[X], [Y]])
        rear_axle_xy = R @ self.rear_axle + np.array([[X], [Y]])

        # Front wheel rotation (steering)
        Rw = np.array([[np.cos(psi + delta), -np.sin(psi + delta)],
                       [np.sin(psi + delta),  np.cos(psi + delta)]])
        wheel_xy = Rw @ self.front_wheel + R @ np.array([[self.Lf], [0]]) + np.array([[X], [Y]])

        # Update body and axles
        self.body_patch[0].get_path().vertices[:, 0] = body_xy[0, :]
        self.body_patch[0].get_path().vertices[:, 1] = body_xy[1, :]
        self.front_axle_line.set_data(front_axle_xy[0, :], front_axle_xy[1, :])
        self.rear_axle_line.set_data(rear_axle_xy[0, :], rear_axle_xy[1, :])
        wheel_path = self.front_wheel_patch[0].get_path()
        wheel_path.vertices[:, 0] = wheel_xy[0, :]
        wheel_path.vertices[:, 1] = wheel_xy[1, :]


        # Velocity vector
        if self.show_velocity:
            self.vel_arrow.set_offsets(np.array([[X, Y]]))
            self.vel_arrow.set_UVC(vx, vy)

        # Trajectory trail
        if self.show_trail:
            self.trail_x.append(X)
            self.trail_y.append(Y)
            self.trail_line.set_data(self.trail_x, self.trail_y)

        self.title.set_text(f"t = {t:.2f} s")

    def show(self):
        plt.show()


def create_animation(vis, x_traj, u_traj, dt):
    """
    Creates an animation for the given vehicle visualizer.
    Args:
        vis: VehicleBicycleVisualizer
        x_traj: ndarray (6, N)
        u_traj: ndarray (2, N)
        dt: float timestep [s]
    """
    def update(i):
        vis.draw(x_traj[:, i], u_traj[:, i], i * dt)
        return []

    ani = animation.FuncAnimation(
        vis.fig, update, frames=x_traj.shape[1], interval=dt * 1000, blit=False
    )
    return ani