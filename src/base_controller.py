import numpy as np

from abc import ABC, abstractmethod
from geodetic_toolbox import quat_to_rpy

class VehicleControlState:
    def __init__(self):
        self.mode = "HOLD"        # MOVE, BRAKE, HOLD
        self.stop_timer = 0.0     # timer to stop the vehicle
        self.desired_pos = None   # 3D position to hold
        self.yaw_ref = None       # integrated yaw
        self.yaw_rate_rps = 0.0


class BaseController(ABC):
    """
    Abstract base class for all vehicle controllers.
    """
    def __init__(self, vehicle_config):
        self.vehicle_config = vehicle_config
        self.ctrl_state = VehicleControlState()
        self.stats = {
            "time_tot": np.array([0.0]),
            "sqp_iter": np.array([0]),
            "qp_iter": np.array([0])
        }

    @abstractmethod
    def compute_control(self, state, keymap, dt_sec):
        """
        Calculates the control output based on current state and user input.
        Returns:
            u (np.array): Control vector for the simulation
            predictedX (np.ndarray or None): Predicted trajectory for visualization
        """
        pass

    def get_stats(self):
        return self.stats


def update_vehicle_control_state(ctrl_state, cmd_b, current_pos, current_vel, q, dt_sec):
    horizontal_vel_mag = np.linalg.norm(current_vel[0:2])
    slow = horizontal_vel_mag < 0.5
    horizontal_cmd = not np.allclose(cmd_b[0:2], 0.0)
    vertical_cmd = not np.isclose(cmd_b[2], 0.0)
    yaw = quat_to_rpy(q)[2]

    if ctrl_state.yaw_ref is None:
        ctrl_state.yaw_ref = yaw
    if ctrl_state.desired_pos is None:
        ctrl_state.desired_pos = current_pos.copy()

    if vertical_cmd:
        ctrl_state.desired_pos[2] = current_pos[2]

    if ctrl_state.mode == "MOVE":
        ctrl_state.desired_pos[0] = current_pos[0]
        ctrl_state.desired_pos[1] = current_pos[1]
        if not horizontal_cmd:
            ctrl_state.mode = "BRAKE"
            ctrl_state.stop_timer = 0.0
            print("braking...")

    elif ctrl_state.mode == "BRAKE":
        ctrl_state.desired_pos[0] = current_pos[0]
        ctrl_state.desired_pos[1] = current_pos[1]
        ctrl_state.stop_timer += dt_sec
        if horizontal_cmd:
            ctrl_state.mode = "MOVE"
        elif slow or ctrl_state.stop_timer >= 5.0:
            ctrl_state.mode = "HOLD"
            print(f"HOLD vehicle to position: {ctrl_state.desired_pos[0]:.2f} {ctrl_state.desired_pos[1]:.2f} {ctrl_state.desired_pos[2]:.2f}. Stop Timer: {ctrl_state.stop_timer:.1f} s. Hor. velocity: {horizontal_vel_mag:.2f} m/s.")

    elif ctrl_state.mode == "HOLD":
        if horizontal_cmd:
            ctrl_state.mode = "MOVE"

