#!/usr/bin/env python3

"""
Main runtime file to watch the simulation environment and vehicle in
real-time.

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
import scipy.linalg
import time
import pygame
import threading
import copy
import traceback
from abc import ABC, abstractmethod
from pathlib import Path  # to figure out path of .stl files

from geodetic_toolbox import *
from multirotorsimulatorenv import MultirotorSimEnv # Physic simulation
from visualization3d import RenderStlPygame # 3D visualization

# NMPC specific imports
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, sqrt, sumsqr
from mpc_copter.copter_model_position import export_copterpos_ode_model
from mpc_copter.build_ocp import build_ocp

#       ┌───────────────────────────────────────────────────────────────┐
#       │                                                               │
#       │                    MAIN THREAD (main.py)                      │
#       │                                                               │
#       │  • Pygame Event Loop (Tastatur-/Mauseingaben)                 │
#       │  • 3D Visualisierung (OpenGL)                                 │
#       │  • Sammelt Input & rendert das Fahrzeug                       │
#       │                                                               │
#       └───────┬───────────────────────────────────────▲───────────────┘
#               │                                       │
#               │ Tastatureingaben                      │ Rotation ('R'), Position ('pos')
#               │ ('keymap')                            │ Vorhersage ('predictedX')
#               ▼                                       │
#       ╔═══════════════════════════════════════════════════════════════╗
#       ║                 GLOBAL MESSAGE BOX (g_thread_msgbox)          ║
#       ║                 [Geschützt durch g_thread_msgbox_lock]        ║
#       ║                                                               ║
#       ║  Enthält: state, u, keymap, R, pos, predictedX, fps, stats... ║
#       ╚══════════╦════════════════════════════════════╦═══════════════╝
#                  │                                    │
#                  │ Lese: 'state', 'keymap'            │ Lese: 'u' (Motorkommandos)
#                  │ Schreibe: 'u', 'predictedX'        │ Schreibe: 'state', 'R', 'pos'
#                  ▼                                    ▼
#       ┌────────────────────────────┐        ┌─────────────────────────┐
#       │                            │        │                         │
#       │      CONTROLLER THREAD     │        │    SIMULATION THREAD    │
#       │                            │        │                         │
#       │  • Läuft z.B. mit 100 Hz   │        │  • Läuft mit >= 240 Hz  │
#       │  • Ruft in einer Schleife  │        │  • Berechnet Physik     │
#       │    compute_control() auf   │        │    (MultirotorSimEnv)   │
#       │                            │        │  • Führt step(u) aus    │
#       └───────┬────────────────────┘        └─────────────────────────┘
#               │
#               │ Nutzt Polymorphismus
#               ▼
#       ┌────────────────────────────┐
#       │       BaseController       │◄── Abstrakte Basisklasse
#       │                            │    (Verwaltet State Machine & FSM)
#       ├──────────────┬─────────────┤
#       │              │             │
#       │NMPCController│PIDController│◄── Spezifische Implementierungen
#       │              │             │    (Berechnen das eigentliche 'u')
#       └──────────────┴─────────────┘


# Global messagebox to exchange data between threads
g_thread_msgbox = {
    "R": np.identity(3),
    "pos": np.array([0.0, 0.0, 0.0]),
    "keymap": {
        "longitudinal_cmd": 0.0,
        "lateral_cmd": 0.0,
        "yaw_cmd": 0.0,
        "vertical_cmd": 0.0,
    },
    "ctrl_fps": 0,
    "render_fps": 0,
    "ctrl_time_tot": np.array([0.0]),
    "ctrl_sqp_iter": np.array([0]),
    "ctrl_qp_iter": np.array([0]),
    "ctrl_time_max": 0.0,
    "ctrl_time_min": 0.0,
    "ctrl_time_avg": 0.0,
    "ctrl_time_std": 0.0,
}
g_thread_msgbox_lock = threading.Lock()
g_sim_running = True

# ---------------------------------------------------------
# FSM & State Management
# ---------------------------------------------------------
class VehicleControlState:
    def __init__(self):
        self.mode = "HOLD"        # MOVE, BRAKE, HOLD
        self.stop_timer = 0.0     # timer to stop the vehicle
        self.desired_pos = None   # 3D position to hold
        self.yaw_ref = None       # integrated yaw
        self.yaw_rate_rps = 0.0

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

# ---------------------------------------------------------
# Controller Architecture
# ---------------------------------------------------------
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

class NMPCController(BaseController):
    """
    Nonlinear Model Predictive Controller using Acados.
    """
    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        ocp_cfg = vehicle_config.ocp_sim
        self.ocp, self.model, self.nx, self.nu, self.ny, self.N_horizon, self.Tf = build_ocp(vehicle_config, ocp_cfg)
        solver_json = "acados_ocp_" + self.model.name + ".json"
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)
        self.predictedX = np.ndarray((self.N_horizon, self.nx))

    def compute_control(self, state, keymap, dt_sec):
        self._update_references(state, keymap, dt_sec)

        # Solve OCP
        u = self.acados_ocp_solver.solve_for_x0(x0_bar=state)

        if self.model.ctrlout_u_is_squared:
            u = np.clip(u, self.vehicle_config.umin**2, self.vehicle_config.umax**2)
            u = np.sqrt(u)
        else:
            u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        for i in range(self.N_horizon):
            self.predictedX[i, :] = self.acados_ocp_solver.get(i, "x")

        # Update stats
        self.stats["time_tot"] = self.acados_ocp_solver.get_stats("time_tot")
        self.stats["sqp_iter"] = self.acados_ocp_solver.get_stats("sqp_iter")
        self.stats["qp_iter"] = self.acados_ocp_solver.get_stats("qp_iter")

        return u, self.predictedX

    def _update_references(self, state, keymap, dt_sec):
        cfg = self.vehicle_config.state_cfg
        pos = state[cfg["pos3d_index"]:cfg["pos3d_index_end"]]
        vel = state[cfg["vel3d_index"]:cfg["vel3d_index_end"]]
        q   = state[cfg["q_index"]:cfg["q_index_end"]]
        yaw = quat_to_rpy(q)[2]

        cmd_b = np.array([
            keymap.get("longitudinal_cmd", 0.0),
            keymap.get("lateral_cmd", 0.0),
            keymap.get("vertical_cmd", 0.0),
        ])
        yaw_cmd = keymap.get("yaw_cmd", 0.0)

        update_vehicle_control_state(self.ctrl_state, cmd_b, pos, vel, q, dt_sec)

        if not np.isclose(yaw_cmd, 0.0):
            self.ctrl_state.yaw_rate_rps = yaw_cmd * self.vehicle_config.max_rotation_rate_rps
            self.ctrl_state.yaw_ref += self.ctrl_state.yaw_rate_rps * dt_sec
            self.ctrl_state.yaw_ref = angle_diff(self.ctrl_state.yaw_ref, 0.0)
        else:
            if not np.isclose(self.ctrl_state.yaw_rate_rps, 0.0):
                self.ctrl_state.yaw_ref = yaw
                self.ctrl_state.yaw_rate_rps = 0.0

        if np.allclose(cmd_b, 0.0):
            vel_n_ref = np.zeros(3)
        else:
            v_b = np.array([
                cmd_b[0] * self.vehicle_config.max_horizontal_velocity_mps,
                cmd_b[1] * self.vehicle_config.max_horizontal_velocity_mps,
                0.0,
            ])
            R_yaw = quat_to_matrix(quat_from_rpy(0.0, 0.0, yaw))
            vel_n_ref = R_yaw @ v_b
            vel_n_ref[2] = cmd_b[2] * self.vehicle_config.max_vertical_velocity_mps

        m = self.vehicle_config.mass_kg
        g = self.vehicle_config.gravity_n[2]
        v_norm = np.hypot(vel_n_ref[0], vel_n_ref[1])
        c_D = self.vehicle_config.windresistance
        F_x = c_D * vel_n_ref[0] * v_norm
        F_y = c_D * vel_n_ref[1] * v_norm
        roll_ref = np.arctan(F_y / (m * g))
        pitch_ref = -np.arctan(F_x / (m * g))
        q_ref = quat_from_rpy(roll_ref, pitch_ref, self.ctrl_state.yaw_ref)
        omega_ref = np.array([0.0, 0.0, self.ctrl_state.yaw_rate_rps])

        pos_ref = self.ctrl_state.desired_pos
        yref = np.copy(self.ocp.cost.yref)
        yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_ref
        yref[cfg["vel3d_index"]:cfg["vel3d_index_end"]] = vel_n_ref
        yref[cfg["q_index"]:cfg["q_index_end"]] = q_ref
        yref[cfg["omega_index"]:cfg["omega_index_end"]] = omega_ref
        self.ocp.cost.yref = np.copy(yref)

        N = self.ocp.solver_options.N_horizon
        pos_pred = np.zeros((N + 1, 3))
        pos_pred[0] = pos_ref
        for j in range(1, N + 1):
            pos_pred[j] = pos_pred[j - 1] + vel_n_ref * dt_sec

        for j in range(N):
            yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[j]
            self.acados_ocp_solver.set(j, "yref", np.copy(yref))

        yref_e = np.copy(yref[:state.size])
        yref_e[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[N]
        self.acados_ocp_solver.set(N, "yref", yref_e)
        self.ocp.cost.yref_e = yref_e

class PidObject:
    """
    Helper for the PID controller
    """
    __slots__ = (
        "kp", "ki", "kd", "kff",
        "i_limit", "output_limit",
        "desired", "integ", "prev_measured",
        "out_p", "out_i", "out_d", "out_ff",
        "initialized",
    )

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, kff=0.0,
                 i_limit=0.0, output_limit=0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.kff = float(kff)
        self.i_limit = float(i_limit)         # 0 means no limit
        self.output_limit = float(output_limit)
        self.desired = 0.0
        self.integ = 0.0
        self.prev_measured = 0.0
        self.out_p = 0.0
        self.out_i = 0.0
        self.out_d = 0.0
        self.out_ff = 0.0
        self.initialized = False

    def reset(self, measured=0.0):
        self.integ = 0.0
        self.prev_measured = measured
        self.initialized = True

    def set_desired(self, desired):
        self.desired = float(desired)

    def update(self, measured, dt, is_yaw_angle=False):
        # Initialize prev_measured on first call to avoid a huge derivative spike
        if not self.initialized:
            self.prev_measured = measured
            self.initialized = True

        error = self.desired - measured
        if is_yaw_angle:
            if error > 180.0:
                error -= 360.0
            elif error < -180.0:
                error += 360.0

        self.out_p = self.kp * error

        # Derivative on measurement: delta = -(measured - prev_measured)
        delta = -(measured - self.prev_measured)
        if is_yaw_angle:
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0
        deriv = delta / dt if dt > 0.0 else 0.0
        if not np.isfinite(deriv):
            deriv = 0.0
        self.out_d = self.kd * deriv

        # Integrator with optional clamp
        self.integ += error * dt
        if self.i_limit != 0.0:
            if self.integ > self.i_limit:
                self.integ = self.i_limit
            elif self.integ < -self.i_limit:
                self.integ = -self.i_limit
        self.out_i = self.ki * self.integ

        self.out_ff = self.kff * self.desired

        output = self.out_p + self.out_i + self.out_d + self.out_ff

        if self.output_limit != 0.0:
            if output > self.output_limit:
                output = self.output_limit
            elif output < -self.output_limit:
                output = -self.output_limit

        self.prev_measured = measured
        return output


def _g(cfg, name, default):
    """Get a gain from vehicle_config or fall back to the Crazyflie default."""
    return float(getattr(cfg, name, default))


class PIDController(BaseController):
    """
    Cascaded PID controller

    Cascade (outer -> inner):
        Position (XYZ)  -> velocity setpoint (NED-frame)
        Velocity (XYZ)  -> roll/pitch attitude setpoint + thrust
        Attitude (RPY)  -> body-rate setpoint
        Body rate       -> raw torque commands (roll, pitch, yaw)
        Power mixer     -> 4 motor commands

    Coordinate frames:
        - Simulation works in NED (x North, y East, z Down).
        - Crazyflie firmware works in FLU (x Forward, y Left, z Up).
          Roll/pitch/yaw conventions otherwise match (roll about x, pitch
          about y, yaw about z, all in degrees in the firmware's high-level
          loops, all in deg/s in the rate loop).
        - Internally we run the PIDs in the Crazyflie's FLU/deg convention
          (so the gain values can be used unchanged) and convert at the
          edges:  vel_z_FLU = -vel_z_NED, pos_z_FLU = -pos_z_NED, and the
          velocity setpoint is rotated into the body frame the same way
          the firmware does it.
    """

    # ---- Crazyflie firmware defaults (used as fallback if the
    # ----  vehicle_config does not provide an attribute). All values
    # ----  are taken verbatim from controller_pid.c / pid.h.
    _DEFAULTS = {
        # Inner: body rates [deg/s]
        "pid_roll_rate_kp":  200.0, "pid_roll_rate_ki":  400.0,
        "pid_roll_rate_kd":    2.5, "pid_roll_rate_kff":   0.0,
        "pid_roll_rate_integration_limit":  33.3,
        "pid_pitch_rate_kp": 200.0, "pid_pitch_rate_ki": 400.0,
        "pid_pitch_rate_kd":   2.5, "pid_pitch_rate_kff":  0.0,
        "pid_pitch_rate_integration_limit": 33.3,
        "pid_yaw_rate_kp":   120.0, "pid_yaw_rate_ki":    16.7,
        "pid_yaw_rate_kd":     0.0, "pid_yaw_rate_kff":    0.0,
        "pid_yaw_rate_integration_limit":  166.7,
        # Attitude [deg]
        "pid_roll_kp":  6.0, "pid_roll_ki":  3.0,
        "pid_roll_kd":  0.0, "pid_roll_kff": 0.0,
        "pid_roll_integration_limit":  20.0,
        "pid_pitch_kp": 6.0, "pid_pitch_ki": 3.0,
        "pid_pitch_kd": 0.0, "pid_pitch_kff":0.0,
        "pid_pitch_integration_limit": 20.0,
        "pid_yaw_kp":   6.0, "pid_yaw_ki":   1.0,
        "pid_yaw_kd":   0.35,"pid_yaw_kff":  0.0,
        "pid_yaw_integration_limit":  360.0,
        # Velocity [m/s]
        "pid_vel_x_kp": 25.0, "pid_vel_x_ki": 1.0,
        "pid_vel_x_kd":  0.0, "pid_vel_x_kff":0.0,
        "pid_vel_y_kp": 25.0, "pid_vel_y_ki": 1.0,
        "pid_vel_y_kd":  0.0, "pid_vel_y_kff":0.0,
        "pid_vel_z_kp": 25.0, "pid_vel_z_ki":15.0,
        "pid_vel_z_kd":  0.0, "pid_vel_z_kff":0.0,
        # Velocity loop limits / thrust mapping
        "pid_vel_roll_max":         20.0,    # deg
        "pid_vel_pitch_max":        20.0,    # deg
        "pid_vel_thrust_base":   30000.0,    # int16 PWM units (CF mapping)
        "pid_vel_thrust_min":    20000.0,
        # Position [m]
        "pid_pos_x_kp": 2.0, "pid_pos_x_ki": 0.0,
        "pid_pos_x_kd": 0.0, "pid_pos_x_kff":0.0,
        "pid_pos_y_kp": 2.0, "pid_pos_y_ki": 0.0,
        "pid_pos_y_kd": 0.0, "pid_pos_y_kff":0.0,
        "pid_pos_z_kp": 2.0, "pid_pos_z_ki": 0.5,
        "pid_pos_z_kd": 0.0, "pid_pos_z_kff":0.0,
        "pid_pos_vel_x_max": 1.0,
        "pid_pos_vel_y_max": 1.0,
        "pid_pos_vel_z_max": 1.0,
    }

    _THRUST_PWM_MAX = 65535.0

    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        cfg = vehicle_config

        def G(name):
            return _g(cfg, name, self._DEFAULTS[name])

        # --- Rate loop (innermost) --------------------------------------
        self.pid_roll_rate = PidObject(
            G("pid_roll_rate_kp"),  G("pid_roll_rate_ki"),
            G("pid_roll_rate_kd"),  G("pid_roll_rate_kff"),
            i_limit=G("pid_roll_rate_integration_limit"))
        self.pid_pitch_rate = PidObject(
            G("pid_pitch_rate_kp"), G("pid_pitch_rate_ki"),
            G("pid_pitch_rate_kd"), G("pid_pitch_rate_kff"),
            i_limit=G("pid_pitch_rate_integration_limit"))
        self.pid_yaw_rate = PidObject(
            G("pid_yaw_rate_kp"),   G("pid_yaw_rate_ki"),
            G("pid_yaw_rate_kd"),   G("pid_yaw_rate_kff"),
            i_limit=G("pid_yaw_rate_integration_limit"))

        # --- Attitude loop ----------------------------------------------
        self.pid_roll = PidObject(
            G("pid_roll_kp"),  G("pid_roll_ki"),
            G("pid_roll_kd"),  G("pid_roll_kff"),
            i_limit=G("pid_roll_integration_limit"))
        self.pid_pitch = PidObject(
            G("pid_pitch_kp"), G("pid_pitch_ki"),
            G("pid_pitch_kd"), G("pid_pitch_kff"),
            i_limit=G("pid_pitch_integration_limit"))
        self.pid_yaw = PidObject(
            G("pid_yaw_kp"),   G("pid_yaw_ki"),
            G("pid_yaw_kd"),   G("pid_yaw_kff"),
            i_limit=G("pid_yaw_integration_limit"))

        # --- Velocity loop ----------------------------------------------
        # No iLimit in the CF firmware for velocity loops (= 0 -> unlimited)
        self.pid_vel_x = PidObject(G("pid_vel_x_kp"), G("pid_vel_x_ki"),
                                   G("pid_vel_x_kd"), G("pid_vel_x_kff"))
        self.pid_vel_y = PidObject(G("pid_vel_y_kp"), G("pid_vel_y_ki"),
                                   G("pid_vel_y_kd"), G("pid_vel_y_kff"))
        self.pid_vel_z = PidObject(G("pid_vel_z_kp"), G("pid_vel_z_ki"),
                                   G("pid_vel_z_kd"), G("pid_vel_z_kff"))

        # --- Position loop ----------------------------------------------
        self.pid_pos_x = PidObject(G("pid_pos_x_kp"), G("pid_pos_x_ki"),
                                   G("pid_pos_x_kd"), G("pid_pos_x_kff"))
        self.pid_pos_y = PidObject(G("pid_pos_y_kp"), G("pid_pos_y_ki"),
                                   G("pid_pos_y_kd"), G("pid_pos_y_kff"))
        self.pid_pos_z = PidObject(G("pid_pos_z_kp"), G("pid_pos_z_ki"),
                                   G("pid_pos_z_kd"), G("pid_pos_z_kff"))

        # --- Limits / thrust mapping ------------------------------------
        self.vel_roll_max     = G("pid_vel_roll_max")        # deg
        self.vel_pitch_max    = G("pid_vel_pitch_max")       # deg
        self.vel_thrust_base  = G("pid_vel_thrust_base")     # PWM counts
        self.vel_thrust_min   = G("pid_vel_thrust_min")
        self.pos_vel_x_max    = G("pid_pos_vel_x_max")       # m/s
        self.pos_vel_y_max    = G("pid_pos_vel_y_max")
        self.pos_vel_z_max    = G("pid_pos_vel_z_max")

        # Integrated yaw setpoint (the firmware integrates the yaw rate
        # command into an absolute yaw angle and then does angle PID).
        self._yaw_desired_deg = None

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _wrap_deg(angle_deg):
        """Wrap to (-180, 180]."""
        a = (angle_deg + 180.0) % 360.0 - 180.0
        if a == -180.0:
            a = 180.0
        return a

    def _reset_all(self, roll_deg, pitch_deg, yaw_deg, pos_ned, vel_ned):
        for pid in (self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate):
            pid.reset(0.0)
        self.pid_roll.reset(roll_deg)
        self.pid_pitch.reset(pitch_deg)
        self.pid_yaw.reset(yaw_deg)
        # FLU velocity (z up) for the velocity PID
        self.pid_vel_x.reset(vel_ned[0])
        self.pid_vel_y.reset(-vel_ned[1])      # NED y(East) -> FLU y(Left) = -y
        self.pid_vel_z.reset(-vel_ned[2])      # NED z(Down) -> FLU z(Up)   = -z
        self.pid_pos_x.reset(pos_ned[0])
        self.pid_pos_y.reset(-pos_ned[1])
        self.pid_pos_z.reset(-pos_ned[2])
        self._yaw_desired_deg = yaw_deg

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def compute_control(self, state, keymap, dt_sec):
        cfg = self.vehicle_config.state_cfg
        pos_ned = np.asarray(state[cfg["pos3d_index"]:cfg["pos3d_index_end"]], dtype=float)
        vel_ned = np.asarray(state[cfg["vel3d_index"]:cfg["vel3d_index_end"]], dtype=float)
        q       = np.asarray(state[cfg["q_index"]:cfg["q_index_end"]], dtype=float)
        omega   = np.asarray(state[cfg["omega_index"]:cfg["omega_index_end"]], dtype=float)

        rpy = quat_to_rpy(q)                       # radians, NED body-frame
        roll_deg  = np.rad2deg(rpy[0])
        pitch_deg = np.rad2deg(rpy[1])
        yaw_deg   = np.rad2deg(rpy[2])

        # Body rates [deg/s]. omega is body-frame angular velocity.
        p_dps = np.rad2deg(omega[0])
        q_dps = np.rad2deg(omega[1])
        r_dps = np.rad2deg(omega[2])

        # FSM updates desired_pos / yaw_ref / yaw_rate_rps
        cmd_b = np.array([
            keymap.get("longitudinal_cmd", 0.0),
            keymap.get("lateral_cmd", 0.0),
            keymap.get("vertical_cmd", 0.0),
        ])
        update_vehicle_control_state(self.ctrl_state, cmd_b, pos_ned, vel_ned, q, dt_sec)
        yaw_cmd = keymap.get("yaw_cmd", 0.0)

        # First call: latch internal references to current state
        if self._yaw_desired_deg is None:
            self._reset_all(roll_deg, pitch_deg, yaw_deg, pos_ned, vel_ned)

        # ---------------- Yaw setpoint integration (rate -> angle) ------
        if not np.isclose(yaw_cmd, 0.0):
            yaw_rate_dps = yaw_cmd * np.rad2deg(self.vehicle_config.max_rotation_rate_rps)
            self._yaw_desired_deg = self._wrap_deg(
                self._yaw_desired_deg + yaw_rate_dps * dt_sec
            )

        # ===============================================================
        # Outer loop:  Position -> velocity setpoint  (in FLU frame)
        # ===============================================================
        # Setpoint horizontal position from FSM (NED), vertical from FSM too.
        pos_sp_ned = self.ctrl_state.desired_pos
        # Convert state and setpoint to FLU for the PIDs
        pos_x_flu     =  pos_ned[0]
        pos_y_flu     = -pos_ned[1]
        pos_z_flu     = -pos_ned[2]
        pos_x_sp_flu  =  pos_sp_ned[0]
        pos_y_sp_flu  = -pos_sp_ned[1]
        pos_z_sp_flu  = -pos_sp_ned[2]

        # Manual horizontal velocity command (when sticks are deflected
        # the firmware's positionController switches to velocity-hold for
        # the respective axis). We mirror that: if user commands lateral
        # or longitudinal motion, bypass the position PID for X/Y and
        # feed a velocity setpoint directly. Z always uses position-hold
        # because vertical_cmd just shifts desired_pos[2] in the FSM.
        horizontal_cmd = not np.allclose(cmd_b[0:2], 0.0)

        if horizontal_cmd:
            # User-commanded velocity in body frame (FLU): x forward, y left
            v_max = self.vehicle_config.max_horizontal_velocity_mps
            vx_sp_body =  cmd_b[0] * v_max     # forward
            vy_sp_body = -cmd_b[1] * v_max     # lateral_cmd: +1 = right (NED-East),
                                               # FLU y is Left, so sign-flip
            # Rotate body -> world (FLU); only yaw matters for level flight
            yaw_rad = np.deg2rad(yaw_deg)
            cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
            vx_sp_flu = cy * vx_sp_body - sy * vy_sp_body
            vy_sp_flu = sy * vx_sp_body + cy * vy_sp_body
            # Reset position integrators so we don't wind up while moving
            self.pid_pos_x.reset(pos_x_flu)
            self.pid_pos_y.reset(pos_y_flu)
        else:
            self.pid_pos_x.set_desired(pos_x_sp_flu)
            self.pid_pos_y.set_desired(pos_y_sp_flu)
            vx_sp_flu = self.pid_pos_x.update(pos_x_flu, dt_sec)
            vy_sp_flu = self.pid_pos_y.update(pos_y_flu, dt_sec)

        # Z position loop is always active (FSM handles vertical_cmd by
        # adjusting desired_pos[2])
        self.pid_pos_z.set_desired(pos_z_sp_flu)
        vz_sp_flu = self.pid_pos_z.update(pos_z_flu, dt_sec)

        # Clip the velocity setpoint to the configured limits
        vx_sp_flu = float(np.clip(vx_sp_flu, -self.pos_vel_x_max, self.pos_vel_x_max))
        vy_sp_flu = float(np.clip(vy_sp_flu, -self.pos_vel_y_max, self.pos_vel_y_max))
        vz_sp_flu = float(np.clip(vz_sp_flu, -self.pos_vel_z_max, self.pos_vel_z_max))

        # ===============================================================
        # Middle loop:  Velocity -> roll/pitch + thrust  (FLU)
        # ===============================================================
        vx_flu =  vel_ned[0]
        vy_flu = -vel_ned[1]
        vz_flu = -vel_ned[2]

        self.pid_vel_x.set_desired(vx_sp_flu)
        self.pid_vel_y.set_desired(vy_sp_flu)
        self.pid_vel_z.set_desired(vz_sp_flu)
        u_vx = self.pid_vel_x.update(vx_flu, dt_sec)
        u_vy = self.pid_vel_y.update(vy_flu, dt_sec)
        u_vz = self.pid_vel_z.update(vz_flu, dt_sec)

        # In the firmware: the velocity-x PID output is an unsigned-ish
        # "tilt forward" demand that the position controller maps directly
        # into a pitch setpoint (negative pitch tilts forward in FLU,
        # because pitch is rotation about y-Left). The y output maps to
        # roll the same way.
        yaw_rad = np.deg2rad(yaw_deg)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        # Rotate world-frame velocity-PID outputs into the body frame so
        # roll/pitch commands stay consistent with the heading.
        u_fwd  =  cy * u_vx + sy * u_vy   # forward demand
        u_left = -sy * u_vx + cy * u_vy   # left demand

        # Tilt: +pitch (CF) noses up -> negative pitch to fly forward.
        pitch_sp_deg = -u_fwd
        # +roll (CF) tips right (about x-Forward, right-hand rule with z-Up
        # -> positive roll is right-wing-down). Rolling right means moving
        # in the -y_left direction, so to move +y_left we need negative roll.
        roll_sp_deg  = -u_left

        roll_sp_deg  = float(np.clip(roll_sp_deg,  -self.vel_roll_max,  self.vel_roll_max))
        pitch_sp_deg = float(np.clip(pitch_sp_deg, -self.vel_pitch_max, self.vel_pitch_max))

        thrust_pwm = self.vel_thrust_base + u_vz
        if thrust_pwm < self.vel_thrust_min:
            thrust_pwm = self.vel_thrust_min
        if thrust_pwm > self._THRUST_PWM_MAX:
            thrust_pwm = self._THRUST_PWM_MAX

        # ===============================================================
        # Attitude loop:  RPY angle -> body rate setpoint  (deg, deg/s)
        # ===============================================================
        self.pid_roll.set_desired(roll_sp_deg)
        self.pid_pitch.set_desired(pitch_sp_deg)
        self.pid_yaw.set_desired(self._yaw_desired_deg)

        roll_rate_sp  = self.pid_roll.update(roll_deg,   dt_sec, is_yaw_angle=False)
        pitch_rate_sp = self.pid_pitch.update(pitch_deg, dt_sec, is_yaw_angle=False)
        yaw_rate_sp   = self.pid_yaw.update(yaw_deg,     dt_sec, is_yaw_angle=True)

        # ===============================================================
        # Rate loop:  body rate -> raw torque commands
        # ===============================================================
        # In the CF firmware the gyro y-axis is negated when fed in
        # (`-sensors->gyro.y`) because the IMU mount differs from the
        # body-rate convention they want for pitch. Our omega comes from
        # the simulator already in the body frame matching RPY, so no
        # sign flip is needed here.
        self.pid_roll_rate.set_desired(roll_rate_sp)
        self.pid_pitch_rate.set_desired(pitch_rate_sp)
        self.pid_yaw_rate.set_desired(yaw_rate_sp)

        roll_out  = self.pid_roll_rate.update(p_dps,  dt_sec)
        pitch_out = self.pid_pitch_rate.update(q_dps, dt_sec)
        yaw_out   = self.pid_yaw_rate.update(r_dps,   dt_sec)

        # CF firmware then does `control->yaw = -control->yaw;` (again,
        # IMU/firmware convention mismatch). Skipped here for the same
        # reason as above.

        # int16 saturation, exactly like saturateSignedInt16
        roll_out  = float(np.clip(roll_out,  -32768.0, 32767.0))
        pitch_out = float(np.clip(pitch_out, -32768.0, 32767.0))
        yaw_out   = float(np.clip(yaw_out,   -32768.0, 32767.0))

        # ===============================================================
        # Power distribution (powerDistributionLegacy / X-config)
        # ===============================================================
        r_half = roll_out  * 0.5
        p_half = pitch_out * 0.5
        m1 = thrust_pwm - r_half + p_half + yaw_out
        m2 = thrust_pwm - r_half - p_half - yaw_out
        m3 = thrust_pwm + r_half - p_half + yaw_out
        m4 = thrust_pwm + r_half + p_half - yaw_out

        motors_pwm = np.array([m1, m2, m3, m4], dtype=float)
        # Normalize from PWM counts to simulation u in [umin, umax]
        u = motors_pwm / self._THRUST_PWM_MAX
        u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        return u, None

# ---------------------------------------------------------
# Threads
# ---------------------------------------------------------
def controller_thread_func(controller, vehicle_config):
    global g_thread_msgbox, g_thread_msgbox_lock, g_sim_running

    CTRL_DT_SEC = 1.0 / 100.0  # run the controller every 10 ms
    timestamp_last_ctrl_update = time.time() - 2 * CTRL_DT_SEC
    ctrl_step_counter = 0
    timestamp_last_fps_update = time.time()

    ctrl_time_max = 0.0
    ctrl_time_min = 999999.0
    ctrl_time_avg = 0.0
    ctrl_time_var = 0.0
    ctrl_time_avg_sample_count = 0

    while g_sim_running:
        timestamp_current = time.time()
        if timestamp_current - timestamp_last_ctrl_update < CTRL_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            keymap = copy.deepcopy(g_thread_msgbox["keymap"])
            state = copy.deepcopy(g_thread_msgbox["state"])

            if timestamp_current - timestamp_last_fps_update >= 1.0:
                g_thread_msgbox["ctrl_fps"] = ctrl_step_counter
                ctrl_step_counter = 0
                timestamp_last_fps_update = timestamp_current

                stats = controller.get_stats()
                g_thread_msgbox["ctrl_time_tot"] = stats.get("time_tot", np.array([0.0]))
                g_thread_msgbox["ctrl_sqp_iter"] = stats.get("sqp_iter", np.array([0]))
                g_thread_msgbox["ctrl_qp_iter"] = stats.get("qp_iter", np.array([0]))

                g_thread_msgbox["ctrl_time_max"] = ctrl_time_max
                g_thread_msgbox["ctrl_time_min"] = ctrl_time_min
                g_thread_msgbox["ctrl_time_avg"] = ctrl_time_avg
                g_thread_msgbox["ctrl_time_std"] = np.sqrt(ctrl_time_var)

        tic_timestamp = time.time()

        # Calculate control
        u, predictedX = controller.compute_control(state, keymap, CTRL_DT_SEC)

        toc_timestamp = time.time()
        ctrl_solve_time_s = toc_timestamp - tic_timestamp

        # Timing stats
        ctrl_time_avg_sample_count += 1
        ctrl_time_avg += (ctrl_solve_time_s - ctrl_time_avg) / ctrl_time_avg_sample_count
        ctrl_time_var += ((ctrl_solve_time_s - ctrl_time_avg) ** 2 - ctrl_time_var) / ctrl_time_avg_sample_count

        if ctrl_solve_time_s < ctrl_time_min: ctrl_time_min = ctrl_solve_time_s
        if ctrl_solve_time_s > ctrl_time_max: ctrl_time_max = ctrl_solve_time_s

        timestamp_last_ctrl_update = timestamp_current
        ctrl_step_counter += 1

        with g_thread_msgbox_lock:
            g_thread_msgbox["u"] = copy.deepcopy(u)
            if predictedX is not None:
                g_thread_msgbox["predictedX"] = copy.deepcopy(predictedX)


def sim_thread_func(env):
    global g_thread_msgbox, g_thread_msgbox_lock, g_sim_running

    TIMESTAMP_START = time.time()
    timestamp_lastupdate = TIMESTAMP_START
    MAX_DT_SEC = 0.1
    SIM_DT_SEC = 1.0 / 240.0
    sim_step_counter = 0
    last_fps_update = timestamp_lastupdate
    ctrl_fps = 0

    while g_sim_running:
        timestamp_current = time.time()
        dt_sec = timestamp_current - timestamp_lastupdate
        if dt_sec < SIM_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            if "u" in g_thread_msgbox:
                u = copy.deepcopy(g_thread_msgbox["u"])
            else:
                continue

        timestamp_lastupdate = timestamp_current
        if dt_sec > MAX_DT_SEC:
            print("Warning, high dt_sec: %.1f" % (dt_sec))
            continue
        dt_sec = np.clip(dt_sec, 0.0, MAX_DT_SEC)

        env.dt_sec = dt_sec
        state, reward, done, _ = env.step(u)
        sim_step_counter += 1

        if done == True:
            g_sim_running = False

        R_b_to_n, pos_n = env.get_render_info()
        with g_thread_msgbox_lock:
            g_thread_msgbox["R"] = copy.deepcopy(R_b_to_n)
            g_thread_msgbox["pos"] = copy.deepcopy(pos_n)
            g_thread_msgbox["state"] = copy.deepcopy(state)
            ctrl_fps = g_thread_msgbox["ctrl_fps"]
            render_fps = g_thread_msgbox["render_fps"]

        if timestamp_current - last_fps_update >= 1.0:
            print("FPS=%3i SIM=%4i CTRL=%3i" % (render_fps, sim_step_counter, ctrl_fps), end=" ")
            last_fps_update = timestamp_current
            sim_step_counter = 0

            print(
                "%.3fs (%6.2f,%6.2f,%6.2f)m (%5.1f,%5.1f,%5.1f)m/s φ=%5.1f° θ=%5.1f° ψ=%6.1f° ω(%6.1f,%5.1f,%6.1f)°/s γ(%9.3fNm,%9.3fNm,%9.3fNm,%8.2fN) u="
                % (
                    timestamp_current - TIMESTAMP_START,
                    env.pos_n[0], env.pos_n[1], env.pos_n[2],
                    env.vel_n[0], env.vel_n[1], env.vel_n[2],
                    env.roll_deg, env.pitch_deg, env.yaw_deg,
                    np.rad2deg(env.omega[0]), np.rad2deg(env.omega[1]), np.rad2deg(env.omega[2]),
                    env.gamma[0], env.gamma[1], env.gamma[2], env.gamma[3],
                ),
                end=" ",
            )
            for elem in u:
                print("%2.0f" % (elem * 99.0), end=" ")
            print("%")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    global g_thread_msgbox, g_thread_msgbox_lock, g_sim_running

    # Create the simulation environment
    env = MultirotorSimEnv(vehicle=0)
    g_thread_msgbox["state"] = env.state
    u = None
    predictedX = None

    # Get project root
    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent
    STL_DIR = PROJECT_ROOT / "stl"
    IMG_DIR = PROJECT_ROOT / "img"
    stl_file = STL_DIR / env.vehicle_config.model_file
    logo_file = IMG_DIR / "logo.png"

    if not stl_file.is_file():
        print("STL file not found: %s" % (stl_file))
        exit(1)
    else:
        print("Loading STL file: %s" % (stl_file))

    try:
        pygame.init()
        render = RenderStlPygame(stl_file, logo_file)
        render.init("multirotor_sim")
        clock = pygame.time.Clock()

        fps_freelook = False
        pygame.mouse.set_visible(not fps_freelook)
        pygame.event.set_grab(fps_freelook)

        cam_dist = np.max(env.vehicle_config.motortable) * 5.0
        cam_altitude = 0.7
        cam_pos_gl = np.array([-cam_dist, cam_altitude, 0.0])
        cam_yaw = 90.0
        cam_pitch = 0.0
        MOUSE_SENSITIVITY = 0.1
        MOVE_SPEED = 10.0

        render.render(
            np.eye(3), np.array([0.0, 0.0, -cam_altitude]),
            cam_pitch, cam_yaw, cam_pos_gl, env.vehicle_config, predictedX, u
        )
        renderer_active = True
    except Exception as e:
        print("OpenGL renderer init failed. Using console output.")
        traceback.print_exc()
        renderer_active = False
        pygame.quit()

    # ==========================================================
    # Controller Selection
    # ==========================================================
    ACTIVE_CONTROLLER = "NMPC" # For PID: change to "PID"

    if ACTIVE_CONTROLLER == "NMPC":
        print("Initializing NMPC Controller...")
        controller = NMPCController(env.vehicle_config)
    elif ACTIVE_CONTROLLER == "PID":
        print("Initializing PID Controller...")
        controller = PIDController(env.vehicle_config)
    else:
        raise ValueError(f"Unknown controller type: {ACTIVE_CONTROLLER}")
    # ==========================================================

    sim_logic_thread = threading.Thread(target=sim_thread_func, kwargs={"env": env})
    sim_logic_thread.start()

    ctrl_thread = threading.Thread(
        target=controller_thread_func,
        kwargs={"controller": controller, "vehicle_config": env.vehicle_config},
    )
    ctrl_thread.start()

    with g_thread_msgbox_lock:
        keymap = copy.deepcopy(g_thread_msgbox["keymap"])

    fps_counter = 0
    time_stamp_last_fps_count = time.time()
    last_render_time = time.time()

    while g_sim_running:
        current_time = time.time()
        dt_sec = current_time - last_render_time
        last_render_time = current_time

        with g_thread_msgbox_lock:
            R_b_to_n = copy.deepcopy(g_thread_msgbox["R"])
            pos_n = copy.deepcopy(g_thread_msgbox["pos"])
            g_thread_msgbox["keymap"] = keymap
            if "predictedX" in g_thread_msgbox:
                predictedX = copy.deepcopy(g_thread_msgbox["predictedX"])
            if "u" in g_thread_msgbox:
                u = copy.deepcopy(g_thread_msgbox["u"])

        fps_counter += 1
        if time.time() - time_stamp_last_fps_count >= 1.0:
            time_stamp_last_fps_count = time.time()
            with g_thread_msgbox_lock:
                g_thread_msgbox["render_fps"] = fps_counter
            fps_counter = 0

        if renderer_active == False:
            time.sleep(0.1)
            continue

        render.render(
            R_b_to_n, pos_n, cam_pitch, cam_yaw, cam_pos_gl, env.vehicle_config, predictedX, u
        )
        clock.tick(120)

        if fps_freelook:
            mouse_rel = pygame.mouse.get_rel()
            cam_yaw += mouse_rel[0] * MOUSE_SENSITIVITY
            cam_yaw %= 360.0
            cam_pitch -= mouse_rel[1] * MOUSE_SENSITIVITY
            cam_pitch = max(-89.0, min(89.0, cam_pitch))

        keys = pygame.key.get_pressed()
        keymap["lateral_cmd"] = 0.0
        keymap["lateral_cmd"] += keys[pygame.K_LEFT] * -1.0
        keymap["lateral_cmd"] += keys[pygame.K_RIGHT] * 1.0
        keymap["longitudinal_cmd"] = 0.0
        keymap["longitudinal_cmd"] += keys[pygame.K_DOWN] * -1.0
        keymap["longitudinal_cmd"] += keys[pygame.K_UP] * 1.0
        keymap["vertical_cmd"] = 0.0
        keymap["vertical_cmd"] += keys[pygame.K_j] * 1.0
        keymap["vertical_cmd"] += keys[pygame.K_k] * -1.0
        keymap["yaw_cmd"] = 0.0
        keymap["yaw_cmd"] += keys[pygame.K_h] * -1.0
        keymap["yaw_cmd"] += keys[pygame.K_l] * 1.0

        if keys[pygame.K_s]:
            cam_pos_gl[0] -= dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw))
            cam_pos_gl[2] += dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw))
        if keys[pygame.K_w]:
            cam_pos_gl[0] += dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw))
            cam_pos_gl[2] -= dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw))
        if keys[pygame.K_a]:
            cam_pos_gl[0] += dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw - 90.0))
            cam_pos_gl[2] -= dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw - 90.0))
        if keys[pygame.K_d]:
            cam_pos_gl[0] += dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw + 90.0))
            cam_pos_gl[2] -= dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw + 90.0))
        if keys[pygame.K_q]:
            cam_pos_gl[1] += dt_sec * MOVE_SPEED
        if keys[pygame.K_e]:
            cam_pos_gl[1] -= dt_sec * MOVE_SPEED

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                g_sim_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    g_sim_running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                fps_freelook = not fps_freelook
                pygame.mouse.set_visible(not fps_freelook)
                pygame.event.set_grab(fps_freelook)
                mouse_rel = pygame.mouse.get_rel()
                if fps_freelook:
                    print("Freelook active, press mousebutton to exit mode")
                else:
                    print("Freelook disabled")

    sim_logic_thread.join()
    ctrl_thread.join()

if __name__ == "__main__":
    main()

