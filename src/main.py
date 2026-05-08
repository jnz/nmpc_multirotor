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

class PIDController(BaseController):
    """
    Cascaded PID + Altitude PID Controller scaffold.
    """
    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        # TODO: Init PID gains here
        # self.kp_pos = np.array([...])
        # self.kd_pos = np.array([...])
        # ...

    def compute_control(self, state, keymap, dt_sec):
        cfg = self.vehicle_config.state_cfg
        pos = state[cfg["pos3d_index"]:cfg["pos3d_index_end"]]
        vel = state[cfg["vel3d_index"]:cfg["vel3d_index_end"]]
        q   = state[cfg["q_index"]:cfg["q_index_end"]]

        cmd_b = np.array([
            keymap.get("longitudinal_cmd", 0.0),
            keymap.get("lateral_cmd", 0.0),
            keymap.get("vertical_cmd", 0.0),
        ])

        # 1. Update setpoints via FSM
        update_vehicle_control_state(self.ctrl_state, cmd_b, pos, vel, q, dt_sec)

        # TODO: Implement Outer Loop (Position -> Target Velocity)
        # TODO: Implement Middle Loop (Velocity -> Target Attitude/Thrust)
        # TODO: Implement Altitude PID (Z Position -> Thrust)
        # TODO: Implement Inner Loop (Attitude -> Motor Commands 'u')

        # Placeholder for 4 motors, hovering/idle
        u = np.ones(4) * self.vehicle_config.umin

        predictedX = None # No prediction horizon in basic PID

        return u, predictedX

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

