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
from pathlib import Path  # to figure out path of .stl files

from geodetic_toolbox import *
from multirotorsimulatorenv import MultirotorSimEnv # Physic simulation
from visualization3d import RenderStlPygame # 3D visualization

# NMPC specific imports
from pid_controller import PIDController
from mpc_controller import NMPCController
from rate_mpc_controller import RateMPCController

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
    env = MultirotorSimEnv(vehicle=21)
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
    ACTIVE_CONTROLLER = "PID"

    if ACTIVE_CONTROLLER == "NMPC":
        print("Initializing Full NMPC Controller...")
        controller = NMPCController(env.vehicle_config)
    elif ACTIVE_CONTROLLER == "PID":
        print("Initializing PID Controller...")
        controller = PIDController(env.vehicle_config)
    elif ACTIVE_CONTROLLER == "RATE_MPC":
        print("Initializing Rate-MPC Controller...")
        controller = RateMPCController(env.vehicle_config)
    else:
        raise ValueError(f"Unknown controller type: {ACTIVE_CONTROLLER}")
    # ==========================================================
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

