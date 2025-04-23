#!/usr/bin/env python3

"""
Main runtime file to watch the simulation environment and vehicle in
real-time.

(c) Jan Zwiener (jan@zwiener.org)
"""

from pathlib import Path  # figure out the path of this file
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import scipy.linalg
import time
from multirotorenv import MultirotorEnv
from geodetic_toolbox import *
import pygame
from render_stl import RenderStlPygame
import threading
import copy
from mpc_copter.copter_model_position import export_copterpos_ode_model
from casadi import SX, vertcat, cos, sin, sqrt, sumsqr

# nmpc_multirotor (main.py)
# -------------------------
#
# Simulation entry point
#
# Block diagram:
#
#      ┌────────────────────┐
#      │                    │
#      │                    │      Rotation 'R', Position 'pos' to render object
#      │  Render Thread     │◄─────────────────────────┐
#      │  Input Thread      │                          │
#      │  main()            │                          │
#      │                    │    'render_fps' ┌────────┴───────────┐
#      │                    ├────────────────►│                    │
#      └───────┬────────────┘                 │                    │
#              │                              │   Simulation       │
#              │ User keyboard input          │   Thread           │
#              │ 'keymap'                     │   sim_thread_func()│
#              │                              │                    │
#              ▼                              │                    │
#      ┌─────────────────────┐                │                    │
#      │                     │                └────────┬───────────┘
#      │                     │                         │       ▲
#      │  NMPC Thread        │                         │       │
#      │  nmpc_thread_func() │◄────────────────────────┘       │
#      │                     │   'state' vector                │
#      │                     │                                 │
#      │                     │                                 │
#      │                     ├─────────────────────────────────┘
#      └─────────────────────┘       Control output 'u'
#                                    'mpc_fps', 'nmpc_time_tot', 'nmpc_sqp_iter', 'nmpc_qp_iter'
#
#
# Global messagebox to exchange data between threads as shown above
g_thread_msgbox = {
    "R": np.identity(3),  # object attitude (rotation matrix body to navigation frame)
    "pos": np.array([0.0, 0.0, 0.0]),  # object position in NED 'pos_n'
    "keymap": {
        "longitudinal_cmd": 0.0,
        "lateral_cmd": 0.0,
        "yaw_cmd": 0.0,
        "vertical_cmd": 0.0,
    },  # keyboard input
    "mpc_fps": 0,  # debug information: fps of NMPC thread
    "render_fps": 0,  # debug information: fps of render thread
    "nmpc_time_tot": np.array([0.0]),  # debug/timing information on NMPC calculation
    "nmpc_sqp_iter": np.array([0]),  # debug information on NMPC calculation
    "nmpc_qp_iter": np.array([0]),  # debug information on NMPC calculation
    "nmpc_time_max": 0.0,  # debug information on NMPC: worst case time for NMPC solver in seconds
    "nmpc_time_min": 0.0,  # debug information on NMPC: lowest frame time seen in seconds
    "nmpc_time_avg": 0.0,  # debug information on NMPC: average frame time in seconds
    "nmpc_time_std": 0.0,  # debug information on NMPC: standard deviation
    # state                            # current state vector from simulation thread
    # u                                # control input from NMPC to simulation thread
    # predictedX                       # predicted state over the MPC horizon
}
g_thread_msgbox_lock = threading.Lock()  # only access g_thread_msgbox with this lock
g_sim_running = True  # Run application as long as this is set to True


# This thread's job is to consume the state vector and emit a control output u
def nmpc_thread_func(vehicle_config, initial_state):
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    ocp = AcadosOcp()  # create ocp object to formulate the OCP

    # model = export_copter_ode_model()
    model = export_copterpos_ode_model(vehicle_config)
    ocp.model = model

    Tf = 3.0  # Time horizon in seconds
    nx = model.x.size()[0]  # state length
    nu = model.u.size()[0]  # control input u vector length
    ny = nx + nu
    ny_e = nx
    N_horizon = int(20 * Tf)  # Epochs for MPC prediction horizon
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q_mat = np.diag(model.weight_diag)  # state weight
    R_mat = np.diag(
        np.ones( nu,) * 1e-1
    )  # weight on control input u
    # R_mat[0,0] = 1e1 # example weight adjustment: be easy on motor #0

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    x0 = initial_state

    setpoint_yref = np.zeros((ny,))
    # Target unit quaternion (identity orientation): [1, 0, 0, 0]
    setpoint_yref[vehicle_config.state_cfg["q_index"]] = (
        1.0  # set q0 (real) unit quaternion part to 1.0
    )
    setpoint_yref[vehicle_config.state_cfg["altitude_index"]] = -0.7
    ocp.cost.yref = setpoint_yref  # np.zeros((ny, ))    # setpoint trajectory
    ocp.cost.yref_e = setpoint_yref[0:nx]  # np.zeros((ny_e, ))  # setpoint end

    ocp.constraints.constr_type = (
        "BGH"  # Comprises simple bounds, polytopic constraints, general non-linear constraints.
    )

    # control input constraints
    ocp.constraints.lbu = (
        np.ones((vehicle_config.motorcount,)) * vehicle_config.umin**2
    )  # the MPC works with u squared
    ocp.constraints.ubu = (
        np.ones((vehicle_config.motorcount,)) * vehicle_config.umax**2
    )
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array(range(nu))

    # State soft-constraints
    nbx = 5  # 3 constraints on the rotation rate, 2 on quaternion
    Jbx = np.zeros((nbx, nx))
    Jbx[
        0:3, vehicle_config.state_cfg["omega_index"] : vehicle_config.state_cfg["omega_index_end"]
    ] = np.eye(3)
    Jbx[3, vehicle_config.state_cfg["q_index"] + 1] = 1
    Jbx[4, vehicle_config.state_cfg["q_index"] + 2] = 1
    max_rotation_rate_rps = vehicle_config.max_rotation_rate_rps
    ocp.constraints.Jbx = Jbx
    ocp.constraints.lbx = np.zeros((nbx, 0))
    ocp.constraints.ubx = np.zeros((nbx, 0))
    ocp.constraints.lbx[0] = -max_rotation_rate_rps
    ocp.constraints.lbx[1] = -max_rotation_rate_rps
    ocp.constraints.lbx[2] = -max_rotation_rate_rps
    ocp.constraints.ubx[0] = max_rotation_rate_rps
    ocp.constraints.ubx[1] = max_rotation_rate_rps
    ocp.constraints.ubx[2] = max_rotation_rate_rps
    ocp.constraints.lbx[3] = -0.3  # not the best way to limit the tilt angle, but linear
    ocp.constraints.ubx[3] = 0.3
    ocp.constraints.lbx[4] = -0.3
    ocp.constraints.ubx[4] = 0.3
    ocp.constraints.Jsbx = np.eye(nbx)  # enable soft constraints

    # Obstacle soft-constraints
    num_obstacles = 0 # disable obstacles test code below
    nsh = num_obstacles  # nsh = number of slack variables for the h-expression
    if num_obstacles > 0:
        con_expr = SX.sym("con_expr", num_obstacles, 1)
        obstacle_radius = np.array([20.0, 20.0])
        pos3d_index = vehicle_config.state_cfg["pos3d_index"]
        pos3d_index_e = vehicle_config.state_cfg["pos3d_index_end"]
        con_expr[0] = sqrt(
            sumsqr(model.x[pos3d_index:pos3d_index_e] - np.array([40.0, -10.0, 0.0]))
        )  # obstacle position 1
        con_expr[1] = sqrt(
            sumsqr(model.x[pos3d_index:pos3d_index_e] - np.array([100.0, 10.0, 0.0]))
        )  # obstacle position 2
        ocp.model.con_h_expr = con_expr
        ocp.model.con_h_expr_e = con_expr
        ocp.constraints.lh = obstacle_radius
        ocp.constraints.uh = 1e9 * np.ones((num_obstacles,))  # make it a very large number
        ocp.constraints.lh_e = obstacle_radius
        ocp.constraints.uh_e = 1e9 * np.ones((num_obstacles,))  # make it a very large number
        ocp.constraints.lsh = np.zeros(nsh)
        ocp.constraints.ush = np.zeros(nsh)
        ocp.constraints.idxsh = np.array(range(nsh))

    # slack variables
    ns = nbx + nsh  # total number of soft constraints (slack)
    L2_pen = 10e3  # Least squares penalty for slack variables
    L1_pen = 0.0  # L1-norm penalty (set to zero to disable)
    ocp.cost.Zl = L2_pen * np.ones((ns,))
    ocp.cost.Zu = L2_pen * np.ones((ns,))
    ocp.cost.zl = L1_pen * np.ones((ns,))
    ocp.cost.zu = L1_pen * np.ones((ns,))

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"  # IRK, GNSF, ERK
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP or SQP_RTI

    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf

    solver_json = "acados_ocp_" + model.name + ".json"
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)
    # create an integrator with the same settings as used in the OCP solver.
    # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    # make sure a MPC update is performed in the first epoch
    MPC_DT_SEC = 1.0 / 100.0  # run the NMPC every XX ms
    timestamp_last_mpc_update = time.time() - 2 * MPC_DT_SEC
    mpc_step_counter = 0  # +1 for every mpc step, reset every 1 sec
    timestamp_last_mpc_fps_update = time.time()

    # debug timing
    nmpc_time_max = 0.0
    nmpc_time_min = 999999.0
    nmpc_time_avg = 0.0
    nmpc_time_std = 0.0
    nmpc_time_var = 0.0  # variance to calculate the std.-dev.
    nmpc_time_avg_sample_count = 0

    predictedX = np.ndarray((N_horizon, nx))

    while g_sim_running:
        timestamp_current = time.time()
        if timestamp_current - timestamp_last_mpc_update < MPC_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            keymap = copy.deepcopy(g_thread_msgbox["keymap"])  # read input from render thread
            state = copy.deepcopy(g_thread_msgbox["state"])  # fetch current state vector
            if timestamp_current - timestamp_last_mpc_fps_update >= 1.0:
                g_thread_msgbox["mpc_fps"] = mpc_step_counter
                mpc_step_counter = 0
                timestamp_last_mpc_fps_update = timestamp_current
                g_thread_msgbox["nmpc_time_tot"] = acados_ocp_solver.get_stats("time_tot")
                g_thread_msgbox["nmpc_sqp_iter"] = acados_ocp_solver.get_stats("sqp_iter")
                g_thread_msgbox["nmpc_qp_iter"] = acados_ocp_solver.get_stats("qp_iter")
                g_thread_msgbox["nmpc_time_max"] = nmpc_time_max
                g_thread_msgbox["nmpc_time_min"] = nmpc_time_min
                g_thread_msgbox["nmpc_time_avg"] = nmpc_time_avg
                g_thread_msgbox["nmpc_time_std"] = nmpc_time_std

        if mpc_step_counter == 0:
            omega = np.array(
                [
                    state[vehicle_config.state_cfg["omega_roll_index"]],
                    state[vehicle_config.state_cfg["omega_pitch_index"]],
                    state[vehicle_config.state_cfg["omega_yaw_index"]],
                ]
            )

            if (
                np.abs(omega[0]) > 1.2 * vehicle_config.max_rotation_rate_rps
                or np.abs(omega[1]) > 1.2 * vehicle_config.max_rotation_rate_rps
                or np.abs(omega[2]) > 1.2 * vehicle_config.max_rotation_rate_rps
            ):
                print(
                    "WARNING rotation rate too high: %.1f %.1f %.1f deg/s"
                    % (np.rad2deg(omega[0]), np.rad2deg(omega[1]), np.rad2deg(omega[2]))
                )

        # Process user input from keymap to update the reference trajectory in
        # the NMPC solver (acados_ocp_solver's yref trajectory and costs in
        # ocp.cost.yref_e)
        vehicle_control(acados_ocp_solver, ocp, state, keymap, vehicle_config)

        tic_timestamp = time.time()
        # solve OCP and get next control input
        u = acados_ocp_solver.solve_for_x0(x0_bar=state)
        # If the solver calculates u^2 commands, we need to take the
        # square root of the result to get the actual control input
        if model.ctrlout_u_is_squared:
            u = np.sqrt(u)

        toc_timestamp = time.time()
        mpc_solve_time_s = toc_timestamp - tic_timestamp
        nmpc_time_avg_sample_count += 1
        nmpc_time_avg += (mpc_solve_time_s - nmpc_time_avg) / nmpc_time_avg_sample_count
        nmpc_time_var += (
            (mpc_solve_time_s - nmpc_time_avg) ** 2 - nmpc_time_var
        ) / nmpc_time_avg_sample_count
        nmpc_time_std = np.sqrt(nmpc_time_var)
        if mpc_solve_time_s < nmpc_time_min:
            nmpc_time_min = mpc_solve_time_s
        if mpc_solve_time_s > nmpc_time_max:
            nmpc_time_max = mpc_solve_time_s

        # get predicted states (x) from solver
        for i in range(N_horizon):
            predictedX[i, :] = acados_ocp_solver.get(i, "x")

        timestamp_last_mpc_update = timestamp_current
        mpc_step_counter += 1
        with g_thread_msgbox_lock:
            g_thread_msgbox["u"] = copy.deepcopy(u)
            g_thread_msgbox["predictedX"] = copy.deepcopy(predictedX)


# This thread's job is it to simulate the world (physic simulation) and in
# particular to emit the current state vector
def sim_thread_func(env):
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    TIMESTAMP_START = time.time()
    timestamp_lastupdate = TIMESTAMP_START
    MAX_DT_SEC = 0.1  # don't allow larger simulation timesteps than this
    SIM_DT_SEC = 1.0 / 240.0  # run the simulation every XX ms
    sim_step_counter = 0  # +1 for every simulation step, reset every 1 sec
    # emit a FPS stat message every second based on this timestamp:
    last_fps_update = timestamp_lastupdate
    mpc_fps = 0

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
        # wait with the simulation thread until we receive the first u vector
        # e.g. the MPC thread needs to compile .c files until it is ready

        timestamp_lastupdate = timestamp_current
        # make sure dt_sec is within a reasonable range
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
            g_thread_msgbox["R"] = (
                R_b_to_n  # emit current attitude body to n-frame for render thread
            )
            g_thread_msgbox["pos"] = pos_n  # emit current pos in NED for render thread
            g_thread_msgbox["state"] = state  # emit current state vector for NMPC thread
            mpc_fps = g_thread_msgbox["mpc_fps"]
            render_fps = g_thread_msgbox["render_fps"]
            nmpc_time_tot = g_thread_msgbox["nmpc_time_tot"]
            nmpc_sqp_iter = g_thread_msgbox["nmpc_sqp_iter"]
            nmpc_qp_iter = g_thread_msgbox["nmpc_qp_iter"]
            nmpc_time_max = g_thread_msgbox["nmpc_time_max"]
            nmpc_time_min = g_thread_msgbox["nmpc_time_min"]
            nmpc_time_avg = g_thread_msgbox["nmpc_time_avg"]
            nmpc_time_std = g_thread_msgbox["nmpc_time_std"]

        if timestamp_current - last_fps_update >= 1.0:
            print("FPS=%3i SIM=%4i MPC=%3i" % (render_fps, sim_step_counter, mpc_fps), end=" ")
            last_fps_update = timestamp_current

            sum_power_mech_kw = np.sum(env.motor_power_mech_kw)
            sim_step_counter = 0
            print(
                "%9.3fs KW=%5.2f NED=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) gamma=(%7.3f,%7.3f,%6.3f,%7.1f) u="
                % (
                    timestamp_current - TIMESTAMP_START,
                    sum_power_mech_kw,
                    env.pos_n[0],
                    env.pos_n[1],
                    env.pos_n[2],
                    env.vel_n[0],
                    env.vel_n[1],
                    env.vel_n[2],
                    env.roll_deg,
                    env.pitch_deg,
                    env.yaw_deg,
                    np.rad2deg(env.omega[0]),
                    np.rad2deg(env.omega[1]),
                    np.rad2deg(env.omega[2]),
                    env.gamma[0],
                    env.gamma[1],
                    env.gamma[2],
                    env.gamma[3],
                ),
                end=" ",
            )
            for elem in u:
                print("%2.0f" % (elem * 99.0), end=" ")
            print("")


def main():
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    # Create the simulation environment
    env = MultirotorEnv()
    g_thread_msgbox["state"] = env.state
    u = None
    predictedX = None

    # Get project root by walking up from this file
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
        render.init("nmpc_multirotor")
        clock = pygame.time.Clock()

        fps_freelook = False
        # Camera Control
        # Allow FPS style camera movement, capture mouse to do this
        pygame.mouse.set_visible(not fps_freelook)
        pygame.event.set_grab(fps_freelook)
        # Camera variables
        cam_dist = (
            np.max(env.vehicle_config.motortable) * 5.0
        )  # try to position camera based on rotor plane size
        cam_altitude = 0.7
        cam_pos_gl = np.array(
            [-cam_dist, cam_altitude, 0.0]
        )  # camera position in OpenGL system (not! North/East/Down)
        cam_yaw = 90.0  # turn do look down OpenGL x-axis (right) / North
        cam_pitch = 0.0
        MOUSE_SENSITIVITY = 0.1
        MOVE_SPEED = 10.0

        render.render(
            np.eye(3),
            np.array([0.0, 0.0, -cam_altitude]),
            cam_pitch,
            cam_yaw,
            cam_pos_gl,
            env.vehicle_config,
            predictedX,
            u,
        )
        renderer_active = True
    except Exception as e:
        print("OpenGL renderer init failed. Using console output.")
        renderer_active = False
        pygame.quit()

    # Spawn simulation thread
    sim_logic_thread = threading.Thread(target=sim_thread_func, kwargs={"env": env})
    sim_logic_thread.start()

    # Spawn NMPC thread
    nmpc_thread = threading.Thread(
        target=nmpc_thread_func,
        kwargs={"vehicle_config": env.vehicle_config, "initial_state": g_thread_msgbox["state"]},
    )
    nmpc_thread.start()

    with g_thread_msgbox_lock:
        keymap = copy.deepcopy(g_thread_msgbox["keymap"])

    fps_counter = 0
    time_stamp_last_fps_count = time.time()
    last_render_time = time.time()

    while g_sim_running:

        current_time = time.time()
        dt_sec = current_time - last_render_time
        last_render_time = current_time

        # Get orientation R and position pos to render the object
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

        keys = pygame.key.get_pressed()  # Get the state of all keyboard buttons
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

        # Move forward (along the camera's view direction)
        if keys[pygame.K_s]:
            cam_pos_gl[0] -= dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw))
            cam_pos_gl[2] += dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw))
        # Move backward
        if keys[pygame.K_w]:
            cam_pos_gl[0] += dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw))
            cam_pos_gl[2] -= dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw))
        # Move left
        if keys[pygame.K_a]:
            cam_pos_gl[0] += dt_sec * MOVE_SPEED * np.sin(np.deg2rad(cam_yaw - 90.0))
            cam_pos_gl[2] -= dt_sec * MOVE_SPEED * np.cos(np.deg2rad(cam_yaw - 90.0))
        # Move right
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
                mouse_rel = pygame.mouse.get_rel()  # consume one get_rel to avoid jump
                if fps_freelook:
                    print("Freelook active, press mousebutton to exit mode")
                else:
                    print("Freelook disabled")

    sim_logic_thread.join()
    nmpc_thread.join()


def vehicle_control(acados_ocp_solver, ocp, state, keymap, vehicle_config):
    # Set a new reference trajectory for the NMPC based
    # on the keymap input (pressed keys)

    # FIXME: this needs a rework. The reference trajectory
    # is not properly setup

    if (
        not vehicle_config.state_cfg["pos3d_available"]
        or not vehicle_config.state_cfg["vel3d_available"]
    ):
        return

    if (
        keymap["lateral_cmd"] != 0.0
        or keymap["longitudinal_cmd"] != 0.0
        or keymap["vertical_cmd"] != 0.0
    ):
        position_change_requested = True
    else:
        position_change_requested = False

    state_cfg = vehicle_config.state_cfg
    current_pos = state[state_cfg["pos3d_index"] : state_cfg["pos3d_index_end"]]
    current_vel = state[state_cfg["vel3d_index"] : state_cfg["vel3d_index_end"]]
    current_q = state[state_cfg["q_index"] : state_cfg["q_index_end"]]

    N_horizon = ocp.dims.N
    Tf = ocp.solver_options.tf
    dt_mpc = Tf / N_horizon

    yrefNew = np.copy(ocp.cost.yref)

    setpoint_vel_mps = 0.0
    if position_change_requested:
        yrefNew[state_cfg["pos3d_index"] : state_cfg["pos3d_index_end"]] = (
            current_pos  # set current position as new setpoint position
        )
        rpy = quat_to_rpy(current_q)  # could also be the reference attitude?
        qnew = quat_from_rpy(0.0, 0.0, rpy[2])
        R = quat_to_matrix(qnew)
        setpoint_vel_mps = 25.0  # max. velocity
        v_b = np.array([keymap["longitudinal_cmd"], keymap["lateral_cmd"], keymap["vertical_cmd"]])
        v_n = setpoint_vel_mps * R @ v_b  # setpoint velocity in n-frame
        yrefNew[state_cfg["vel3d_index"] : state_cfg["vel3d_index_end"]] = v_n  # setpoint velocity
    else:
        yrefNew[state_cfg["vel3d_index"] : state_cfg["vel3d_index_end"]] = np.array(
            [0.0, 0.0, 0.0]
        )  # reset velocity

    ocp.cost.yref = yrefNew
    for j in range(N_horizon):
        acados_ocp_solver.set(j, "yref", np.copy(yrefNew))

    nx = state.size
    yref_N_new = np.copy(yrefNew[0:nx])
    acados_ocp_solver.set(N_horizon, "yref", yref_N_new)
    ocp.cost.yref_e = yref_N_new

if __name__ == "__main__":
    main()
