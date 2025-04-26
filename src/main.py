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
from pathlib import Path  # figure out path of .stl files
from geodetic_toolbox import *
from multirotorsimulatorenv import MultirotorSimEnv # Physic simulation
from visualization3d import RenderStlPygame # 3D visualization
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, sqrt, sumsqr
from mpc_copter.copter_model_position import export_copterpos_ode_model

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

# VehicleControlState is a helper class to store the vehicle control state and
# the desired position. Managed inside the NMPC thread.
class VehicleControlState:
    def __init__(self):
        self.mode = "HOLD"        # MOVE, BRAKE, HOLD
        self.stop_timer = 0.0     # timer to stop the vehicle
        self.desired_pos = None   # 3D position to hold
        self.yaw_ref = None       # integrated yaw

# This thread's job is to consume the state vector and emit a control output u
def nmpc_thread_func(vehicle_config, initial_state):
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    # Keep track of e.g. the position hold setpoint:
    vehicle_control_state = VehicleControlState()

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
    ocp.solver_options.N_horizon = N_horizon

    # set cost module
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q_mat = np.diag(model.weight_diag)  # weights on state vector for costs
    R_mat = np.diag(
        np.ones(nu,) * model.cost_u_weight
    )  # weight on control input u
    # R_mat[0,0] = 1e1 # example weight adjustment: be easy on motor #0

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat
    np.set_printoptions(precision=3, suppress=True, linewidth=400)
    print("Weights: ", end="")
    print(np.diag(Q_mat))
    print("Weights on control input:", end="")
    print(np.diag(R_mat))

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

    print("Initial state setpoint: ", end="")
    print(setpoint_yref)

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

    enable_soft_constraints = True
    if enable_soft_constraints:
        # <Soft Constraints>
        # 1. Define which state indices to constrain
        constrained_state_indices = list(range(
            vehicle_config.state_cfg["omega_index"],
            vehicle_config.state_cfg["omega_index_end"]
        ))
        ocp.constraints.idxbx_e = np.array(constrained_state_indices)

        # 2. Set hard bounds for these states
        max_rotation_rate_rps = vehicle_config.max_rotation_rate_rps
        print("Max. rotation rate soft constraint: %.1f deg/s" % (np.rad2deg(max_rotation_rate_rps)))
        ocp.constraints.lbx_e = np.array([-max_rotation_rate_rps] * len(constrained_state_indices))
        ocp.constraints.ubx_e = np.array([ max_rotation_rate_rps] * len(constrained_state_indices))

        # 3. Make all these bounds soft
        ocp.constraints.idxsbx_e = np.arange(len(constrained_state_indices))
        ns = len(constrained_state_indices)  # number of softened bounds
        ocp.constraints.lsbx_e = np.zeros((ns,))
        ocp.constraints.usbx_e = 1e2 * np.ones((ns,))

        # 4. Set slack penalties for soft constraints
        L2_penalty = 1.0  # quadratic penalty (Z terms)
        L1_penalty = 0.0  # linear penalty (z terms, often zero)

        ocp.cost.Zl_e = L2_penalty * np.ones((ns,))
        ocp.cost.Zu_e = L2_penalty * np.ones((ns,))
        ocp.cost.zl_e = L1_penalty * np.ones((ns,))
        ocp.cost.zu_e = L1_penalty * np.ones((ns,))
        # </Soft Constraints>

    # Solver options
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
        vehicle_control(acados_ocp_solver, ocp, state, keymap, vehicle_config,
                        vehicle_control_state, MPC_DT_SEC)

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
            # emit current attitude body to n-frame for render thread
            g_thread_msgbox["R"] = copy.deepcopy(R_b_to_n)
            g_thread_msgbox["pos"] = copy.deepcopy(pos_n)  # emit current pos in NED for render thread
            g_thread_msgbox["state"] = copy.deepcopy(state)  # emit current state vector for NMPC thread
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
    env = MultirotorSimEnv()
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

def update_vehicle_control_state(ctrl_state, cmd_b, current_pos, current_vel, q, dt_sec):
    # helper function for vehicle_control function
    # updating the vehicle control state machine

    vel_mag = np.linalg.norm(current_vel)
    stopping = np.allclose(cmd_b, 0.0)
    slow = vel_mag < 0.5  # m/s threshold
    # is a vertical command requested?
    vertical_cmd = not np.isclose(cmd_b[2], 0.0)
    yaw = quat_to_rpy(q)[2]

    if ctrl_state.yaw_ref is None:
        ctrl_state.yaw_ref = yaw
    if ctrl_state.desired_pos is None:
        ctrl_state.desired_pos = current_pos.copy()

    if vertical_cmd:
        # if a vertical command is requested, then set the desired altitude to
        # the current altitude
        ctrl_state.desired_pos[2] = current_pos[2]

    # if yaw_ref deviates more than 10 degrees from the current yaw
    # then set the yaw_ref to the current yaw:
    # if np.abs(angle_diff(yaw, ctrl_state.yaw_ref)) > np.deg2rad(10.0):
    #     ctrl_state.yaw_ref = yaw

    elif ctrl_state.mode == "MOVE":
        ctrl_state.desired_pos[0] = current_pos[0]
        ctrl_state.desired_pos[1] = current_pos[1]
        if stopping:
            ctrl_state.mode = "BRAKE"
            ctrl_state.stop_timer = 0.0
            print("braking...")

    elif ctrl_state.mode == "BRAKE":
        ctrl_state.desired_pos[0] = current_pos[0]
        ctrl_state.desired_pos[1] = current_pos[1]
        if not stopping:
            ctrl_state.mode = "MOVE"
        elif slow:
            ctrl_state.stop_timer += dt_sec
            if ctrl_state.stop_timer > 0.75:
                ctrl_state.mode = "HOLD"
                print("HOLD vehicle to position: %.2f %.2f %.2f" % (ctrl_state.desired_pos[0], ctrl_state.desired_pos[1], ctrl_state.desired_pos[2]))
        else:
            ctrl_state.stop_timer = 0.0

    elif ctrl_state.mode == "HOLD":
        if not stopping:
            ctrl_state.mode = "MOVE"

def vehicle_control(acados_ocp_solver, ocp, state, keymap, vehicle_config,
                    ctrl_state, dt_sec):
    """
    Generate consistent NMPC references based on user keymap input and FSM.
    Sets: position, velocity, attitude (quat), angular rate.
    :param acados_ocp_solver: acados solver object
    :param ocp: acados ocp object
    :param state: current state vector
    :param keymap: user input keymap
    :param vehicle_config: vehicle configuration with vehicle parameters
    :param ctrl_state: vehicle control state class VehicleControlState
    :param dt_sec: time step in seconds (MPC update rate)
    """
    cfg = vehicle_config.state_cfg

    # Extract position, velocity etc. from state vector
    pos = state[cfg["pos3d_index"]:cfg["pos3d_index_end"]]
    vel = state[cfg["vel3d_index"]:cfg["vel3d_index_end"]]
    q   = state[cfg["q_index"]:cfg["q_index_end"]]
    yaw = quat_to_rpy(q)[2]

    # Body-frame user/pilot input
    cmd_b = np.array([
        keymap.get("longitudinal_cmd", 0.0),
        keymap.get("lateral_cmd", 0.0),
        keymap.get("vertical_cmd", 0.0),
    ])
    yaw_cmd = keymap.get("yaw_cmd", 0.0)

    update_vehicle_control_state(ctrl_state, cmd_b, pos, vel, q, dt_sec)

    # Integrate yaw
    max_yaw_rate = vehicle_config.max_rotation_rate_rps # rad/s
    yaw_rate_ref = yaw_cmd * max_yaw_rate
    ctrl_state.yaw_ref += yaw_rate_ref * dt_sec

    # Velocity in nav frame
    if np.allclose(cmd_b, 0.0):
        vel_n_ref = np.zeros(3)
    else:
        v_b = np.array([
            cmd_b[0] * vehicle_config.max_horizontal_velocity_mps,
            cmd_b[1] * vehicle_config.max_horizontal_velocity_mps,
            0.0,
        ])
        R_yaw = quat_to_matrix(quat_from_rpy(0.0, 0.0, yaw))
        vel_n_ref = R_yaw @ v_b
        vel_n_ref[2] = cmd_b[2] * vehicle_config.max_vertical_velocity_mps

    # Compute attitude for velocity
    m = vehicle_config.mass_kg
    g = vehicle_config.gravity_n[2]
    v_norm = np.hypot(vel_n_ref[0], vel_n_ref[1])
    c_D = vehicle_config.windresistance
    F_x = c_D * vel_n_ref[0] * v_norm
    F_y = c_D * vel_n_ref[1] * v_norm
    roll_ref = np.arctan(F_y / (m * g))
    pitch_ref = -np.arctan(F_x / (m * g))
    q_ref = quat_from_rpy(roll_ref, pitch_ref, ctrl_state.yaw_ref)
    omega_ref = np.array([0.0, 0.0, yaw_rate_ref])

    # Reference position
    pos_ref = ctrl_state.desired_pos

    # Build yref vector
    yref = np.copy(ocp.cost.yref)
    yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_ref
    yref[cfg["vel3d_index"]:cfg["vel3d_index_end"]] = vel_n_ref
    yref[cfg["q_index"]:cfg["q_index_end"]] = q_ref
    yref[cfg["omega_index"]:cfg["omega_index_end"]] = omega_ref
    ocp.cost.yref = np.copy(yref)

    N = ocp.solver_options.N_horizon # N = prediction horizon epochs
    pos_pred = np.zeros((N + 1, 3)) # future positions
    pos_pred[0] = pos_ref
    for j in range(1, N + 1):
        pos_pred[j] = pos_pred[j - 1] + vel_n_ref * dt_sec

    # Push to acados
    for j in range(N):
        # if j==0 and (not np.allclose(cmd_b, 0.0) or yaw_cmd != 0.0):
        #      print(yref)
        #      print("pos_ref=%.2f %.2f %.2f" % (pos_ref[0], pos_ref[1], pos_ref[2]), end=" ")
        #      print("vel_ref=%.2f %.2f %.2f" % (vel_n_ref[0], vel_n_ref[1], vel_n_ref[2]), end="")
        #      print("roll_ref=%.2f pitch_ref=%.2f yaw_ref=%.2f" % (np.rad2deg(roll_ref), np.rad2deg(pitch_ref), np.rad2deg(ctrl_state.yaw_ref)), end="")
        #      print("")
        yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[j]
        acados_ocp_solver.set(j, "yref", np.copy(yref))

    yref_e = np.copy(yref[:state.size])
    yref_e[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[N]
    acados_ocp_solver.set(N, "yref", yref_e)
    ocp.cost.yref_e = yref_e

if __name__ == "__main__":
    main()

