"""
NMPC model definition for acados — RATE-OUTPUT variant.

The NMPC plans over the full rigid-body + motor PT1 dynamics, but only the
predicted body angular rates from the first prediction step are consumed by
the outer controller; they are passed as setpoints to an inner rate PID.
The direct control output u of this NMPC is therefore discarded by design
(see "Control Concept Note" in the system design document).

Model summary
-------------
    State x  in R^17:
        q         (4)  attitude quaternion, body -> navigation (Hamilton)
        omega^b   (3)  body angular rate vs. navigation, expressed in body frame
        p^n       (3)  position in NED navigation frame [m]
        v^n       (3)  velocity in NED navigation frame [m/s]
        omega_m   (4)  individual motor angular rates [rad/s]

    Control u in R^4:
        normalized motor commands in [0, 1]; mapped through the allocation
        matrix M to body torques and total thrust as in the original model.

The structure mirrors the `not gamma_available` branch of
copter_model_position.py but is split into its own file for clarity and
to keep the cost weights / state layout independent.

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat


def export_copterrates_ode_model(vehicle_config) -> AcadosModel:

    model_name = "copterrates_ode"

    # -----------------------------------------------------------------
    # Constants and configuration
    # -----------------------------------------------------------------
    gravity = vehicle_config.gravity_n[2]
    state_cfg = vehicle_config.state_cfg

    # Diagonal moment of inertia
    J_11 = vehicle_config.J[0, 0]
    J_22 = vehicle_config.J[1, 1]
    J_33 = vehicle_config.J[2, 2]

    # Sanity: the rate-output variant requires motor speeds in the state and
    # does NOT use the gamma (PT1 wrench) formulation.
    assert state_cfg["pos3d_available"], "Rate-NMPC requires pos3d in the state"
    assert state_cfg["vel3d_available"], "Rate-NMPC requires vel3d in the state"
    assert not state_cfg["gamma_available"], (
        "Rate-NMPC uses motor speeds in the state, not gamma. "
        "Set state_cfg['gamma_available'] = False for this model."
    )

    # -----------------------------------------------------------------
    # Symbolic state x and its derivative xdot
    # -----------------------------------------------------------------
    # Attitude quaternion (body -> navigation), Hamilton convention
    q_0 = SX.sym("q_0")  # qw
    q_1 = SX.sym("q_1")  # qx
    q_2 = SX.sym("q_2")  # qy
    q_3 = SX.sym("q_3")  # qz

    # Body angular rate (body vs. navigation, expressed in body frame)
    omega_x = SX.sym("omega_x")
    omega_y = SX.sym("omega_y")
    omega_z = SX.sym("omega_z")

    # Position and velocity in NED navigation frame
    pos = SX.sym("pos", 3, 1)
    vel = SX.sym("vel", 3, 1)

    # Individual motor angular rates [rad/s]
    cur_rotoromega = SX.sym("cur_rotoromega", vehicle_config.motorcount, 1)

    # Corresponding xdot symbols (acados implicit-form requirement)
    q0_dot = SX.sym("q0_dot")
    q1_dot = SX.sym("q1_dot")
    q2_dot = SX.sym("q2_dot")
    q3_dot = SX.sym("q3_dot")
    omega_x_dot = SX.sym("omega_x_dot")
    omega_y_dot = SX.sym("omega_y_dot")
    omega_z_dot = SX.sym("omega_z_dot")
    pos_dot = SX.sym("pos_dot", 3, 1)
    vel_dot = SX.sym("vel_dot", 3, 1)
    cur_rotoromega_dot = SX.sym("cur_rotoromega_dot", vehicle_config.motorcount, 1)

    x = vertcat(
        q_0, q_1, q_2, q_3,
        omega_x, omega_y, omega_z,
        pos,
        vel,
        cur_rotoromega,
    )
    xdot = vertcat(
        q0_dot, q1_dot, q2_dot, q3_dot,
        omega_x_dot, omega_y_dot, omega_z_dot,
        pos_dot,
        vel_dot,
        cur_rotoromega_dot,
    )

    nx = x.size()[0]
    assert nx == state_cfg["state_length"], (
        f"State size mismatch: model has {nx}, state_cfg expects "
        f"{state_cfg['state_length']}"
    )

    # -----------------------------------------------------------------
    # Control input u (normalized motor commands in [0, 1])
    # -----------------------------------------------------------------
    u = SX.sym("u", vehicle_config.motorcount, 1)

    # The control output of this NMPC is *not* squared — the outer
    # controller ignores u and uses the predicted body rates instead, but
    # this flag is kept for compatibility with the build_ocp / base solver
    # plumbing used by the position-output variant.
    ctrlout_u_is_squared = False

    # -----------------------------------------------------------------
    # Cost function weights (diagonal)
    # -----------------------------------------------------------------
    # Layout matches state_cfg indices. Motor speeds receive default zero
    # weight: the inner rate PID will track the planned omega^b, and biasing
    # the planner toward specific motor speeds tends to make the solver
    # fight the rate loop rather than help it.
    cost_u_weight = vehicle_config.cost_u_weight
    weight_diag = np.zeros((nx,))

    weight_diag[state_cfg["q_index"]:state_cfg["q_index_end"]] = (
        vehicle_config.weight_q * np.ones(4)
    )
    weight_diag[state_cfg["omega_roll_index"]] = vehicle_config.weight_omega_roll
    weight_diag[state_cfg["omega_pitch_index"]] = vehicle_config.weight_omega_pitch
    weight_diag[state_cfg["omega_yaw_index"]] = vehicle_config.weight_omega_yaw

    weight_diag[state_cfg["north_index"]] = vehicle_config.weight_north_east
    weight_diag[state_cfg["east_index"]] = vehicle_config.weight_north_east
    weight_diag[state_cfg["altitude_index"]] = vehicle_config.weight_altitude

    weight_diag[state_cfg["vel_north_index"]] = vehicle_config.weight_velocity_horizontal
    weight_diag[state_cfg["vel_east_index"]] = vehicle_config.weight_velocity_horizontal
    weight_diag[state_cfg["vel_down_index"]] = vehicle_config.weight_velocity_vertical

    # -----------------------------------------------------------------
    # System dynamics
    # -----------------------------------------------------------------
    # Allocation: M maps squared normalized motor commands to the body
    # wrench gamma = [tau_b; T]. With motor speeds in the state, the
    # "effective" normalized squared input is (omega_m / omega_max)^2.
    u_effective = cur_rotoromega / vehicle_config.motor_maxOmega_rad_per_sec
    gamma = vehicle_config.M @ (u_effective ** 2)
    # gamma[0:3] = body torques [N·m], gamma[3] = total thrust [N]

    # Motor PT1: omega_m_dot = (omega_max * u - omega_m) / T_motor
    rotoromega_dot_eq = (
        vehicle_config.motor_maxOmega_rad_per_sec * u - cur_rotoromega
    ) / vehicle_config.motor_T

    # Linear acceleration in NED. Thrust acts along -Z_body; gravity is +Z_NED.
    # The expressions below are the standard expansion of
    #     v_dot^n = (1/m) * R^n_b @ [0, 0, -T]^T + g * e_3
    # with the Hamilton-convention quaternion (q_0 = qw).
    accel = vertcat(
        -2.0 * (q_0 * q_2 + q_1 * q_3) * gamma[3] / vehicle_config.mass_kg,
        -2.0 * (q_2 * q_3 - q_0 * q_1) * gamma[3] / vehicle_config.mass_kg,
        -(1.0 - 2.0 * q_1 * q_1 - 2.0 * q_2 * q_2) * (gamma[3] / vehicle_config.mass_kg)
        + gravity,
    )

    # Optional drag model — left disabled until a calibrated Crazyflie
    # drag/rotor-drag model is available. Enabling this with a guessed
    # coefficient biases the planned omega^b and degrades inner-loop
    # tracking. See Faessler et al. 2018 for an anisotropic CF model.
    model_drag = False
    if model_drag:
        c_D = vehicle_config.windresistance / vehicle_config.mass_kg
        epsilon = 0.02  # avoids zero-velocity solver stalls
        vel_norm = (vel.T @ vel + epsilon) ** 0.5
        drag = -c_D * vel_norm * vel
        drag[2] = 0.0  # no vertical drag in this simple model
        accel = accel + drag

    # Body-rate dynamics: Euler's equation with diagonal inertia.
    omega_dot = vertcat(
        (gamma[0] + J_22 * omega_y * omega_z - J_33 * omega_y * omega_z) / J_11,
        (gamma[1] - J_11 * omega_x * omega_z + J_33 * omega_x * omega_z) / J_22,
        (gamma[2] + J_11 * omega_x * omega_y - J_22 * omega_x * omega_y) / J_33,
    )

    # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0; omega^b]
    q_dot = vertcat(
        -(omega_x * q_1) / 2.0 - (omega_y * q_2) / 2.0 - (omega_z * q_3) / 2.0,
         (omega_x * q_0) / 2.0 - (omega_y * q_3) / 2.0 + (omega_z * q_2) / 2.0,
         (omega_y * q_0) / 2.0 + (omega_x * q_3) / 2.0 - (omega_z * q_1) / 2.0,
         (omega_y * q_1) / 2.0 - (omega_x * q_2) / 2.0 + (omega_z * q_0) / 2.0,
    )

    f_expl = vertcat(
        q_dot,
        omega_dot,
        vel,
        accel,
        rotoromega_dot_eq,
    )

    f_impl = xdot - f_expl

    # -----------------------------------------------------------------
    # Pack as AcadosModel
    # -----------------------------------------------------------------
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.ctrlout_u_is_squared = ctrlout_u_is_squared
    model.weight_diag = weight_diag
    model.cost_u_weight = cost_u_weight
    model.state_cfg = state_cfg

    return model
