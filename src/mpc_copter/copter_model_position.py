"""
NMPC model definition for acados.

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, DM

# This file is the main entry point for the model definition with
# the help of casadi as symbolic expression.
#
# The main definitions are:
# - the state vector x
# - the control input vector u (u_c)
# - the dynamic model f_expl (dx/dt) = f(x, u)


def export_copterpos_ode_model(vehicle_config) -> AcadosModel:

    model_name = "copterpos_ode"

    # constants
    # ---------
    gravity = vehicle_config.gravity_n[2]
    state_cfg = vehicle_config.state_cfg

    # moment of inertia, assuming a diagonal inertia matrix
    J_11 = vehicle_config.J[0, 0]
    J_22 = vehicle_config.J[1, 1]
    J_33 = vehicle_config.J[2, 2]

    # set up state vector 'x'
    q_0 = SX.sym("q_0")  # qw (quaternion from body to navigation frame)
    q_1 = SX.sym("q_1")  # qx
    q_2 = SX.sym("q_2")  # qy
    q_3 = SX.sym("q_3")  # qz
    omega_x = SX.sym("omega_x")  # rotation rates body vs. navigation frame in body frame
    omega_y = SX.sym("omega_y")
    omega_z = SX.sym("omega_z")

    # Setup casadi variables for xdot
    q0_dot = SX.sym("q0_dot")
    q1_dot = SX.sym("q1_dot")
    q2_dot = SX.sym("q2_dot")
    q3_dot = SX.sym("q3_dot")
    omega_x_dot = SX.sym("omega_x_dot")
    omega_y_dot = SX.sym("omega_y_dot")
    omega_z_dot = SX.sym("omega_z_dot")

    if state_cfg["pos3d_available"]:
        pos = SX.sym("pos", 3, 1)  # position in north/east/down (NED) in meter
        pos_dot = SX.sym("pos_dot", 3, 1)

    if state_cfg["vel3d_available"]:
        vel = SX.sym("vel", 3, 1)  # velocity in NED (m/s)
        vel_dot = SX.sym("vel_dot", 3, 1)

    if state_cfg["gamma_available"]:
        gamma = SX.sym("gamma", 4, 1)  # torque and thrust (1D) as 4d vector
        gamma_dot = SX.sym("gamma_dot", 4, 1)

    if (
        state_cfg["pos3d_available"]
        and state_cfg["vel3d_available"]
        and state_cfg["gamma_available"]
    ):
        # This is the normal case: a 3D position and velocity is and gamma is
        # estimated in the state vector
        x = vertcat(q_0, q_1, q_2, q_3, omega_x, omega_y, omega_z, pos, vel, gamma)
        xdot = vertcat(
            q0_dot,
            q1_dot,
            q2_dot,
            q3_dot,
            omega_x_dot,
            omega_y_dot,
            omega_z_dot,
            pos_dot,
            vel_dot,
            gamma_dot,
        )
    elif not state_cfg["gamma_available"]:
        # this is the "old" approach: gamma is not in the state vector and
        # individual rotor speeds are in the state vector
        cur_rotoromega = SX.sym("cur_rotoromega", vehicle_config.motorcount, 1)
        cur_rotoromega_dot = SX.sym("cur_rotoromega_dot", vehicle_config.motorcount, 1)

        x = vertcat(q_0, q_1, q_2, q_3, omega_x, omega_y, omega_z, pos, vel, cur_rotoromega)
        xdot = vertcat(
            q0_dot,
            q1_dot,
            q2_dot,
            q3_dot,
            omega_x_dot,
            omega_y_dot,
            omega_z_dot,
            pos_dot,
            vel_dot,
            cur_rotoromega_dot,
        )
    else:
        assert False, "No controller available for this configuration"

    nx = x.size()[0]
    assert nx == state_cfg["state_length"], "State size mismatch"

    # Weights for the (N)MPC cost function
    weight_diag = np.ones((nx,)) * 1e-6  # default weight
    weight_diag[state_cfg["q_index"] : state_cfg["q_index_end"]] = (
        vehicle_config.weight_q
    )  # quaternion / attitude weight
    weight_diag[state_cfg["omega_roll_index"]] = (
        vehicle_config.weight_omega_roll
    )  # rotation rate weight body roll-axis
    weight_diag[state_cfg["omega_pitch_index"]] = (
        vehicle_config.weight_omega_pitch
    )  # rotation rate weight body pitch-axis
    weight_diag[state_cfg["omega_yaw_index"]] = (
        vehicle_config.weight_omega_yaw
    )  # rotation rate weight body yaw-axis

    if state_cfg["pos3d_available"]:
        weight_diag[state_cfg["north_index"]] = (
            vehicle_config.weight_north_east
        )  # position North (m/s)
        weight_diag[state_cfg["east_index"]] = vehicle_config.weight_north_east  # position East
        weight_diag[state_cfg["altitude_index"]] = vehicle_config.weight_altitude  # position Down

    if state_cfg["vel3d_available"]:
        weight_diag[state_cfg["vel_north_index"]] = (
            vehicle_config.weight_velocity_horizontal
        )  # velocity North
        weight_diag[state_cfg["vel_east_index"]] = (
            vehicle_config.weight_velocity_horizontal
        )  # velocity East
        weight_diag[state_cfg["vel_down_index"]] = (
            vehicle_config.weight_velocity_vertical
        )  # velocity Down
    # gamma is using the default weight above

    # Control input u (u_c)
    u = SX.sym("u", vehicle_config.motorcount, 1)

    # System Dynamics
    # ---------------

    # This is the normal case: a 3D position and velocity is available and
    # gamma is estimated in the state vector (this is the new approach from the paper)
    if (
        state_cfg["pos3d_available"]
        and state_cfg["vel3d_available"]
        and state_cfg["gamma_available"]
    ):

        ctrlout_u_is_squared = True
        gamma_req = vehicle_config.M @ u  # gamma_req is the torque and thrust "requested" from u
        # Note: u contains squared commands in the gamma_available version.
        # u = (omega_1^2, omega_2^2, ... omega_n^2) / vehicle_config.motor_maxOmega_rad_per_sec^2
        # gamma is the actual torque and thrust acting on the aircraft at the moment.
        # gamma[0:3] is the torque acting on the aircraft in the aircraft body coordinate system
        # gamma[3] is the thrust component in the body aircraft coordinate system
        gamma_dot_eq = (gamma_req - gamma) / vehicle_config.motor_T

        accel = vertcat(
            -2.0 * (q_0 * q_2 + q_1 * q_3) * gamma[3] / vehicle_config.mass_kg,
            -2.0 * (q_2 * q_3 - q_0 * q_1) * gamma[3] / vehicle_config.mass_kg,
            -(1.0 - 2.0 * q_1 * q_1 - 2.0 * q_2 * q_2) * (gamma[3] / vehicle_config.mass_kg)
            + gravity,
        )

        # drag model
        # ----------
        model_drag = False  # Set to False to disable drag model
        if model_drag:
            c_D = vehicle_config.windresistance / vehicle_config.mass_kg  # simple drag model
            epsilon = 0.02  # solver gets stuck otherwise at zero velocity
            vel_norm = (vel.T @ vel + epsilon) ** 0.5
            drag = -c_D * vel_norm * vel
            drag[2] = 0.0  # no drag in vertical direction
            accel += drag  # add drag to acceleration

        omega_dot = vertcat(
            (gamma[0] + J_22 * omega_y * omega_z - J_33 * omega_y * omega_z) / J_11,
            (gamma[1] - J_11 * omega_x * omega_z + J_33 * omega_x * omega_z) / J_22,
            (gamma[2] + J_11 * omega_x * omega_y - J_22 * omega_x * omega_y) / J_33,
        )

        model_rotor_inertia = False  # Set to True to include rotor inertia in the MPC
        if model_rotor_inertia:
            # approximate motor omega in rad/s with control input u
            epsilon = 0.02  # solver gets stuck otherwise at zero velocity
            sqr_u = (u + epsilon) ** 0.5
            motordirection = DM(
                vehicle_config.motordirection * vehicle_config.motor_maxOmega_rad_per_sec
            )
            total_spin = motordirection.T @ sqr_u  # total spin in rad/s
            hR_0 = 0.0
            hR_1 = 0.0
            hR_2 = -vehicle_config.Jrotor * total_spin

            omega0 = (
                (gamma[3] / (vehicle_config.motorcount * vehicle_config.CT)) + 10.0 * epsilon
            ) ** 0.5
            beta = vehicle_config.Jrotor / (2 * omega0 * vehicle_config.CQ)
            hR_dot_0 = 0
            hR_dot_1 = 0
            hR_dot_2 = -beta * gamma_dot_eq[2]
            # Include rotor inertia in omega dot equation:
            omega_dot = vertcat(
                (
                    gamma[0]
                    - hR_dot_0
                    - omega_y * (J_33 * omega_z + hR_2)
                    + omega_z * (J_22 * omega_y + hR_1)
                )
                / J_11,
                (
                    gamma[1]
                    - hR_dot_1
                    + omega_x * (J_33 * omega_z + hR_2)
                    - omega_z * (J_11 * omega_x + hR_0)
                )
                / J_22,
                (
                    gamma[2]
                    - hR_dot_2
                    - omega_x * (J_22 * omega_y + hR_1)
                    + omega_y * (J_11 * omega_x + hR_0)
                )
                / J_33,
            )

        f_expl = vertcat(
            -(omega_x * q_1) / 2.0 - (omega_y * q_2) / 2.0 - (omega_z * q_3) / 2.0,
            (omega_x * q_0) / 2.0 - (omega_y * q_3) / 2.0 + (omega_z * q_2) / 2.0,
            (omega_y * q_0) / 2.0 + (omega_x * q_3) / 2.0 - (omega_z * q_1) / 2.0,
            (omega_y * q_1) / 2.0 - (omega_x * q_2) / 2.0 + (omega_z * q_0) / 2.0,
            omega_dot,
            vel,
            accel,
            gamma_dot_eq,
        )
    elif not state_cfg["gamma_available"]:
        # this is the "old" approach: gamma is not in the state vector and individual
        # rotor speeds are in the state vector

        u_effective = cur_rotoromega / vehicle_config.motor_maxOmega_rad_per_sec
        gamma = vehicle_config.M @ u_effective**2
        rotoromega_dot_eq = (
            vehicle_config.motor_maxOmega_rad_per_sec * u - cur_rotoromega
        ) / vehicle_config.motor_T

        ctrlout_u_is_squared = False

        # Not included (TODO): drag model and rotor inertia

        f_expl = vertcat(
            -(omega_x * q_1) / 2.0 - (omega_y * q_2) / 2.0 - (omega_z * q_3) / 2.0,
            (omega_x * q_0) / 2.0 - (omega_y * q_3) / 2.0 + (omega_z * q_2) / 2.0,
            (omega_y * q_0) / 2.0 + (omega_x * q_3) / 2.0 - (omega_z * q_1) / 2.0,
            (omega_y * q_1) / 2.0 - (omega_x * q_2) / 2.0 + (omega_z * q_0) / 2.0,
            (gamma[0] + J_22 * omega_y * omega_z - J_33 * omega_y * omega_z) / J_11,
            (gamma[1] - J_11 * omega_x * omega_z + J_33 * omega_x * omega_z) / J_22,
            (gamma[2] + J_11 * omega_x * omega_y - J_22 * omega_x * omega_y) / J_33,
            vel,
            -2.0 * (q_0 * q_2 + q_1 * q_3) * gamma[3] / vehicle_config.mass_kg,
            -2.0 * (q_2 * q_3 - q_0 * q_1) * gamma[3] / vehicle_config.mass_kg,
            -(1.0 - 2.0 * q_1 * q_1 - 2.0 * q_2 * q_2) * (gamma[3] / vehicle_config.mass_kg)
            + gravity,  # gamma[3] is the thrust component
            rotoromega_dot_eq,
        )
    else:
        assert False, "No dynamic model available for this configuration"

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.ctrlout_u_is_squared = ctrlout_u_is_squared
    model.name = model_name
    model.weight_diag = weight_diag
    model.state_cfg = state_cfg

    return model
