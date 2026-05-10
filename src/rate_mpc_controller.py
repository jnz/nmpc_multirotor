"""
Hybrid NMPC + inner-rate-PID controller.

The NMPC plans over the full rigid-body + motor PT1 dynamics. From the
solver's predicted trajectory we extract the planned body angular rates
omega^b at the *first* prediction step (k=1) and pass them as setpoints
to the inner rate PID. The direct control output u of the NMPC is
discarded by design (see "Control Concept Note" in the system design
document).

A planned thrust-equivalent is taken from the same prediction step and
mapped to a base PWM that is then mixed with the PID outputs in the
Crazyflie-style power distribution.

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np

from base_controller import BaseController, update_vehicle_control_state
from geodetic_toolbox import quat_to_rpy, quat_from_rpy, quat_to_matrix, angle_diff

from acados_template import AcadosOcpSolver
from mpc_copter.copter_model_rates import export_copterrates_ode_model
from mpc_copter.build_ocp import build_ocp

# =====================================================================
# Inner rate PID
# =====================================================================
class PidObject:
    """
    Standard PID with anti-windup, derivative on measurement, optional
    feed-forward and output clamping.
    """
    __slots__ = (
        "kp", "ki", "kd", "kff",
        "i_limit", "output_limit",
        "desired", "integ", "prev_measured",
        "out_p", "out_i", "out_d", "out_ff",
        "initialized",
    )

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, kff=0.0, i_limit=0.0, output_limit=0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.kff = float(kff)
        self.i_limit = float(i_limit)
        self.output_limit = float(output_limit)
        self.desired = 0.0
        self.integ = 0.0
        self.prev_measured = 0.0
        self.initialized = False

    def reset(self, measured=0.0):
        self.integ = 0.0
        self.prev_measured = measured
        self.initialized = True

    def set_desired(self, desired):
        self.desired = float(desired)

    def update(self, measured, dt, is_yaw_angle=False):
        if not self.initialized:
            self.prev_measured = measured
            self.initialized = True

        error = self.desired - measured
        self.out_p = self.kp * error

        # Derivative on measurement (avoids derivative kick on setpoint changes).
        delta = -(measured - self.prev_measured)
        deriv = delta / dt if dt > 0.0 else 0.0
        if not np.isfinite(deriv):
            deriv = 0.0
        self.out_d = self.kd * deriv

        # Integrator with optional clamp.
        self.integ += error * dt
        if self.i_limit != 0.0:
            self.integ = np.clip(self.integ, -self.i_limit, self.i_limit)
        self.out_i = self.ki * self.integ

        self.out_ff = self.kff * self.desired
        output = self.out_p + self.out_i + self.out_d + self.out_ff

        if self.output_limit != 0.0:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_measured = measured
        return output


def _g(cfg, name, default):
    return float(getattr(cfg, name, default))


# =====================================================================
# Hybrid NMPC + rate-PID controller
# =====================================================================
class RateMPCController(BaseController):
    """
    Hybrid controller:
      - Outer loop: NMPC over the full copter dynamics; emits planned
        body rates (and a planned thrust) at the next prediction step.
      - Inner loop: per-axis rate PID drives the measured body rate to
        the planned setpoint and produces torque-like commands that are
        mixed with a base thrust PWM in the Crazyflie power distribution.
    """

    _DEFAULTS = {
        "pid_roll_rate_kp":  200.0, "pid_roll_rate_ki":  400.0,
        "pid_roll_rate_kd":    2.5, "pid_roll_rate_kff":   0.0,
        "pid_roll_rate_integration_limit":  33.3,
        "pid_pitch_rate_kp": 200.0, "pid_pitch_rate_ki": 400.0,
        "pid_pitch_rate_kd":   2.5, "pid_pitch_rate_kff":  0.0,
        "pid_pitch_rate_integration_limit": 33.3,
        "pid_yaw_rate_kp":   120.0, "pid_yaw_rate_ki":    16.7,
        "pid_yaw_rate_kd":     0.0, "pid_yaw_rate_kff":    0.0,
        "pid_yaw_rate_integration_limit":  166.7,
        "pid_vel_thrust_base":   37000.0,
        "pid_vel_thrust_min":    20000.0,
    }
    _THRUST_PWM_MAX = 65535.0

    # Index in the planned wrench gamma where the total thrust [N] sits.
    # gamma = M @ u^2 = [tau_x, tau_y, tau_z, T_total]^T
    _GAMMA_THRUST_IDX = 3

    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        cfg = vehicle_config
        def G(name): return _g(cfg, name, self._DEFAULTS[name])

        # -----------------------------------------------------------------
        # 1. Inner rate-PID loop
        # -----------------------------------------------------------------
        self.pid_roll_rate = PidObject(
            G("pid_roll_rate_kp"), G("pid_roll_rate_ki"),
            G("pid_roll_rate_kd"), G("pid_roll_rate_kff"),
            i_limit=G("pid_roll_rate_integration_limit"),
        )
        self.pid_pitch_rate = PidObject(
            G("pid_pitch_rate_kp"), G("pid_pitch_rate_ki"),
            G("pid_pitch_rate_kd"), G("pid_pitch_rate_kff"),
            i_limit=G("pid_pitch_rate_integration_limit"),
        )
        self.pid_yaw_rate = PidObject(
            G("pid_yaw_rate_kp"), G("pid_yaw_rate_ki"),
            G("pid_yaw_rate_kd"), G("pid_yaw_rate_kff"),
            i_limit=G("pid_yaw_rate_integration_limit"),
        )
        self.vel_thrust_min = G("pid_vel_thrust_min")
        self.vel_thrust_base = G("pid_vel_thrust_base")

        # -----------------------------------------------------------------
        # 2. NMPC outer loop
        # -----------------------------------------------------------------
        # Use the dedicated rate-output OCP block of the vehicle config if
        # provided, otherwise fall back to the position-output one. The
        # underlying acados model is the rate-output variant either way.
        ocp_cfg = getattr(vehicle_config, "ocp_rate_sim",
                          getattr(vehicle_config, "ocp_sim"))

        # build_ocp is reused - it's model-agnostic as long as the model
        # exposes weight_diag, cost_u_weight, state_cfg, etc.
        self.ocp, self.model, self.nx, self.nu, self.ny, self.N_horizon, self.Tf = (
            build_ocp(vehicle_config, ocp_cfg,
                      model_factory=export_copterrates_ode_model)
        )

        solver_json = "acados_ocp_" + self.model.name + ".json"
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)
        self.predictedX = np.ndarray((self.N_horizon, self.nx))

        # Cache hover thrust [N] for thrust-to-PWM mapping (linear approximation
        # around hover). Computed from m * g on first compute_control call so
        # we don't rely on import order of vehicle_config.
        self._hover_thrust_N = abs(vehicle_config.mass_kg * vehicle_config.gravity_n[2])

        # Total max thrust [N] from the allocation matrix: row 3 sums the
        # thrust contribution of all motors at u_i^2 = 1.
        self._max_thrust_N = float(np.sum(np.asarray(vehicle_config.M)[self._GAMMA_THRUST_IDX, :]))

    # -----------------------------------------------------------------
    # Reference handling - same shape as the position-output controller
    # -----------------------------------------------------------------
    def _update_references(self, state, keymap, dt_sec):
        """
        Update the OCP yref/yref_e from the keymap-driven setpoint pipeline.

        Mirrors the position-output controller but does not synthesize a
        feed-forward attitude reference from a drag model — for the
        Crazyflie we don't have a calibrated drag model, and the NMPC will
        plan the necessary tilt itself given a velocity reference.
        """
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

        # Delegated to the existing helper used by the position-output
        # controller — keeps the desired_pos / yaw integration consistent.
        update_vehicle_control_state(self.ctrl_state, cmd_b, pos, vel, q, dt_sec)

        if not np.isclose(yaw_cmd, 0.0):
            self.ctrl_state.yaw_rate_rps = yaw_cmd * self.vehicle_config.max_rotation_rate_rps
            self.ctrl_state.yaw_ref += self.ctrl_state.yaw_rate_rps * dt_sec
            self.ctrl_state.yaw_ref = angle_diff(self.ctrl_state.yaw_ref, 0.0)
        else:
            if not np.isclose(self.ctrl_state.yaw_rate_rps, 0.0):
                self.ctrl_state.yaw_ref = yaw
                self.ctrl_state.yaw_rate_rps = 0.0

        # Velocity reference in NED, derived from body-frame stick command
        # rotated by the current yaw reference.
        if np.allclose(cmd_b, 0.0):
            vel_n_ref = np.zeros(3)
        else:
            v_b = np.array([
                cmd_b[0] * self.vehicle_config.max_horizontal_velocity_mps,
                cmd_b[1] * self.vehicle_config.max_horizontal_velocity_mps,
                0.0,
            ])
            R_yaw = quat_to_matrix(quat_from_rpy(0.0, 0.0, self.ctrl_state.yaw_ref))
            vel_n_ref = R_yaw @ v_b
            vel_n_ref[2] = cmd_b[2] * self.vehicle_config.max_vertical_velocity_mps

        # Attitude reference: only yaw is fixed; roll/pitch are left at
        # level (the NMPC plans the actual tilt). omega reference is zero
        # except for the yaw-rate command.
        q_ref = quat_from_rpy(0.0, 0.0, self.ctrl_state.yaw_ref)
        omega_ref = np.array([0.0, 0.0, self.ctrl_state.yaw_rate_rps])

        # Build the running yref. We propagate the position reference along
        # the horizon using the velocity reference so the planner doesn't
        # see a step at k=0 followed by a static target.
        yref = np.zeros(self.ny)
        yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = self.ctrl_state.desired_pos
        yref[cfg["vel3d_index"]:cfg["vel3d_index_end"]] = vel_n_ref
        yref[cfg["q_index"]:cfg["q_index_end"]] = q_ref
        yref[cfg["omega_index"]:cfg["omega_index_end"]] = omega_ref

        N = self.N_horizon
        pos_pred = np.zeros((N + 1, 3))
        pos_pred[0] = self.ctrl_state.desired_pos
        for j in range(1, N + 1):
            pos_pred[j] = pos_pred[j - 1] + vel_n_ref * dt_sec

        for j in range(N):
            yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[j]
            self.acados_ocp_solver.set(j, "yref", np.copy(yref))

        # Terminal cost has no input component.
        yref_e = np.copy(yref[: self.nx])
        yref_e[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[N]
        self.acados_ocp_solver.set(N, "yref", yref_e)

    # -----------------------------------------------------------------
    # Thrust [N] -> PWM helper
    # -----------------------------------------------------------------
    def _thrust_N_to_pwm(self, thrust_N):
        """
        Linear hover-anchored mapping from total thrust [N] to base PWM.

        At T = m*g we want the configured hover-base PWM (vel_thrust_base);
        scaling above/below hover is linear in T. This matches what the
        Crazyflie firmware does in the velocity-controller branch.
        """
        if self._hover_thrust_N <= 0.0:
            return self.vel_thrust_base
        ratio = thrust_N / self._hover_thrust_N
        return self.vel_thrust_base * ratio

    # -----------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------
    def compute_control(self, state, keymap, dt_sec):
        cfg = self.vehicle_config.state_cfg

        # Update yref / yref_e along the horizon from the user setpoints.
        self._update_references(state, keymap, dt_sec)

        # ---- NMPC solve ----------------------------------------------
        # solve_for_x0 internally sets x0 as initial-state constraint and
        # returns the optimal u_0; we discard u_0 by design and read the
        # predicted state trajectory instead.
        _ = self.acados_ocp_solver.solve_for_x0(x0_bar=state)

        for i in range(self.N_horizon):
            self.predictedX[i, :] = self.acados_ocp_solver.get(i, "x")

        # ---- Extract setpoints from the planned trajectory ----------
        # Use prediction step k=1: predictedX[0] equals the current state
        # (initial condition) and is useless as a setpoint, while step 1
        # is the first dynamically-planned state and gives the rate PID
        # the next step's target.
        x_next = self.predictedX[1] if self.N_horizon > 1 else self.predictedX[0]

        omega_b_sp = x_next[cfg["omega_index"]:cfg["omega_index_end"]]  # [rad/s]
        roll_rate_sp  = float(np.rad2deg(omega_b_sp[0]))
        pitch_rate_sp = float(np.rad2deg(omega_b_sp[1]))
        yaw_rate_sp   = float(np.rad2deg(omega_b_sp[2]))

        # Planned thrust [N] from motor speeds at step 1, via the same
        # allocation row used by the model.
        rotoromega_next = x_next[cfg["rotoromega_index"]:cfg["rotoromega_index_end"]] \
            if "rotoromega_index" in cfg else None
        if rotoromega_next is not None:
            u_eff = rotoromega_next / self.vehicle_config.motor_maxOmega_rad_per_sec
            thrust_N = float(np.asarray(self.vehicle_config.M)[self._GAMMA_THRUST_IDX, :]
                             @ (u_eff ** 2))
        else:
            # Fallback: assume hover thrust if motor speeds aren't indexed.
            thrust_N = self._hover_thrust_N

        thrust_pwm = self._thrust_N_to_pwm(thrust_N)
        thrust_pwm = float(np.clip(thrust_pwm, self.vel_thrust_min, self._THRUST_PWM_MAX))

        # ---- Inner rate-PID loop -------------------------------------
        # Measured body rate (from the current state, not the prediction).
        omega_meas = np.asarray(state[cfg["omega_index"]:cfg["omega_index_end"]],
                                dtype=float)
        p_dps = np.rad2deg(omega_meas[0])
        q_dps = np.rad2deg(omega_meas[1])
        r_dps = np.rad2deg(omega_meas[2])

        self.pid_roll_rate.set_desired(roll_rate_sp)
        self.pid_pitch_rate.set_desired(pitch_rate_sp)
        self.pid_yaw_rate.set_desired(yaw_rate_sp)

        roll_out  = self.pid_roll_rate.update(p_dps, dt_sec)
        pitch_out = self.pid_pitch_rate.update(q_dps, dt_sec)
        yaw_out   = self.pid_yaw_rate.update(r_dps, dt_sec)

        # Match the firmware's int16 saturation on torque commands.
        roll_out  = float(np.clip(roll_out,  -32768.0, 32767.0))
        pitch_out = float(np.clip(pitch_out, -32768.0, 32767.0))
        yaw_out   = float(np.clip(yaw_out,   -32768.0, 32767.0))

        # ---- Crazyflie-style power distribution ----------------------
        r_half = roll_out * 0.5
        p_half = pitch_out * 0.5
        m1 = thrust_pwm - r_half + p_half + yaw_out
        m2 = thrust_pwm - r_half - p_half - yaw_out
        m3 = thrust_pwm + r_half - p_half + yaw_out
        m4 = thrust_pwm + r_half + p_half - yaw_out

        motors_pwm = np.array([m1, m2, m3, m4], dtype=float)

        # Normalize to [umin, umax] for the simulation interface.
        u = motors_pwm / self._THRUST_PWM_MAX
        u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        # ---- Stats ---------------------------------------------------
        self.stats["time_tot"] = self.acados_ocp_solver.get_stats("time_tot")
        self.stats["sqp_iter"] = self.acados_ocp_solver.get_stats("sqp_iter")
        self.stats["qp_iter"]  = self.acados_ocp_solver.get_stats("qp_iter")

        return u, self.predictedX
