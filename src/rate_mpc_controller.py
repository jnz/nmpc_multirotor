import numpy as np
from base_controller import BaseController
from geodetic_toolbox import quat_to_rpy

class PidObject:
    """
    PID
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

        delta = -(measured - self.prev_measured)
        deriv = delta / dt if dt > 0.0 else 0.0
        if not np.isfinite(deriv): deriv = 0.0
        self.out_d = self.kd * deriv

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

class RateMPCController(BaseController):
    """
    Hybrid-Controller:
    PID for inner rate loop
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

    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        cfg = vehicle_config
        def G(name): return _g(cfg, name, self._DEFAULTS[name])

        # --- 1. Rate PID Controller Init ---
        self.pid_roll_rate = PidObject(
            G("pid_roll_rate_kp"), G("pid_roll_rate_ki"), G("pid_roll_rate_kd"), G("pid_roll_rate_kff"),
            i_limit=G("pid_roll_rate_integration_limit")
        )
        self.pid_pitch_rate = PidObject(
            G("pid_pitch_rate_kp"), G("pid_pitch_rate_ki"), G("pid_pitch_rate_kd"), G("pid_pitch_rate_kff"),
            i_limit=G("pid_pitch_rate_integration_limit")
        )
        self.pid_yaw_rate = PidObject(
            G("pid_yaw_rate_kp"), G("pid_yaw_rate_ki"), G("pid_yaw_rate_kd"), G("pid_yaw_rate_kff"),
            i_limit=G("pid_yaw_rate_integration_limit")
        )
        self.vel_thrust_min = G("pid_vel_thrust_min")
        self.vel_thrust_base = G("pid_vel_thrust_base")

        # --- 2. MPC Init (STUB) ---
        # TODO: add here
        self.predictedX = None

    def compute_control(self, state, keymap, dt_sec):
        # State extract
        cfg = self.vehicle_config.state_cfg
        omega = np.asarray(state[cfg["omega_index"]:cfg["omega_index_end"]], dtype=float)

        # Angular velocity in deg/s
        p_dps = np.rad2deg(omega[0])
        q_dps = np.rad2deg(omega[1])
        r_dps = np.rad2deg(omega[2])

        # ===============================================================
        # STUB: MPC
        # ===============================================================
        # TODO:
        # replace by: u_mpc = self.solver.solve_for_x0(...)

        roll_rate_sp = 0.0   # deg/s
        pitch_rate_sp = 0.0  # deg/s
        yaw_rate_sp = 0.0    # deg/s
        thrust_pwm = self.vel_thrust_base # ~ Hover Thrust
        # ===============================================================

        thrust_pwm = float(np.clip(thrust_pwm, self.vel_thrust_min, self._THRUST_PWM_MAX))

        # ===============================================================
        # Rate Loop (PID) -> Raw Torque Commands
        # ===============================================================
        self.pid_roll_rate.set_desired(roll_rate_sp)
        self.pid_pitch_rate.set_desired(pitch_rate_sp)
        self.pid_yaw_rate.set_desired(yaw_rate_sp)

        roll_out  = self.pid_roll_rate.update(p_dps, dt_sec)
        pitch_out = self.pid_pitch_rate.update(q_dps, dt_sec)
        yaw_out   = self.pid_yaw_rate.update(r_dps, dt_sec)

        roll_out  = float(np.clip(roll_out,  -32768.0, 32767.0))
        pitch_out = float(np.clip(pitch_out, -32768.0, 32767.0))
        yaw_out   = float(np.clip(yaw_out,   -32768.0, 32767.0))

        # ===============================================================
        # Power Distribution
        # ===============================================================
        pitch_mix = pitch_out
        yaw_mix   = yaw_out

        r_half = roll_out * 0.5
        p_half = pitch_mix * 0.5
        m1 = thrust_pwm - r_half + p_half + yaw_mix
        m2 = thrust_pwm - r_half - p_half - yaw_mix
        m3 = thrust_pwm + r_half - p_half + yaw_mix
        m4 = thrust_pwm + r_half + p_half - yaw_mix

        motors_pwm = np.array([m1, m2, m3, m4], dtype=float)

        # Normalisieren auf [umin, umax] für die Simulation
        u = motors_pwm / self._THRUST_PWM_MAX
        u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        return u, self.predictedX
