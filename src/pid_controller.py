import numpy as np

from geodetic_toolbox import quat_to_rpy
from base_controller import BaseController, update_vehicle_control_state

class PidObject:
    """
    Mirror of the Crazyflie PidObject (see firmware: pid.c / pid.h).
    Implements derivative on measurement (no derivative kick) and a
    yaw-aware wrap correction for both error and delta when isYawAngle.
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
        self.i_limit = float(i_limit)         # 0 means no limit (CF convention)
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
    Crazyflie 2.1 Brushless cascaded PID controller, ported from firmware
    (controller_pid.c, attitude_pid_controller.c, position_controller_pid.c,
    pid.c, power_distribution_quadrotor.c).

    Cascade (outer -> inner):
        Position (XYZ)  -> velocity setpoint
        Velocity (XYZ)  -> roll/pitch attitude setpoint + thrust
        Attitude (RPY)  -> body-rate setpoint
        Body rate       -> raw torque commands (roll, pitch, yaw)
        Power mixer     -> 4 motor commands

    Frame: NED (x North, y East, z Down).
    """

    # ---- Crazyflie firmware defaults.
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
        # Thrust is on the firmware's PWM scale [0 .. UINT16_MAX = 65535].
        # Hover sits around ~37000 (~55% PWM).
        "pid_vel_thrust_base":   37000.0,
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

    # Crazyflie firmware PWM scale: motor commands live in [0 .. 65535]
    # (UINT16_MAX). Hover thrust on a stock Crazyflie 2.1 is ~37000.
    # The simulation expects normalized motor commands u in [umin, umax]
    # (the renderer prints u*99, so u ~ [0, 1]). We divide by 65535 to
    # convert.
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
        # Match CF's capAngle, which returns in (-180, 180]
        if a == -180.0:
            a = 180.0
        return a

    def _reset_all(self, roll_deg, pitch_deg, yaw_deg, pos_ned, vel_ned):
        for pid in (self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate):
            pid.reset(0.0)
        self.pid_roll.reset(roll_deg)
        self.pid_pitch.reset(pitch_deg)
        self.pid_yaw.reset(yaw_deg)
        self.pid_vel_x.reset(vel_ned[0])
        self.pid_vel_y.reset(vel_ned[1])
        self.pid_vel_z.reset(vel_ned[2])
        self.pid_pos_x.reset(pos_ned[0])
        self.pid_pos_y.reset(pos_ned[1])
        self.pid_pos_z.reset(pos_ned[2])
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
        # In NED yaw is positive about z-Down (= turn right seen from above),
        # which matches the user's intuition for yaw_cmd = +1 -> turn right.
        if not np.isclose(yaw_cmd, 0.0):
            yaw_rate_dps = yaw_cmd * np.rad2deg(self.vehicle_config.max_rotation_rate_rps)
            self._yaw_desired_deg = self._wrap_deg(
                self._yaw_desired_deg + yaw_rate_dps * dt_sec
            )

        # ===============================================================
        # Outer loop:  Position -> velocity setpoint  (NED)
        # ===============================================================
        pos_sp_ned = self.ctrl_state.desired_pos

        # Manual horizontal velocity command: when sticks are deflected
        # the firmware switches the corresponding axes to velocity-hold.
        # Z always uses position-hold because vertical_cmd just shifts
        # desired_pos[2] in the FSM.
        horizontal_cmd = not np.allclose(cmd_b[0:2], 0.0)

        if horizontal_cmd:
            # User-commanded velocity in body frame (NED body): x forward,
            # y right. lateral_cmd = +1 -> right -> +y_East in body.
            v_max = self.vehicle_config.max_horizontal_velocity_mps
            vx_sp_body = cmd_b[0] * v_max     # forward
            vy_sp_body = cmd_b[1] * v_max     # right
            # Rotate body -> world (NED) using yaw
            yaw_rad = np.deg2rad(yaw_deg)
            cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
            vx_sp_ned = cy * vx_sp_body - sy * vy_sp_body
            vy_sp_ned = sy * vx_sp_body + cy * vy_sp_body
            # Reset position integrators so we don't wind up while moving
            self.pid_pos_x.reset(pos_ned[0])
            self.pid_pos_y.reset(pos_ned[1])
        else:
            self.pid_pos_x.set_desired(pos_sp_ned[0])
            self.pid_pos_y.set_desired(pos_sp_ned[1])
            vx_sp_ned = self.pid_pos_x.update(pos_ned[0], dt_sec)
            vy_sp_ned = self.pid_pos_y.update(pos_ned[1], dt_sec)

        # Z position loop is always active (FSM handles vertical_cmd by
        # adjusting desired_pos[2]). vertical_cmd = +1 means "go up" in
        # the user's frame; the FSM stores desired_pos in NED, so up is
        # already mapped to a smaller (more negative) z.
        self.pid_pos_z.set_desired(pos_sp_ned[2])
        vz_sp_ned = self.pid_pos_z.update(pos_ned[2], dt_sec)

        # Clip the velocity setpoint to the configured limits
        vx_sp_ned = float(np.clip(vx_sp_ned, -self.pos_vel_x_max, self.pos_vel_x_max))
        vy_sp_ned = float(np.clip(vy_sp_ned, -self.pos_vel_y_max, self.pos_vel_y_max))
        vz_sp_ned = float(np.clip(vz_sp_ned, -self.pos_vel_z_max, self.pos_vel_z_max))

        # ===============================================================
        # Middle loop:  Velocity -> roll/pitch + thrust  (NED)
        # ===============================================================
        self.pid_vel_x.set_desired(vx_sp_ned)
        self.pid_vel_y.set_desired(vy_sp_ned)
        self.pid_vel_z.set_desired(vz_sp_ned)
        u_vx = self.pid_vel_x.update(vel_ned[0], dt_sec)
        u_vy = self.pid_vel_y.update(vel_ned[1], dt_sec)
        u_vz = self.pid_vel_z.update(vel_ned[2], dt_sec)

        # Rotate world-frame velocity-PID outputs into the body frame so
        # roll/pitch commands stay consistent with the heading.
        yaw_rad = np.deg2rad(yaw_deg)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        u_fwd   =  cy * u_vx + sy * u_vy   # forward demand (body x)
        u_right = -sy * u_vx + cy * u_vy   # right   demand (body y, NED)

        # In NED:
        #   +pitch = nose down -> backward acceleration
        #   to fly forward (+u_fwd) we need negative pitch
        #   +roll  = right wing down -> right acceleration
        #   to fly right (+u_right) we need positive roll
        pitch_sp_deg = -u_fwd
        roll_sp_deg  =  u_right

        roll_sp_deg  = float(np.clip(roll_sp_deg,  -self.vel_roll_max,  self.vel_roll_max))
        pitch_sp_deg = float(np.clip(pitch_sp_deg, -self.vel_pitch_max, self.vel_pitch_max))

        thrust_scale = 1000.0
        thrust_pwm = self.vel_thrust_base - u_vz*thrust_scale
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
        self.pid_roll_rate.set_desired(roll_rate_sp)
        self.pid_pitch_rate.set_desired(pitch_rate_sp)
        self.pid_yaw_rate.set_desired(yaw_rate_sp)

        roll_out  = self.pid_roll_rate.update(p_dps, dt_sec)
        pitch_out = self.pid_pitch_rate.update(q_dps, dt_sec)
        yaw_out   = self.pid_yaw_rate.update(r_dps, dt_sec)

        # int16 saturation, like saturateSignedInt16 in firmware
        roll_out  = float(np.clip(roll_out,  -32768.0, 32767.0))
        pitch_out = float(np.clip(pitch_out, -32768.0, 32767.0))
        yaw_out   = float(np.clip(yaw_out,   -32768.0, 32767.0))

        # ===============================================================
        # Power distribution (powerDistributionLegacy / X-config)
        # ===============================================================
        # The firmware mixer is written for FLU. Our control outputs are
        # in NED, where pitch and yaw have opposite sign vs. FLU. We
        # absorb both flips here so the rest of the cascade stays in NED.
        pitch_mix =  pitch_out
        yaw_mix   =  yaw_out

        r_half = roll_out * 0.5
        p_half = pitch_mix * 0.5
        m1 = thrust_pwm - r_half + p_half + yaw_mix
        m2 = thrust_pwm - r_half - p_half - yaw_mix
        m3 = thrust_pwm + r_half - p_half + yaw_mix
        m4 = thrust_pwm + r_half + p_half - yaw_mix

        motors_pwm = np.array([m1, m2, m3, m4], dtype=float)
        # Normalize from PWM counts to simulation u in [umin, umax]
        u = motors_pwm / self._THRUST_PWM_MAX
        u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        return u, None

