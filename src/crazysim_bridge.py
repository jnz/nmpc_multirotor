#!/usr/bin/env python3
"""
CrazySim NMPC bridge.

Connects the RateMPCController outer NMPC loop to a CrazySim MuJoCo
simulation instance via cflib over UDP (simulated CrazyRadio link).

Usage:
    python crazysim_bridge.py [--uri udp://127.0.0.1:19850]

Architecture:
    CrazySim -> cflib log -> assemble NED state -> NMPC outer loop
    planned angular-rate setpoints -> send_setpoint -> CF firmware rate PID
    motors -> CrazySim

    The inner rate PID runs inside the CrazySim firmware.  The outer NMPC
    runs here and sends rate setpoints (roll/pitch/yaw in deg/s) plus a
    thrust (uint16) to the firmware's stabiliser.

Frame conventions
-----------------
CrazySim / CF firmware world frame:  ENU  — x = East,    y = North,  z = Up
CrazySim / CF firmware body frame:   FLU  — x = forward, y = left,   z = up
NMPC / internal simulation frame:    NED  — x = North,   y = East,   z = Down
NMPC body frame:                     FRD  — x = forward, y = right,  z = down

Position / velocity:
    (N, E, D)_ned = (y_enu, x_enu, -z_enu)

Body angular rates (gyro, body-frame only — world frame irrelevant):
    (p, q, r)_ned = (p_flu,  -q_flu,  -r_flu)

Quaternion (body->world):
    q_ned = q_ENU2NED ⊗ q_cf ⊗ q_FRD2FLU
    where q_ENU2NED = [0, 1/√2, 1/√2, 0]  (180° around axis [1,1,0]/√2)
          q_FRD2FLU = [0, 1,    0,    0]   (180° around body x)

(c) Jan Zwiener (jan@zwiener.org)
"""

import sys
import os
import tty
import termios
import select
import time
import logging
import argparse
import threading
from pathlib import Path

CFLIB_SIM_PATH = os.path.expanduser("~/developer/CrazySim/crazyflie-lib-python")
CFLIB_REAL_PATH = os.path.expanduser("~/developer/cflib_real")
cf_mode = os.getenv("CF_MODE", "sim").lower()
if cf_mode == "real":
    print("Initializing real Crazyflie interface...")
    sys.path.insert(0, CFLIB_REAL_PATH)
else:
    print("Initializing CrazySim interface...")
    sys.path.insert(0, CFLIB_SIM_PATH)

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Locate the src directory so imports work when called from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geodetic_toolbox import quat_multiply

from vehicleconfig import CopterConfig
from rate_mpc_controller import RateMPCController

# Suppress cflib INFO noise; set to logging.DEBUG for detailed CRTP traces.
logging.basicConfig(level=logging.WARNING)

DEFAULT_URI = "udp://127.0.0.1:19850"
CTRL_HZ = 100.0
CTRL_DT = 1.0 / CTRL_HZ

# -----------------------------------------------------------------------
# Frame-conversion helpers
# -----------------------------------------------------------------------

# Constant quaternions used for the ENU↔NED frame conversion.
# q_ENU2NED: 180° rotation around the world axis [1,1,0]/√2  →  [0, 1/√2, 1/√2, 0]
# q_FRD2FLU: 180° rotation around body x                     →  [0, 1,    0,    0  ]
_S = 1.0 / np.sqrt(2.0)
_Q_ENU2NED = np.array([0.0, _S, _S, 0.0])   # [qw, qx, qy, qz]
_Q_FRD2FLU = np.array([0.0, 1.0, 0.0, 0.0])


def cf_to_ned_pos(x: float, y: float, z: float) -> np.ndarray:
    """CF world position ENU (x=East, y=North, z=Up) → NED (x=North, y=East, z=Down)."""
    return np.array([y, x, -z])


def cf_to_ned_vel(vx: float, vy: float, vz: float) -> np.ndarray:
    """CF world velocity ENU to NED."""
    return np.array([vy, vx, -vz])


def cf_to_ned_omega(gx_dps: float, gy_dps: float, gz_dps: float) -> np.ndarray:
    """CF body angular rate FLU (deg/s) -> NED body FRD (rad/s).
    Body-frame only; world frame does not affect this conversion.
    FLU→FRD: x unchanged, y and z negated."""
    return np.deg2rad(np.array([gx_dps, -gy_dps, -gz_dps]))


def cf_to_ned_quat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    CF quaternion (FLU body to ENU world) -> NED quaternion (FRD body -> NED world).

    q_ned = q_ENU2NED ⊗ q_cf ⊗ q_FRD2FLU
    """
    q_cf = np.array([qw, qx, qy, qz])
    q = quat_multiply(quat_multiply(_Q_ENU2NED, q_cf), _Q_FRD2FLU)
    return q / np.linalg.norm(q)


def ned_omega_to_cf_dps(omega_ned: np.ndarray) -> np.ndarray:
    """NED body FRD angular rate (rad/s) -> CF body FLU (deg/s)."""
    return np.rad2deg(np.array([omega_ned[0], -omega_ned[1], -omega_ned[2]]))


# -----------------------------------------------------------------------
# Bridge class
# -----------------------------------------------------------------------

class CrazySimBridge:
    """
    Runs the NMPC outer loop and exchanges state/setpoints with CrazySim.
    """

    def __init__(self, uri: str, vehicle_config: CopterConfig):
        self.uri = uri
        self.vehicle_config = vehicle_config

        # Latest log data received from cflib callbacks.
        self._log_data: dict = {}
        self._log_lock = threading.Lock()
        self._log_ready = threading.Event()

        # Estimated motor angular rates maintained via software PT1 model.
        # Initialized to 70 % of max omega (rough hover approximation).
        hover_omega = vehicle_config.motor_maxOmega_rad_per_sec * 0.7
        self._motor_omega = np.ones(vehicle_config.motorcount) * hover_omega

        # Outer NMPC controller (inner rate PID runs inside CF firmware).
        self.controller = RateMPCController(vehicle_config)

        # Stick/velocity commands forwarded to the NMPC reference generator.
        # All-zero -> hold current position.
        self._keymap = {
            "longitudinal_cmd": 0.0,
            "lateral_cmd":      0.0,
            "yaw_cmd":          0.0,
            "vertical_cmd":     0.0,
        }

        self._running = False
        self._keymap_lock = threading.Lock()

    # -------------------------------------------------------------------
    # Keyboard input
    # -------------------------------------------------------------------

    _KEY_BINDINGS = {
        'w': ('longitudinal_cmd', +1.0),
        's': ('longitudinal_cmd', -1.0),
        'd': ('lateral_cmd',      +1.0),
        'a': ('lateral_cmd',      -1.0),
        'e': ('yaw_cmd',          +1.0),
        'q': ('yaw_cmd',          -1.0),
        'r': ('vertical_cmd',     +1.0),
        'f': ('vertical_cmd',     -1.0),
    }

    def _keyboard_thread(self):
        """
        Reads single keystrokes in raw terminal mode.

        Hold a key → axis stays at ±1.  Release (no input for 80 ms) → axis
        resets to 0.  Ctrl-C / ESC stop the bridge.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.08)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ('\x03', '\x1b'):   # Ctrl-C or ESC
                        self._running = False
                        break
                    if ch in self._KEY_BINDINGS:
                        axis, value = self._KEY_BINDINGS[ch]
                        with self._keymap_lock:
                            self._keymap[axis] = value
                else:
                    # No key held — zero all axes.
                    with self._keymap_lock:
                        for k in self._keymap:
                            self._keymap[k] = 0.0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # -------------------------------------------------------------------
    # cflib logging
    # -------------------------------------------------------------------

    def _log_cb(self, timestamp, data, logconf):
        with self._log_lock:
            self._log_data.update(data)
            self._log_ready.set()

    def _start_logging(self, cf):
        """Register log configurations and start streaming."""
        # FP16 = 2 bytes each; 26-byte packet limit leaves 21 bytes for data
        # after 5 bytes of block-ID + timestamp overhead -> max 10 FP16 per block.
        configs = [
            ("PosVel", 10, [
                ("stateEstimate.x",  "FP16"),
                ("stateEstimate.y",  "FP16"),
                ("stateEstimate.z",  "FP16"),
                ("stateEstimate.vx", "FP16"),
                ("stateEstimate.vy", "FP16"),
                ("stateEstimate.vz", "FP16"),
            ]),
            ("Quat", 10, [
                ("stateEstimate.qx", "FP16"),
                ("stateEstimate.qy", "FP16"),
                ("stateEstimate.qz", "FP16"),
                ("stateEstimate.qw", "FP16"),
            ]),
            ("Gyro", 10, [
                ("gyro.x", "FP16"),
                ("gyro.y", "FP16"),
                ("gyro.z", "FP16"),
            ]),
        ]
        for name, period_ms, variables in configs:
            lc = LogConfig(name=name, period_in_ms=period_ms)
            for var, typ in variables:
                lc.add_variable(var, typ)
            cf.log.add_config(lc)
            lc.data_received_cb.add_callback(self._log_cb)
            lc.start()

    # -------------------------------------------------------------------
    # State assembly
    # -------------------------------------------------------------------

    def _assemble_state(self) -> np.ndarray:
        """
        Build the NMPC state vector from the latest cflib log snapshot.

        The vector layout matches CopterConfig(vehicle=21, state_layout=
        "rotorspeed_in_state"):
            [0:4]   quaternion        [qw, qx, qy, qz]  (body → NED)
            [4:7]   angular rates     [p, q, r]          rad/s, NED body
            [7:10]  position          [N, E, D]          m, NED
            [10:13] velocity          [vN, vE, vD]       m/s, NED
            [13:17] motor omega       [ω1..ω4]           rad/s
        """
        with self._log_lock:
            d = dict(self._log_data)

        cfg = self.vehicle_config.state_cfg
        state = np.zeros(cfg["state_length"])

        # --- Attitude (quaternion) ---
        qw = d.get("stateEstimate.qw", 1.0)
        qx = d.get("stateEstimate.qx", 0.0)
        qy = d.get("stateEstimate.qy", 0.0)
        qz = d.get("stateEstimate.qz", 0.0)
        state[cfg["q_index"]:cfg["q_index_end"]] = cf_to_ned_quat(qw, qx, qy, qz)

        # --- Angular rates (gyro) ---
        state[cfg["omega_index"]:cfg["omega_index_end"]] = cf_to_ned_omega(
            d.get("gyro.x", 0.0),
            d.get("gyro.y", 0.0),
            d.get("gyro.z", 0.0),
        )

        # --- Position ---
        # Default z = -0.3 (30 cm above ground) so the NMPC sees a hovering
        # initial condition before the first real measurement arrives.
        state[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = cf_to_ned_pos(
            d.get("stateEstimate.x",  0.0),
            d.get("stateEstimate.y",  0.0),
            d.get("stateEstimate.z",  0.3),
        )

        # --- Velocity ---
        state[cfg["vel3d_index"]:cfg["vel3d_index_end"]] = cf_to_ned_vel(
            d.get("stateEstimate.vx", 0.0),
            d.get("stateEstimate.vy", 0.0),
            d.get("stateEstimate.vz", 0.0),
        )

        # --- Motor angular rates (tracked via software PT1, see _track_motors) ---
        if cfg["rotoromega_available"]:
            state[cfg["rotoromega_index"]:cfg["rotoromega_index_end"]] = self._motor_omega

        return state

    def _track_motors(self, u_cmd: np.ndarray, dt_sec: float):
        """
        Update the internal motor-speed estimate using the PT1 model.

        u_cmd: normalised motor commands [0..1] as returned by
               RateMPCController (Crazyflie power-distribution output).
        """
        omega_target = self.vehicle_config.motor_maxOmega_rad_per_sec * np.clip(
            u_cmd, self.vehicle_config.umin, self.vehicle_config.umax
        )
        self._motor_omega += (
            (omega_target - self._motor_omega) / self.vehicle_config.motor_T * dt_sec
        )
        np.clip(
            self._motor_omega,
            0.0, self.vehicle_config.motor_maxOmega_rad_per_sec,
            out=self._motor_omega,
        )

    # -------------------------------------------------------------------
    # Thrust extraction helper
    # -------------------------------------------------------------------

    def _planned_thrust_uint16(self, x_next: np.ndarray) -> int:
        """Convert planned motor speeds at prediction step k=1 to CF uint16 thrust."""
        cfg = self.vehicle_config.state_cfg
        if cfg.get("rotoromega_available"):
            omega1 = x_next[cfg["rotoromega_index"]:cfg["rotoromega_index_end"]]
            u_eff = omega1 / self.vehicle_config.motor_maxOmega_rad_per_sec
            thrust_N = float(np.asarray(self.vehicle_config.M)[3, :] @ (u_eff ** 2))
        else:
            thrust_N = self.controller._hover_thrust_N

        thrust_pwm = self.controller._thrust_N_to_pwm(thrust_N)
        return int(np.clip(thrust_pwm,
                           self.controller.vel_thrust_min,
                           self.controller._THRUST_PWM_MAX))

    # -------------------------------------------------------------------
    # Main control loop
    # -------------------------------------------------------------------

    def run(self, cf):
        """
        Configure the CF firmware and run the 100 Hz NMPC control loop.
        Blocks until KeyboardInterrupt or an unrecoverable error.
        """
        print("Configuring CF firmware parameters...")

        # Use EKF state estimator (needed for position + velocity).
        cf.param.set_value("stabilizer.estimator", "2")
        time.sleep(0.3)

        # Switch roll and pitch to rate mode so send_setpoint interprets
        # the first two arguments as deg/s rather than degrees.
        # Yaw is always rate in the RPYT commander.
        # NOTE: parameter names depend on the CF firmware version.
        #       "flightmode.stabModeRoll" = 0 (ANGLE) / 1 (RATE)
        try:
            cf.param.set_value("flightmode.stabModeRoll",  "1")
            cf.param.set_value("flightmode.stabModePitch", "1")
        except Exception as exc:
            print(f"Warning: could not set rate mode parameters: {exc}")
            print("Continuing – verify manually that the firmware is in rate mode.")

        time.sleep(0.3)

        print("Starting log streams...")
        self._start_logging(cf)

        print("Waiting for first state measurement...")
        if not self._log_ready.wait(timeout=5.0):
            print("ERROR: No log data within 5 s. Check URI and that CrazySim is running.",
                  file=sys.stderr)
            return

        # Send a zero setpoint once to unlock the CF commander watchdog.
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

        print(
            "\nControls (hold key):\n"
            "  W / S  — forward / backward\n"
            "  A / D  — left / right\n"
            "  Q / E  — yaw left / right\n"
            "  R / F  — up / down\n"
            "  ESC / Ctrl-C — stop\n"
        )
        print(f"NMPC control loop active at {CTRL_HZ:.0f} Hz.\n")

        self._running = True
        kb_thread = threading.Thread(target=self._keyboard_thread, daemon=True)
        kb_thread.start()

        next_t = time.time()
        iter_count = 0

        try:
            while self._running:
                # Rate-limit to CTRL_HZ.
                now = time.time()
                wait = next_t - now
                if wait > 0:
                    time.sleep(wait)
                next_t += CTRL_DT

                # ------ Assemble NMPC state vector ------
                state = self._assemble_state()

                with self._keymap_lock:
                    keymap = dict(self._keymap)

                # ------ NMPC outer-loop solve ------
                u, predictedX = self.controller.compute_control(
                    state, keymap, CTRL_DT
                )

                # Update motor-speed estimate (used on the next iteration).
                self._track_motors(u, CTRL_DT)

                # ------ Extract planned rates at step k=1 ------
                cfg = self.vehicle_config.state_cfg
                x1 = (predictedX[1]
                      if self.controller.N_horizon > 1
                      else predictedX[0])

                omega_sp_ned = x1[cfg["omega_index"]:cfg["omega_index_end"]]

                # Convert NED rate setpoints to CF body frame [deg/s].
                omega_cf = ned_omega_to_cf_dps(omega_sp_ned)
                roll_rate  = float(np.clip(omega_cf[0], -720.0, 720.0))
                pitch_rate = float(np.clip(omega_cf[1], -720.0, 720.0))
                yaw_rate   = float(np.clip(omega_cf[2], -720.0, 720.0))

                thrust_uint16 = self._planned_thrust_uint16(x1)

                # ------ Send to CrazySim ------
                cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust_uint16)

                # ------ Console status (1 Hz) ------
                iter_count += 1
                if iter_count % int(CTRL_HZ) == 0:
                    stats = self.controller.get_stats()
                    t_ms = float(np.asarray(stats.get("time_tot", [0.0])).flat[0]) * 1e3
                    pos = state[cfg["pos3d_index"]:cfg["pos3d_index_end"]]
                    print(
                        f"NMPC {t_ms:.1f} ms | "
                        f"pos NED ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m | "
                        f"rates CF: r={roll_rate:+6.1f} p={pitch_rate:+6.1f} y={yaw_rate:+6.1f} °/s | "
                        f"T={thrust_uint16:5d}"
                    )

        except KeyboardInterrupt:
            print("\nInterrupted - sending stop command.")
        finally:
            cf.commander.send_setpoint(0, 0, 0, 0)
            self._running = False


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Connect RateMPCController to CrazySim via cflib UDP"
    )
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help=f"CrazySim CRTP URI  (default: {DEFAULT_URI})",
    )
    args = parser.parse_args()

    cflib.crtp.init_drivers()

    vehicle_config = CopterConfig(vehicle=21, state_layout="rotorspeed_in_state")

    bridge = CrazySimBridge(args.uri, vehicle_config)

    print(f"Connecting to {args.uri} ...")
    cache_dir = str(Path(__file__).resolve().parent.parent / "cache")
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        bridge.run(scf.cf)


if __name__ == "__main__":
    main()
