"""
Multirotor Simulation Environment Class
(Physic Simulation)

(c) Jan Zwiener (jan@zwiener.org)
"""

# import gym                # uncomment this line if you want to use the OpenAI Gym interface
# from gym import spaces    # uncomment this line if you want to use the OpenAI Gym interface
import numpy as np
import random

# Get the vehicle config
from vehicleconfig import get_vehicle_config
from visualization3d import RenderStlPygame
from geodetic_toolbox import *

# This is a simple multirotor simulation environment with an OpenAI Gym
# interface.


# <Gym Interface>
# class MultirotorSimEnv(gym.Env):
# </Gym Interface>
class MultirotorSimEnv:
    def __init__(self, vehicle=0):
        # vehicle config describes the vehicle specific properties:
        self.vehicle_config = get_vehicle_config(vehicle)

        self.reset_count = 0  # keep track of calls to reset() function
        self.time_sec = 0.0  # keep track of simulation time
        self.dt_sec = 1.0 / 240.0  # update rate of the simulation

        # initialize state of the vehicle
        # <state>
        state = self.reset()  # reset will add and reset additional basic state variables
        # </state>

        # uncomment these lines if you want to use the OpenAI Gym interface:
        # <Gym Interface>
        # Action space is set to actuator umin/umax limits
        # self.action_space = spaces.Box(low=self.vehicle_config.umin, high=self.vehicle_config.umax, shape=(self.vehicle_config.motorcount,))
        # obs_hi = np.ones(state.shape[0]) * 1000.0
        # self.observation_space = spaces.Box(low=-obs_hi, high=obs_hi, dtype=np.float32)
        # </Gym Interface>

        # by default don't open a window until render() is being called
        # If render() is being called, use RenderStlPygame from render_stl.py
        # to visualize
        self.user_wants_to_render = False

    def reset(self):
        """
        Gym interface. Reset the simulation.
        :return state (state vector)
        """

        print("Resetting simulation...")

        # <state>
        self.pos_n = np.array([0.0, 0.0, -0.7])
        self.vel_n = np.zeros(3)
        self.wind_n = np.array([0.0, 0.0, 0.0])  # wind in NED frame

        # Simulate the loss of motors:
        self.list_of_inactive_motors = (
            []
        )  # add motors to this list to simulate a loss. For random 3 motors for example: random.sample(list(range(0, self.vehicle_config.motorcount)), 3)

        # Maintain the attitude as quaternion and Euler angles. The source of truth is
        # the quaternion (self.q) and roll_deg, pitch_deg and yaw_deg will be updated
        # based on the quaternion. But here for initialization the Euler angles are
        # used to initialize the orientation (Euler angles are a bit more readable)
        self.roll_deg = 0.0  # Random initialization with e.g. np.random.uniform(-10, 10)
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        # Careful: this quaternion is in the order: qw, qx,qy,qz (qw is the real part)
        self.q = quat_from_rpy(
            np.deg2rad(self.roll_deg), np.deg2rad(self.pitch_deg), np.deg2rad(self.yaw_deg)
        )

        roll_rate_rps = 0.0  # np.deg2rad(np.random.uniform(-5, 5))
        pitch_rate_rps = 0.0  # np.deg2rad(np.random.uniform(-5, 5))
        yaw_rate_rps = 0.0
        self.omega = np.array(
            [roll_rate_rps, pitch_rate_rps, yaw_rate_rps]
        )  # vehicle rotation rates
        # motor/rotor rotation rates, start at 70% of max. rotation rates (70% is a very rough approximation for a typical hover setpoint)
        self.motoromega_rad_per_sec = (
            np.ones(self.vehicle_config.motorcount)
            * self.vehicle_config.motor_maxOmega_rad_per_sec
            * 0.7
        )
        self.motoromega_dot_rad_per_sec2 = (
            self.motoromega_rad_per_sec * 0.0
        )  # initial acceleration of the motors
        self.motor_power_mech_kw = np.zeros(
            self.vehicle_config.motorcount
        )  # mechanical power demand per motor in kW
        self.motor_thrust_N = np.zeros(self.vehicle_config.motorcount)  # thrust per motor in Newton
        self.motor_torque_Nm = np.zeros(
            self.vehicle_config.motorcount
        )  # torque per motor in Nm (without motor direction!)
        self.gamma = np.zeros(
            (4,)
        )  # will be updated by step() to reflect the status of motoromega_rad_per_sec
        self.energy_burned_kwh = 0.0  # total amount of energy used by motors
        # </state>
        self.update_state()  # create/update state vector

        # <simulation>
        self.time_sec = 0.0
        self.reset_count += 1
        # </simulation>

        return self.state

    def update_state(self):
        """
        Internal helper function to update self.state vector based on attributes such as self.q, self.pos, etc.
        """
        euler = quat_to_rpy(self.q)
        self.roll_deg = np.rad2deg(euler[0])
        self.pitch_deg = np.rad2deg(euler[1])
        self.yaw_deg = np.rad2deg(euler[2])

        # Produce state vector:
        state_cfg = self.vehicle_config.state_cfg
        self.state = np.zeros((state_cfg["state_length"],))
        self.state[state_cfg["q_index"] : state_cfg["q_index_end"]] = self.q
        self.state[state_cfg["omega_index"] : state_cfg["omega_index_end"]] = self.omega

        if state_cfg["pos3d_available"]:
            self.state[state_cfg["pos3d_index"] : state_cfg["pos3d_index_end"]] = self.pos_n
        elif state_cfg["altitude_available"]:
            self.state[state_cfg["altitude_index"]] = self.pos_n[2]

        if state_cfg["vel3d_available"]:
            self.state[state_cfg["vel3d_index"] : state_cfg["vel3d_index_end"]] = self.vel_n
        else:
            self.state[state_cfg["vel_down_index"]] = self.vel_n[2]

        if state_cfg["gamma_available"]:
            self.state[state_cfg["gamma_index"] : state_cfg["gamma_index_end"]] = self.gamma

        if state_cfg["rotoromega_available"]:
            self.state[state_cfg["rotoromega_index"] : state_cfg["rotoromega_index_end"]] = (
                self.motoromega_rad_per_sec
            )

    def step(self, action):
        """
        Gym interface step function to simulate the system.
        :param action Control input to the simulation, i.e. motor/rotor
                      setpoints between 0 and 1
                      (actually umin and umax to be precise)
        :return state (state vector), reward (score), done (simulation done?)
        """

        # <MOTOR SIMULATION>
        # action is the u vector with actuator setpoints between 0 and 1 for each motor/rotor
        action = np.clip(action, self.vehicle_config.umin, self.vehicle_config.umax)
        # Angular acceleration of the motor in rad/s^2:
        self.motoromega_dot_rad_per_sec2 = (
            self.vehicle_config.motor_maxOmega_rad_per_sec * action - self.motoromega_rad_per_sec
        ) / self.vehicle_config.motor_T
        # simulate the dynamic behavior of the motor/rotor RPM as first order lag ("PT1")
        self.motoromega_rad_per_sec += self.motoromega_dot_rad_per_sec2 * self.dt_sec
        # Reaaally make sure the motor RPM is only in the correct range:
        self.motoromega_rad_per_sec = np.clip(
            self.motoromega_rad_per_sec, 0.0, self.vehicle_config.motor_maxOmega_rad_per_sec
        )

        self.motor_thrust_N = (
            self.vehicle_config.CT * self.motoromega_rad_per_sec**2
        )  # vector of thrusts per motor in Newton
        self.motor_torque_Nm = (
            self.vehicle_config.CQ * self.motoromega_rad_per_sec**2
        )  # vector of torques per motor in Nm (without motor direction!)
        self.motor_power_mech_kw = 0.001 * np.sqrt(
            self.motor_thrust_N**3 / (2 * self.vehicle_config.rho * self.vehicle_config.rotor_area)
        )
        self.energy_burned_kwh += (
            np.sum(self.motor_power_mech_kw) / self.vehicle_config.efficiency_propulsion_system
        )
        # </MOTOR SIMULATION>

        self.internal_physics()

        # Compute a reward score (e.g. for reinforced learning)
        # -----------------------------------------------------
        reward = 0.0  # Implement a custom reward function here
        done = False  # Don't stop simulation

        self.time_sec = self.time_sec + self.dt_sec
        self.update_state()
        return self.state, reward, done, {}

    def internal_physics(self):
        # calculate the torque and thrust based on current motor speed (omega
        # in rad/s) with an u vector (u_effective) that corresponds to these RPMs
        u_effective = self.motoromega_rad_per_sec / self.vehicle_config.motor_maxOmega_rad_per_sec
        # exclude inactive motors:
        for inactive_motor in self.list_of_inactive_motors:
            u_effective[inactive_motor] = 0.0
        gamma = self.vehicle_config.M @ u_effective**2  # Note that u is squared here
        self.gamma = gamma
        torque_b = gamma[
            0:3
        ]  # current torque produced by rotors, acting on vehicle in Nm (in body frame)
        thrust = gamma[3]  # thrust produced by rotors along Z-axis in body frame

        # basic update attitude based on Euler's rigid body dynamics equations
        #       J*omega_dot + omega x J* omega = torque
        #
        # Limitation: the torque is
        # only produced from the motors and rotors, but e.g. torque from
        # aerodynamics is not considered here (but it could be added here)
        omega_cross = np.array(
            [
                [0, -self.omega[2], self.omega[1]],
                [self.omega[2], 0, -self.omega[0]],
                [-self.omega[1], self.omega[0], 0],
            ]
        )

        # <rotor inertia>
        e3 = np.array([0.0, 0.0, 1.0])
        total_spin = self.vehicle_config.motordirection.T @ self.motoromega_rad_per_sec
        hR = -self.vehicle_config.Jrotor * e3 * total_spin
        total_accel = self.vehicle_config.motordirection.T @ self.motoromega_dot_rad_per_sec2
        hR_dot = -self.vehicle_config.Jrotor * e3 * total_accel
        # </rotor inertia>

        omega_dot = self.vehicle_config.Jinv @ (
            torque_b - hR_dot - omega_cross @ (self.vehicle_config.J @ self.omega + hR)
        )
        omeganext = self.omega + omega_dot * self.dt_sec

        # update attitude (quaternion) based on the angular velocity omega
        delta = self.omega * self.dt_sec
        delta_abs = np.sqrt(delta @ delta)
        if delta_abs > 1e-9:
            img_part = delta / delta_abs * np.sin(delta_abs * 0.5)
            qr = np.block([np.cos(delta_abs * 0.5), img_part])
            qnext = quat_multiply(self.q, qr)
            qnext = quat_norm(qnext)
        else:
            qnext = self.q

        self.q = qnext
        self.omega = omeganext

        # Position/velocity update
        # Limitations:
        #     - Earth rotation rate is not compensated (incl. coriolis force)
        #     - Rotor plane wind resistance is neglegted
        thrust_body = np.array([0.0, 0.0, -thrust])
        a = (1.0 / self.vehicle_config.mass_kg) * thrust_body  # acceleration in body frame
        velrel_n = self.vel_n - self.wind_n  # relative air speed (m/s)
        velrel_n[2] = 0.0  # don't model rotor plane wind resistance
        velrel_n_norm = np.sqrt(velrel_n[0] ** 2 + velrel_n[1] ** 2 + velrel_n[2] ** 2)
        # self.vehicle_config.windresistance is c_D in paper (eq. \ref{eq:modelVorg})
        # Quadratic drag: F = -c_D * v * |v|
        ares_n = (
            -self.vehicle_config.windresistance
            * velrel_n
            * velrel_n_norm
            / self.vehicle_config.mass_kg
        )
        R = quat_to_matrix(self.q)
        # basic differential equation (Wendel, 2nd ed. equation 3.151):
        # dvdt_n is the acceleration of the body in n-frame (m/s/s). The centrifugal force
        # is assumed to be included in the gravity_n vector:
        force_external_n = np.array([0.0, 0, 0])  # external force in n-frame (m/s/s)
        dvdt_n = R @ a + self.vehicle_config.gravity_n + ares_n + force_external_n
        # Info: what an accelerometer would measure in the body frame:
        # self.specific_force_n = dvdt_n - self.gravity_n
        # specific_force_b = R.transpose()@self.specific_force_n

        # Velocity change
        # ---------------

        # the velocity for the next epoch k+1:
        velnext_n = self.vel_n + dvdt_n * self.dt_sec

        # Position change
        # ---------------
        # Trapezoid integration, change in position in n-frame (meter) is:
        dpos = 0.5 * self.dt_sec * (self.vel_n + velnext_n)

        # Finally update position and velocity:
        self.vel_n = velnext_n
        self.pos_n += dpos

    def print_state(self):
        ident_str = ""
        print(
            "%sNED=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) gamma=(%7.0f,%7.0f,%6.0f,%7.0f)"
            % (
                ident_str,
                self.pos_n[0],
                self.pos_n[1],
                self.pos_n[2],
                self.vel_n[0],
                self.vel_n[1],
                self.vel_n[2],
                self.roll_deg,
                self.pitch_deg,
                self.yaw_deg,
                np.rad2deg(self.omega[0]),
                np.rad2deg(self.omega[1]),
                np.rad2deg(self.omega[2]),
                self.gamma[0],
                self.gamma[1],
                self.gamma[2],
                self.gamma[3],
            )
        )

    def get_render_info(self):
        """
        Helper function to get position and orientation of simulated object.
        :return 3x3 rotation matrix (body to navigation), position in local NED frame (north/east/down)
        """
        rotationBody_to_NED = quat_to_matrix(self.q)
        return rotationBody_to_NED, self.pos_n

    def render(self):
        """
        Gym interface. Render current simulation status.
        """
        if not self.user_wants_to_render:
            self.user_wants_to_render = True
            self.render = RenderStlPygame(self.vehicle_config.model_file)
            self.render.init("")

        R, pos = self.get_render_info()
        self.render.render(R, pos)
