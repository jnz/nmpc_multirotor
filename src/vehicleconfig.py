"""
Vehicle Definition

(c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
import matplotlib.pyplot as plt

# This class describes the attributes of a vehicle (mass, moment of inertia,
# actuator positions, .stl file path for the renderer, etc.).
# The vehicle is defined in a NED frame (North, East, Down).


class CopterConfig:
    def __init__(self):

        # Basic model

        # Thrust in Newton per motor i depends on rotation rate of motor/rotor "motorOmega_i" squared,
        # the rotor area and the air density rho:
        #   F_i = rho * A * Cl * motorOmega_i^2                    (1)
        #
        # Rotation rate of motor/rotor depends on the max. rotation rate and
        # the control input u (between 0.0 and 1.0):
        #   motorOmega_i = maxOmega * u_i                          (2)
        #
        # So in total:
        #   F_i = rho * A * Cl * maxOmega^2 * u_i^2                (3)
        #
        # So the total max. thrust is:
        #   F_i_max = rho * A * Cl * maxOmega^2 * 1.0              (4)
        #
        # The same logic applies for the torque T_i along the motor axis i:
        #   T_i = rho * A * Cd * maxOmega^2 * u_i^2                (5)
        # -----------------------------------------------------------------------------------------

        # MPC default weights (might be overwritten below per vehicle)
        # <weights>
        self.weight_q = 16.0
        self.weight_omega_roll = 1.0
        self.weight_omega_pitch = 1.0
        self.weight_omega_yaw = 1.0

        self.weight_north_east = 0.2
        self.weight_altitude = 3.2
        self.weight_velocity_horizontal = 8.0
        self.weight_velocity_vertical = 8.0
        # </weights>

        self.umin = 0.15  # box constaints on control input (min. u)
        self.umax = 1.00  # box constaints on control input (max. u)
        self.rho = 1.225  # air density (kg/m^3)
        self.gravity_n = np.array([0, 0, 9.81])  # gravity in NED frame (m/s^2)

        # Default efficiency
        self.efficiency_propulsion_system = (
            0.92  # elect. power * efficiency_propulsion_system = mech. power
        )
        self.motor_blockage_factor = 1.0  # 1.0 means by default no thrust is lost in the structure

        vehicle = 18  # default vehicle has 18 rotors

        if vehicle == 18:
            # <manned multirotor aircraft>
            self.vehicle_name = "rotor18"
            self.model_file = "evtol.stl"
            self.center_of_mass_b = np.array(
                [0.0, 0.0, -0.8]
            )  # center_of_mass_b + motortable_position = position of motor wrt. CoG. Z component is typically negative
            self.model_offset = np.array([0.0, 0.0, 0.0])  # drawing offset of model
            self.mass_kg = 975.0  # total vehicle mass in kg
            self.motortable = np.array(
                [
                    [4.47, 1.19],
                    [2.07, 1.19],
                    [3.26, 3.26],
                    [1.19, 4.46],
                    [0.00, 2.38],
                    [-1.19, 4.46],
                    [-3.27, 3.26],
                    [-2.07, 1.19],
                    [-4.47, 1.19],
                    [-4.47, -1.19],
                    [-2.07, -1.19],
                    [-3.26, -3.26],
                    [-1.19, -4.46],
                    [0.00, -2.38],
                    [1.19, -4.46],
                    [3.27, -3.26],
                    [2.07, -1.19],
                    [4.47, -1.19],
                ]
            )  # motor positions in vehicle body frame (north / east)

            self.motordirection = np.array(
                [
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                ]
            )  # spinning direction of motors, must have same length as motortable
            self.motorcount = self.motortable.shape[0]  # number of motors
            self.motor_T = 0.3  # motor time constant (PT1 time delay)
            self.motor_maxOmega_rad_per_sec = 200.0  # max. rotation rate of rotor in rad/s
            self.motorinclination_rad = np.deg2rad(
                6.0
            )  # motor inclination to increase torque along yaw-axis
            self.rotor_area = 4.5  # m**2 rotor area of individual rotor
            self.motor_blockage_factor = 0.88  # 12% loss of thrust due to structural blockage.
            self.Cl = self.motor_blockage_factor * 0.0041  # lift coefficient
            self.Cd = 0.00031  # rotor drag coefficient
            self.J = np.array(
                [[2000.0, 0.0, 0.0], [0.0, 2200.0, 0.0], [0.0, 0.0, 3500.0]]
            )  # moment of inertia
            # Motor rotor of inertia
            self.Jrotor = 1.0  # kg*m^2
            self.max_horizontal_velocity_mps = 25.0  # max vertical velocity of the vehicle in m/s
            self.max_vertical_velocity_mps = 5.0  # max vertical velocity of the vehicle in m/s
            self.windresistance = (
                8.0  # wind resistance coefficient = area (m**2) * cw * rho (kg/m**3)
            )
            self.max_rotation_rate_rps = np.deg2rad(20.0)  # max. rotation rate in rad/s
            # </manned multirotor aircraft>
        elif vehicle == 4:
            # <quadcopter>
            self.vehicle_name = "Quadcopter"
            self.model_file = "quadcopter.stl"
            self.mass_kg = 0.570  # total vehicle mass in kg
            self.model_offset = np.array([0.0, 0.0, 0.0])
            self.center_of_mass_b = np.array(
                [0.0, 0.0, 0.07]
            )  #  # center_of_mass_b + motortable_position = position of motor wrt. CoG. Z component is typically negative
            self.motortable = np.array(
                [
                    [0.0884, 0.0884],  # front/right motor (blue)
                    [0.0884, -0.0884],  # front/left motor (red)
                    [-0.0884, -0.0884],
                    [-0.0884, 0.0884],
                ]
            )  # motor positions in vehicle body frame (north / east)
            self.motordirection = np.array(
                [1.0, -1.0, 1.0, -1.0]
            )  # spinning direction of motors, must have same length as motortable
            self.motorcount = self.motortable.shape[0]  # number of motors
            self.motor_T = 0.3  # motor time constant (PT1 time delay)
            self.motor_maxOmega_rad_per_sec = 523.0  # max. rotation rate of rotor in rad/s
            self.motorinclination_rad = np.deg2rad(
                0.0
            )  # motor inclination to increase torque along yaw-axis
            self.rotor_area = 0.024  # m**2 rotor area of individual rotor
            self.Cl = self.motor_blockage_factor * 0.00036  # lift coefficient
            self.Cd = self.Cl * 0.1  # drag coefficient of rotor (not vehicle)
            self.J = np.array(
                [
                    [0.57 / 12.0 * (0.253**2 + 0.077**2), 0.0, 0.0],
                    [0.0, 0.57 / 12.0 * (0.183**2 + 0.077**2), 0.0],
                    [0.0, 0.0, 0.57 / 12.0 * (0.183**2 + 0.253**2)],
                ]
            )  # moment of inertia
            self.Jrotor = 0.0001  # kg*m^2
            self.max_horizontal_velocity_mps = 5.0  # max horizontal velocity of the vehicle in m/s
            self.max_vertical_velocity_mps = 3.0  # max vertical velocity of the vehicle in m/s
            # estimating the exposed area to 0.04 m**2 and the drag coefficient to be 0.5
            self.windresistance = (
                0.04 * 0.5 * self.rho
            )  # wind resistance coefficient = area (m**2) * cw * rho (kg/m**3)
            self.max_rotation_rate_rps = np.deg2rad(100.0)  # max. rotation rate in rad/s
            self.weight_q = 0.2
            self.weight_omega_roll = 1.0
            self.weight_omega_pitch = 1.0
            self.weight_omega_yaw = 10.0

            self.weight_north_east = 8.0
            self.weight_altitude = 32.2
            self.weight_velocity_horizontal = 8.0
            self.weight_velocity_vertical = 64.0
            # </quadcopter>

        else:
            assert False, "Unknown vehicle type %i" % (vehicle)

        # -----------------------------------------------------------------------------------------

        assert self.rotor_area > 0.0, "rotor area must be > 0.0"
        assert self.rho > 0.0, "air density must be > 0.0"

        self.CT = self.rotor_area * self.rho * self.Cl  # rotor thrust Ti = CT*omega_rad_per_sec^2
        self.CQ = self.rotor_area * self.rho * self.Cd  # rotor torque Qi = CQ*omega_rad_per_sec^2

        self.motormax_thrust_per_motor_N = (
            self.CT * self.motor_maxOmega_rad_per_sec**2
        )  # equation (4)
        self.motormax_torque_per_motor_Nm = (
            self.CQ * self.motor_maxOmega_rad_per_sec**2
        )  # equation (5)

        # M is the "motor matrix", note that M is multiplied by u-squared
        # gamma (torque 3x1, thrust) = M*u^2
        self.M = control_matrix(
            self.motortable,
            self.motordirection,
            self.motorinclination_rad,
            self.motormax_thrust_per_motor_N,
            self.motormax_torque_per_motor_Nm,
            self.center_of_mass_b,
        )

        self.Jinv = np.linalg.inv(self.J)

        max_rpm = self.umax * self.motor_maxOmega_rad_per_sec * 60.0 / (2.0 * np.pi)
        max_commandable_thrust = self.CT * ((self.umax * self.motor_maxOmega_rad_per_sec) ** 2)
        max_commandable_torque = self.CQ * ((self.umax * self.motor_maxOmega_rad_per_sec) ** 2)

        self.thrust_to_weight_ratio = (
            max_commandable_thrust * self.motorcount / self.gravity_n[2]
        ) / self.mass_kg

        print(
            "Active vehicle config: %s with %i motors and %.2f kg MTOM. Thrust/weight ratio: %.1f"
            % (self.vehicle_name, self.motorcount, self.mass_kg, self.thrust_to_weight_ratio)
        )
        print("Max. commandable thrust per motor: %.1f N" % (max_commandable_thrust))
        print("Max. commandable torque per motor: %.1f Nm" % (max_commandable_torque))
        print("Max. commandable (umax) RPM: %.1f RPM." % (max_rpm))
        assert (
            self.thrust_to_weight_ratio > 1.0
        ), "Aircraft must have at least enough thrust to hover"

        # Configure the expected state vector for this vehicle
        # This struct is a dictionary what information is at which
        # position in the state vector. The simulation will also use this
        # dictionary to supply a state vector according to this configuration.
        self.state_cfg = {
            "q_index": 0,  # attitude quaternion
            "q_index_end": 4,
            "omega_index": 4,  # rotation rates in body frame
            "omega_index_end": 7,
            "omega_roll_index": 4,
            "omega_pitch_index": 5,
            "omega_yaw_index": 6,
            "pos3d_available": True,  # position in NED
            "pos3d_index": 7,
            "pos3d_index_end": 10,
            "north_index": 7,
            "east_index": 8,
            "altitude_available": True,  # basically ignored if a full 3d position is available
            "altitude_index": 9,
            "vel3d_available": True,  # vel NED in m/s
            "vel3d_index": 10,
            "vel3d_index_end": 13,
            "vel_north_index": 10,
            "vel_east_index": 11,
            "vel_down_index": 12,  # if altitude_available is True, vel_down is also available
            # If gamma_available is set to true, the controller
            # gets the 4x1 gamma vector which includes the current torque acting on the
            # aircraft body and the thrust from the rotor plane.
            "gamma_available": True,  # torque thrust vector available for state
            "gamma_index": 13,
            "gamma_index_end": 17,
            # The alternative to gamma_available (mutually exclusive)
            # is that the current rotor speeds are added to the state
            "rotoromega_available": False,
            # 'rotoromega_index'      : 13,
            # 'rotoromega_index_end'  : 13 + self.motorcount,
        }
        # automatically calculate the state vector size based on the highest
        # index that ends with '_end':
        max_index = max([v for k, v in self.state_cfg.items() if k.endswith("_end")])
        self.state_cfg["state_length"] = max_index
        assert (
            self.state_cfg["gamma_available"] == True
            and self.state_cfg["rotoromega_available"] == False
        ) or (
            self.state_cfg["gamma_available"] == False
            and self.state_cfg["rotoromega_available"] == True
        ), "gamma_available is mutually exclusive to rotoromega_available"


def get_vehicle_config():
    return CopterConfig()


def control_matrix(
    motor_positions_north_east,
    motor_direction,
    motor_inclination_rad,
    max_thrust_per_motor_N,
    max_torque_per_motor_Nm,
    cg_shift_m,
):
    # Constructs the control effectiveness matrix K mapping squared motor inputs to total torque and thrust.
    # gamma = K * u^2
    # where gamma = [τx, τy, τz, Fz]ᵀ (torques and thrust), u ∈ [0,1]^n (actuator inputs)
    # Args:
    #     motor_positions_north_east: (n x 2) array of motor positions in the rotor plane (body frame)
    #     motor_direction: (n,) array of +1 (CW) or -1 (CCW) from top view
    #     motor_inclination_rad: scalar inclination of motor thrust vector (used to generate yaw torque)
    #     max_thrust_per_motor_N: scalar max thrust per motor at u = 1.0
    #     max_torque_per_motor_Nm: scalar max torque per motor at u = 1.0
    #     cg_shift_m: (3,) vector shift of rotor plane origin relative to CoG

    # Returns:
    #     K: (4 x n) motor matrix that maps squared inputs u² to gamma (torques and thrust)

    n = motor_positions_north_east.shape[0]
    K = np.zeros((4, n))

    for i in range(n):
        angle_motor_rad = np.arctan2(
            motor_positions_north_east[i, 1], motor_positions_north_east[i, 0]
        )
        thrust_axis = np.array(
            [
                -np.sin(angle_motor_rad) * np.sin(motor_inclination_rad * motor_direction[i]),
                np.cos(angle_motor_rad) * np.sin(motor_inclination_rad * motor_direction[i]),
                -np.cos(motor_inclination_rad),
            ]
        )
        # position of motor wrt CoG
        r_i = np.array(
            [
                motor_positions_north_east[i, 0] + cg_shift_m[0],
                motor_positions_north_east[i, 1] + cg_shift_m[1],
                cg_shift_m[2],
            ]
        )
        # force vector of actuator
        f_i = thrust_axis * max_thrust_per_motor_N
        # torque vector
        t_i = -motor_direction[i] * thrust_axis * max_torque_per_motor_Nm
        K[0:3, i] = np.cross(r_i, f_i) + t_i
        K[3, i] = -f_i[2]

    return K


if __name__ == "__main__":
    config = get_vehicle_config()
    print(config.M)
    print(config.J)
    xrange = np.arange(config.umin, config.umax, 0.1)
    yrange = np.copy(xrange)
    for i in range(len(xrange)):
        gamma = config.M @ np.ones((config.motorcount, 1)) * (xrange[i] ** 2)
        yrange[i] = gamma[3]
        print(
            "u = %4.2f --> thrust %6.1f N (per motor: %6.1f)"
            % (xrange[i], gamma[3], gamma[3] / config.motorcount)
        )

    plt.figure()
    plt.plot(xrange, yrange)
    plt.xlabel("u")
    plt.ylabel("Total thrust in Newton")
    plt.title("Total thrust (all motors) over control input u")
    plt.grid(True)
    plt.show()
    input("Press enter to close plot")
