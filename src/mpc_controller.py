import numpy as np

from base_controller import BaseController

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, sqrt, sumsqr
from mpc_copter.copter_model_position import export_copterpos_ode_model
from mpc_copter.build_ocp import build_ocp

class NMPCController(BaseController):
    """
    Nonlinear Model Predictive Controller using Acados.
    """
    def __init__(self, vehicle_config):
        super().__init__(vehicle_config)
        ocp_cfg = vehicle_config.ocp_sim
        self.ocp, self.model, self.nx, self.nu, self.ny, self.N_horizon, self.Tf = build_ocp(vehicle_config, ocp_cfg)
        solver_json = "acados_ocp_" + self.model.name + ".json"
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)
        self.predictedX = np.ndarray((self.N_horizon, self.nx))

    def compute_control(self, state, keymap, dt_sec):
        self._update_references(state, keymap, dt_sec)

        # Solve OCP
        u = self.acados_ocp_solver.solve_for_x0(x0_bar=state)

        if self.model.ctrlout_u_is_squared:
            u = np.clip(u, self.vehicle_config.umin**2, self.vehicle_config.umax**2)
            u = np.sqrt(u)
        else:
            u = np.clip(u, self.vehicle_config.umin, self.vehicle_config.umax)

        for i in range(self.N_horizon):
            self.predictedX[i, :] = self.acados_ocp_solver.get(i, "x")

        # Update stats
        self.stats["time_tot"] = self.acados_ocp_solver.get_stats("time_tot")
        self.stats["sqp_iter"] = self.acados_ocp_solver.get_stats("sqp_iter")
        self.stats["qp_iter"] = self.acados_ocp_solver.get_stats("qp_iter")

        return u, self.predictedX

    def _update_references(self, state, keymap, dt_sec):
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

        update_vehicle_control_state(self.ctrl_state, cmd_b, pos, vel, q, dt_sec)

        if not np.isclose(yaw_cmd, 0.0):
            self.ctrl_state.yaw_rate_rps = yaw_cmd * self.vehicle_config.max_rotation_rate_rps
            self.ctrl_state.yaw_ref += self.ctrl_state.yaw_rate_rps * dt_sec
            self.ctrl_state.yaw_ref = angle_diff(self.ctrl_state.yaw_ref, 0.0)
        else:
            if not np.isclose(self.ctrl_state.yaw_rate_rps, 0.0):
                self.ctrl_state.yaw_ref = yaw
                self.ctrl_state.yaw_rate_rps = 0.0

        if np.allclose(cmd_b, 0.0):
            vel_n_ref = np.zeros(3)
        else:
            v_b = np.array([
                cmd_b[0] * self.vehicle_config.max_horizontal_velocity_mps,
                cmd_b[1] * self.vehicle_config.max_horizontal_velocity_mps,
                0.0,
            ])
            R_yaw = quat_to_matrix(quat_from_rpy(0.0, 0.0, yaw))
            vel_n_ref = R_yaw @ v_b
            vel_n_ref[2] = cmd_b[2] * self.vehicle_config.max_vertical_velocity_mps

        m = self.vehicle_config.mass_kg
        g = self.vehicle_config.gravity_n[2]
        v_norm = np.hypot(vel_n_ref[0], vel_n_ref[1])
        c_D = self.vehicle_config.windresistance
        F_x = c_D * vel_n_ref[0] * v_norm
        F_y = c_D * vel_n_ref[1] * v_norm
        roll_ref = np.arctan(F_y / (m * g))
        pitch_ref = -np.arctan(F_x / (m * g))
        q_ref = quat_from_rpy(roll_ref, pitch_ref, self.ctrl_state.yaw_ref)
        omega_ref = np.array([0.0, 0.0, self.ctrl_state.yaw_rate_rps])

        pos_ref = self.ctrl_state.desired_pos
        yref = np.copy(self.ocp.cost.yref)
        yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_ref
        yref[cfg["vel3d_index"]:cfg["vel3d_index_end"]] = vel_n_ref
        yref[cfg["q_index"]:cfg["q_index_end"]] = q_ref
        yref[cfg["omega_index"]:cfg["omega_index_end"]] = omega_ref
        self.ocp.cost.yref = np.copy(yref)

        N = self.ocp.solver_options.N_horizon
        pos_pred = np.zeros((N + 1, 3))
        pos_pred[0] = pos_ref
        for j in range(1, N + 1):
            pos_pred[j] = pos_pred[j - 1] + vel_n_ref * dt_sec

        for j in range(N):
            yref[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[j]
            self.acados_ocp_solver.set(j, "yref", np.copy(yref))

        yref_e = np.copy(yref[:state.size])
        yref_e[cfg["pos3d_index"]:cfg["pos3d_index_end"]] = pos_pred[N]
        self.acados_ocp_solver.set(N, "yref", yref_e)
        self.ocp.cost.yref_e = yref_e


