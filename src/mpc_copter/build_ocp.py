"""
build_ocp.py - Shared OCP-Builder-Function
Used by main.py (simulation) and generate_embedded.py (embedded).
Reads data from vehicle_config and ocp_cfg, builds and returns the AcadosOcp-object.

Model selection
---------------
By default the position-output model (export_copterpos_ode_model) is used, which
matches the historical behavior. To build an OCP for a different model variant
(e.g. the rate-output model used by the hybrid NMPC + rate-PID controller),
pass a `model_factory` callable that takes `vehicle_config` and returns an
AcadosModel:

    from mpc_copter.copter_model_rates import export_copterrates_ode_model
    ocp, model, nx, nu, ny, N, Tf = build_ocp(
        vehicle_config, ocp_cfg, model_factory=export_copterrates_ode_model
    )

Control-input bounds adapt to the model's `ctrlout_u_is_squared` flag:
  - True  -> u represents squared normalized commands, bounds are [umin^2, umax^2]
  - False -> u is the raw normalized command in [umin, umax]
"""

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp


def build_ocp(vehicle_config, ocp_cfg, model_factory=None):
    """
    Build AcadosOcp-object from vehicle_config and ocp_cfg.

    Args:
        vehicle_config : CopterConfig-Instance
        ocp_cfg        : OcpConfig-Instance (e.g. vehicle_config.ocp_sim
                         or vehicle_config.ocp_embedded)
        model_factory  : Optional callable (vehicle_config) -> AcadosModel.
                         If None, the position-output model is used (legacy
                         behavior). Pass export_copterrates_ode_model for the
                         rate-output variant.

    Returns:
        ocp, model, nx, nu, ny, N_horizon, Tf
    """
    ocp = AcadosOcp()

    if model_factory is None:
        # Local import keeps the rate-output build path independent of the
        # position-output model file.
        from mpc_copter.copter_model_position import export_copterpos_ode_model
        model = export_copterpos_ode_model(vehicle_config)
    else:
        model = model_factory(vehicle_config)

    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # --- Horizon ---
    if ocp_cfg.shooting_nodes is not None:
        # Non-uniform grid
        nodes     = np.asarray(ocp_cfg.shooting_nodes)
        N_horizon = len(nodes) - 1
        Tf        = float(nodes[-1])
        ocp.solver_options.shooting_nodes = nodes
    else:
        # Fixed step size
        N_horizon = ocp_cfg.N_horizon
        Tf        = ocp_cfg.Tf

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf        = Tf

    # --- Costs ---
    weight_diag = model.weight_diag.copy()
    # Override omega weights from ocp_cfg (if specified, otherwise use model defaults)
    if ocp_cfg.weight_omega_roll is not None:
        weight_diag[vehicle_config.state_cfg["omega_roll_index"]]  = ocp_cfg.weight_omega_roll
    if ocp_cfg.weight_omega_pitch is not None:
        weight_diag[vehicle_config.state_cfg["omega_pitch_index"]] = ocp_cfg.weight_omega_pitch
    if ocp_cfg.weight_omega_yaw is not None:
        weight_diag[vehicle_config.state_cfg["omega_yaw_index"]]   = ocp_cfg.weight_omega_yaw
    Q_mat = np.diag(weight_diag)

    R_mat = np.diag(np.ones(nu) * model.cost_u_weight)
    # R_mat[0,0] = 1e1 # example weight adjustment: be easy on motor #0

    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W           = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e         = Q_mat

    ocp.cost.Vx           = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu                    = np.zeros((ny, nu))
    Vu[nx:nx + nu, 0:nu]  = np.eye(nu)
    ocp.cost.Vu           = Vu
    ocp.cost.Vx_e         = np.eye(nx)

    yref = np.zeros((ny,))
    yref[vehicle_config.state_cfg["q_index"]]        = 1.0
    yref[vehicle_config.state_cfg["altitude_index"]] = 0.0
    ocp.cost.yref   = yref
    ocp.cost.yref_e = yref[:nx]

    # --- Constraints ---
    # Control bounds depend on whether u carries squared or raw normalized
    # commands. The model advertises this via the ctrlout_u_is_squared flag.
    ocp.constraints.constr_type = "BGH"
    if getattr(model, "ctrlout_u_is_squared", True):
        # Position-output model: u = (omega_m / omega_max)^2 in [umin^2, umax^2]
        ocp.constraints.lbu = np.ones(nu) * vehicle_config.umin ** 2
        ocp.constraints.ubu = np.ones(nu) * vehicle_config.umax ** 2
    else:
        # Rate-output model (and legacy "motor speed in state" variant):
        # u = normalized motor command in [umin, umax]
        ocp.constraints.lbu = np.ones(nu) * vehicle_config.umin
        ocp.constraints.ubu = np.ones(nu) * vehicle_config.umax
    ocp.constraints.idxbu = np.arange(nu)

    x0 = np.zeros(nx)
    x0[vehicle_config.state_cfg["q_index"]] = 1.0
    ocp.constraints.x0 = x0

    # --- Soft Rate Constraints ---
    if ocp_cfg.soft_rate_constraints:
        cfg       = vehicle_config.state_cfg
        omega_idx = list(range(cfg["omega_index"], cfg["omega_index_end"]))
        max_rr    = vehicle_config.max_rotation_rate_rps
        ns        = len(omega_idx)

        ocp.constraints.idxbx  = np.array(omega_idx)
        ocp.constraints.lbx    = np.full(ns, -max_rr)
        ocp.constraints.ubx    = np.full(ns,  max_rr)
        ocp.constraints.idxsbx = np.arange(ns)
        ocp.constraints.lsbx   = np.zeros(ns)
        ocp.constraints.usbx   = np.zeros(ns)
        ocp.cost.Zl = ocp_cfg.soft_rate_constraint_L2 * np.ones(ns)
        ocp.cost.Zu = ocp_cfg.soft_rate_constraint_L2 * np.ones(ns)
        ocp.cost.zl = ocp_cfg.soft_rate_constraint_L1 * np.ones(ns)
        ocp.cost.zu = ocp_cfg.soft_rate_constraint_L1 * np.ones(ns)

    # --- Integrator ---
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.num_stages      = ocp_cfg.num_stages  # e.g. 4 for classic Runge-Kutta
    ocp.solver_options.num_steps       = ocp_cfg.num_steps   # per shooting interval

    # --- QP-Solver ---
    ocp.solver_options.qp_solver = ocp_cfg.qp_solver
    # cond_N = N_horizon -> no condensing   (= sparse QP, many small blocks)
    # cond_N = 0         -> full condensing (= single large dense QP)
    # cond_N = N/2       -> half condensing (compromise, potentially good for embedded)
    cond_N = ocp_cfg.qp_solver_cond_N if ocp_cfg.qp_solver_cond_N is not None \
             else N_horizon
    ocp.solver_options.qp_solver_cond_N     = cond_N
    ocp.solver_options.qp_solver_warm_start = ocp_cfg.qp_warm_start
    ocp.solver_options.qp_solver_iter_max   = ocp_cfg.qp_iter_max
    ocp.solver_options.hessian_approx       = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type      = "SQP_RTI"

    return ocp, model, nx, nu, ny, N_horizon, Tf
