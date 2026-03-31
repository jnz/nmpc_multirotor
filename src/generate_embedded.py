#!/usr/bin/env python3
"""
generate_embedded.py  -  acados NMPC Embedded Code Generator
=============================================================

Project layout expected:
  nmpc_multirotor/
    acados/                          acados submodule
    src/
      generate_embedded.py           <- this file
      main.py
      vehicleconfig.py
      mpc_copter/
        copter_model_position.py

Generates:
  <outdir>/
    acados_ocp_<model>.json          acados OCP definition
    c_generated_code/                acados solver C sources  (platform-independent)
    nmpc_copter.h                    clean embedded API header
    nmpc_copter.c                    wrapper implementation
    nmpc_copter_config.h             vehicle constants as #defines
    nmpc_example_usage.c             integration example
    Makefile                         builds acados static libs for the chosen target
    README.md                        full build + integration guide

Usage (run from project root with venv active):
  source env.sh
  python src/generate_embedded.py --target stm32n6
  python src/generate_embedded.py --target rpi4   --vehicle 4
  python src/generate_embedded.py --target linux  --outdir build/sitl
  python src/generate_embedded.py --list-targets

(c) Jan Zwiener (jan@zwiener.org)
"""

import argparse
import os
import sys
import textwrap
import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                  # .../nmpc_multirotor
ACADOS_ROOT  = os.path.join(PROJECT_ROOT, "acados")

sys.path.insert(0, SCRIPT_DIR)

from acados_template import AcadosOcp, AcadosOcpSolver

try:
    from mpc_copter.copter_model_position import export_copterpos_ode_model
    from vehicleconfig import CopterConfig
    from mpc_copter.build_ocp import build_ocp as _build_ocp
except ImportError as e:
    print("ERROR: Could not import project modules: %s" % e)
    print("Make sure the venv is active:  source env.sh")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Target configuration table
#
# The generated C sources (c_generated_code/, nmpc_copter.*) are IDENTICAL
# for all targets.  Only the static libs need to be rebuilt per target.
#
# Fields:
#   description      human-readable label shown in --list-targets
#   blasfeo_target   passed to cmake -DBLASFEO_TARGET=...
#   hpipm_target     EMBEDDED (bare-metal) or PC (Linux/RTOS with malloc)
#   cc               C cross-compiler executable
#   ar               archiver executable
#   c_flags          compiler flags for the target hardware
#   c_defs           preprocessor defines to add to the application build
#   notes            optional hint printed after generation
# ---------------------------------------------------------------------------
TARGETS = {
    "stm32n6": dict(
        description    = "STM32N6  (Cortex-M55, FPv5-D16, 600 MHz)",
        # BLASFEO has no Cortex-M kernels; GENERIC = plain C, works everywhere.
        # cmake_c_flags go to the acados cmake build only - no -mcpu here because
        # BLASFEO GENERIC sets its own -march internally and mixing them causes
        # a conflict warning that breaks some gcc versions.
        blasfeo_target = "GENERIC",
        hpipm_target   = "EMBEDDED",
        cc             = "arm-none-eabi-gcc",
        ar             = "arm-none-eabi-ar",
        # -w suppresses the harmless "-mcpu conflicts with -march" warning
        # that BLASFEO GENERIC emits when it appends its own -march internally.
        # All objects must be Thumb-mode — Cortex-M has no ARM mode.
        # -mthumb-interwork ensures consistent ARM/Thumb calling convention.
        cmake_c_flags  = "-mcpu=cortex-m55 -mthumb -mthumb-interwork -mfloat-abi=hard -mfpu=fpv5-d16 -O3 -g -ffast-math -w",
        c_flags        = "-mcpu=cortex-m55 -mthumb -mthumb-interwork -mfloat-abi=hard -mfpu=fpv5-d16"
                         " -O3 -g -ffast-math -ffunction-sections -fdata-sections",
        c_defs         = "-DBLASFEO_TARGET_GENERIC"
                         " -DHPIPM_TARGET_EMBEDDED -DACADOS_SILENT",
        notes          = "Link with: -Wl,--gc-sections  --specs=nosys.specs",
    ),
    "rpi3": dict(
        description    = "Raspberry Pi 3 (Cortex-A53, AArch64, Linux)",
        blasfeo_target = "ARMV8A_ARM_CORTEX_A53",
        hpipm_target   = "PC",
        cc             = "aarch64-linux-gnu-gcc",
        ar             = "aarch64-linux-gnu-ar",
        c_flags        = "-mcpu=cortex-a53 -O3 -ffast-math",
        c_defs         = "-DBLASFEO_TARGET_ARMV8A_ARM_CORTEX_A53",
        notes          = "Runs Linux - malloc/printf available natively.",
    ),
    "rpi4": dict(
        description    = "Raspberry Pi 4  (Cortex-A72, AArch64, Linux)",
        blasfeo_target = "ARMV8A_ARM_CORTEX_A57",
        hpipm_target   = "PC",
        cc             = "aarch64-linux-gnu-gcc",
        ar             = "aarch64-linux-gnu-ar",
        c_flags        = "-mcpu=cortex-a72 -O3 -ffast-math",
        c_defs         = "-DBLASFEO_TARGET_ARMV8A_ARM_CORTEX_A57",
        notes          = "Runs Linux - malloc/printf available natively.",
    ),
    "rpi5": dict(
        description    = "Raspberry Pi 5  (Cortex-A76, AArch64, Linux)",
        blasfeo_target = "ARMV8A_ARM_CORTEX_A57",   # A76 not yet in BLASFEO; A57 profile works
        hpipm_target   = "PC",
        cc             = "aarch64-linux-gnu-gcc",
        ar             = "aarch64-linux-gnu-ar",
        c_flags        = "-mcpu=cortex-a76 -O3 -ffast-math",
        c_defs         = "-DBLASFEO_TARGET_ARMV8A_ARM_CORTEX_A57",
        notes          = "Runs Linux - malloc/printf available natively.",
    ),
    "linux": dict(
        description    = "Linux / PC  (x86-64)",
        blasfeo_target = "X64_AUTOMATIC",
        hpipm_target   = "PC",
        cc             = "gcc",
        ar             = "ar",
        c_flags        = "-O3 -ffast-math -march=native",
        c_defs         = "-DBLASFEO_TARGET_X64_AUTOMATIC",
        notes          = "Useful for SITL or rapid prototyping on the host machine.",
    ),
}

DEFAULT_TARGET  = "stm32n6"
DEFAULT_VEHICLE = 18


# ---------------------------------------------------------------------------
# OCP builder
# ---------------------------------------------------------------------------


def build_ocp(vehicle_config):
    """
    Build the AcadosOcp object from the vehicle config.
    The OCP definition is needed to generate the acados solver C code
    (c_generated_code/), which is platform-independent. The same OCP definition
    is also used in main.py for the simulation, ensuring consistency.

    Return values:
     ocp, model, nx, nu, ny, N_horizon, Tf
    """
    return _build_ocp(vehicle_config, vehicle_config.ocp_embedded)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_to_c_rows(M, indent="    "):
    rows = []
    for row in M:
        vals = ", ".join("%.10g" % v for v in row)
        rows.append("%s{ %s }" % (indent, vals))
    return ",\n".join(rows)

def array_to_c_row(arr, indent="    "):
    return indent + ", ".join("%.10g" % v for v in arr)


# ---------------------------------------------------------------------------
# File templates
# ---------------------------------------------------------------------------

HEADER_TEMPLATE = """\
/**
 * nmpc_copter.h  -  Embedded NMPC interface
 *
 * AUTO-GENERATED by generate_embedded.py  -  DO NOT EDIT
 *
 * Vehicle  : {vehicle_name}
 * Target   : {target_id}  ({target_description})
 * Motors   : {motorcount}
 * States   : {nx}  (NX)
 * Inputs   : {nu}  (NU)
 * Horizon  : {N_horizon} stages over {Tf} s
 * Rate     : {mpc_hz} Hz
 *
 * Quick start
 * -----------
 *   NmpcCopter_t ctrl;
 *   double x0[NMPC_NX] = {{0}}; x0[NMPC_IDX_Q0] = 1.0;
 *   nmpc_init(&ctrl, x0);
 *
 *   // in {mpc_hz} Hz task:
 *   NmpcMeasurement_t meas = {{ ... }};
 *   NmpcSetpoint_t    sp   = {{ .pos_ned = {{0,0,-5}}, .q_ref = {{1,0,0,0}} }};
 *   double u[NMPC_MOTORCOUNT];
 *   int status = nmpc_step(&ctrl, &meas, &sp, u);
 *   // u[i] in [0..1]  ->  motor throttle
 */

#pragma once
#include <stdint.h>

/* --- Dimensions --------------------------------------------------------- */
#define NMPC_NX         {nx}         /**< State vector length              */
#define NMPC_NU         {nu}         /**< Control inputs  (== motors)      */
#define NMPC_NY         {ny}         /**< Cost ref length (NX + NU)        */
#define NMPC_N_HORIZON  {N_horizon}  /**< Prediction horizon stages        */
#define NMPC_MOTORCOUNT {motorcount} /**< Number of motors                 */
#define NMPC_MPC_HZ     {mpc_hz}     /**< Nominal update rate [Hz]         */

/* --- State vector indices  (mirrors state_cfg in vehicleconfig.py) ------ */
#define NMPC_IDX_Q0          {q_index}       /**< Quaternion q0 (real part)  */
#define NMPC_IDX_Q_END       {q_index_end}
#define NMPC_IDX_OMEGA       {omega_index}   /**< Body rot rates [rad/s]     */
#define NMPC_IDX_OMEGA_END   {omega_index_end}
#define NMPC_IDX_POS         {pos3d_index}   /**< NED position [m]           */
#define NMPC_IDX_POS_END     {pos3d_index_end}
#define NMPC_IDX_VEL         {vel3d_index}   /**< NED velocity [m/s]         */
#define NMPC_IDX_VEL_END     {vel3d_index_end}
#define NMPC_IDX_GAMMA       {gamma_index}   /**< Torque + thrust state      */
#define NMPC_IDX_GAMMA_END   {gamma_index_end}

/* --- Input structures --------------------------------------------------- */

/** Fill from sensor fusion / AHRS output every cycle. */
typedef struct {{
    double q[4];        /**< Attitude quaternion [q0 q1 q2 q3] body->NED  */
    double omega_b[3];  /**< Body rotation rates [rad/s] roll/pitch/yaw   */
    double pos_ned[3];  /**< NED position [m]                             */
    double vel_ned[3];  /**< NED velocity [m/s]                           */
    double gamma[4];    /**< Torque (3) + thrust (1)  [Nm Nm Nm N]       */
}} NmpcMeasurement_t;

/** Desired flight state. Zero-initialise and fill only what you need. */
typedef struct {{
    double pos_ned[3];   /**< Desired NED position [m]                    */
    double vel_ned[3];   /**< Desired NED velocity [m/s]  (0 = hold)      */
    double q_ref[4];     /**< Desired quaternion  (level = {{1,0,0,0}})    */
    double omega_ref[3]; /**< Desired body rates  [rad/s] (0 = hover)     */
}} NmpcSetpoint_t;

/* --- Controller handle  (treat as opaque) ------------------------------- */
typedef struct {{
    void  *capsule;
    double yref  [NMPC_NY];
    double yref_e[NMPC_NX];
    int    initialized;
}} NmpcCopter_t;

/* --- API ---------------------------------------------------------------- */

/**
 * Initialize the NMPC controller. Call once at startup.
 * @param ctrl  Caller-owned NmpcCopter_t instance.
 * @param x0    Initial state vector [NMPC_NX]. Use all-zero with q0=1 if unknown.
 * @return      0 on success, negative on error.
 */
int nmpc_init(NmpcCopter_t *ctrl, const double *x0);

/**
 * Run one SQP-RTI step. Call at NMPC_MPC_HZ ({mpc_hz} Hz).
 * @param ctrl   Initialized controller handle.
 * @param meas   Current vehicle state from sensor fusion.
 * @param sp     Desired setpoint.
 * @param u_out  Motor throttle commands [0..1], length NMPC_MOTORCOUNT.
 * @return       0=optimal  1=max-iter (output still valid)  <0=fatal error.
 */
int nmpc_step(NmpcCopter_t            *ctrl,
              const NmpcMeasurement_t *meas,
              const NmpcSetpoint_t    *sp,
              double                  *u_out);

/** Free all resources allocated by nmpc_init(). */
void nmpc_free(NmpcCopter_t *ctrl);

/** Pack NmpcMeasurement_t into a raw state vector (useful for nmpc_init / logging). */
void nmpc_measurement_to_state(const NmpcMeasurement_t *meas, double *x_out);
"""

# --------------------------------------------------------------------------

SOURCE_TEMPLATE = """\
/**
 * nmpc_copter.c  -  Embedded NMPC wrapper
 *
 * AUTO-GENERATED by generate_embedded.py  -  DO NOT EDIT
 *
 * Vehicle : {vehicle_name}
 * Target  : {target_id}
 */

#include "nmpc_copter.h"
#include "c_generated_code/acados_solver_{model_name}.h"
#include <math.h>
#include <string.h>

/* ---- internal helpers -------------------------------------------------- */

void nmpc_measurement_to_state(const NmpcMeasurement_t *meas, double *x)
{{
    memset(x, 0, NMPC_NX * sizeof(double));
    x[NMPC_IDX_Q0 + 0] = meas->q[0];
    x[NMPC_IDX_Q0 + 1] = meas->q[1];
    x[NMPC_IDX_Q0 + 2] = meas->q[2];
    x[NMPC_IDX_Q0 + 3] = meas->q[3];
    x[NMPC_IDX_OMEGA + 0] = meas->omega_b[0];
    x[NMPC_IDX_OMEGA + 1] = meas->omega_b[1];
    x[NMPC_IDX_OMEGA + 2] = meas->omega_b[2];
    x[NMPC_IDX_POS + 0] = meas->pos_ned[0];
    x[NMPC_IDX_POS + 1] = meas->pos_ned[1];
    x[NMPC_IDX_POS + 2] = meas->pos_ned[2];
    x[NMPC_IDX_VEL + 0] = meas->vel_ned[0];
    x[NMPC_IDX_VEL + 1] = meas->vel_ned[1];
    x[NMPC_IDX_VEL + 2] = meas->vel_ned[2];
    x[NMPC_IDX_GAMMA + 0] = meas->gamma[0];
    x[NMPC_IDX_GAMMA + 1] = meas->gamma[1];
    x[NMPC_IDX_GAMMA + 2] = meas->gamma[2];
    x[NMPC_IDX_GAMMA + 3] = meas->gamma[3];
}}

static void build_yref(const NmpcSetpoint_t *sp, double *yref)
{{
    memset(yref, 0, NMPC_NY * sizeof(double));
    yref[NMPC_IDX_Q0 + 0] = sp->q_ref[0];
    yref[NMPC_IDX_Q0 + 1] = sp->q_ref[1];
    yref[NMPC_IDX_Q0 + 2] = sp->q_ref[2];
    yref[NMPC_IDX_Q0 + 3] = sp->q_ref[3];
    yref[NMPC_IDX_OMEGA + 0] = sp->omega_ref[0];
    yref[NMPC_IDX_OMEGA + 1] = sp->omega_ref[1];
    yref[NMPC_IDX_OMEGA + 2] = sp->omega_ref[2];
    yref[NMPC_IDX_POS + 0] = sp->pos_ned[0];
    yref[NMPC_IDX_POS + 1] = sp->pos_ned[1];
    yref[NMPC_IDX_POS + 2] = sp->pos_ned[2];
    yref[NMPC_IDX_VEL + 0] = sp->vel_ned[0];
    yref[NMPC_IDX_VEL + 1] = sp->vel_ned[1];
    yref[NMPC_IDX_VEL + 2] = sp->vel_ned[2];
    /* gamma reference: zero  (solver chooses freely)  */
    /* control input reference: zero  (cleared by memset) */
}}

/* ---- public API -------------------------------------------------------- */

int nmpc_init(NmpcCopter_t *ctrl, const double *x0)
{{
    if (!ctrl) return -1;
    memset(ctrl, 0, sizeof(NmpcCopter_t));

    {model_name}_solver_capsule *cap = {model_name}_acados_create_capsule();
    if (!cap) return -2;
    if ({model_name}_acados_create(cap) != 0) {{
        {model_name}_acados_free_capsule(cap);
        return -3;
    }}
    ctrl->capsule     = (void *)cap;
    ctrl->initialized = 1;

    /* pin stage-0 to initial state */
    ocp_nlp_constraints_model_set(cap->nlp_config, cap->nlp_dims,
                                  cap->nlp_in, cap->nlp_out, 0, "lbx", (void *)x0);
    ocp_nlp_constraints_model_set(cap->nlp_config, cap->nlp_dims,
                                  cap->nlp_in, cap->nlp_out, 0, "ubx", (void *)x0);

    /* default yref: level hover at origin */
    ctrl->yref[NMPC_IDX_Q0]   = 1.0;
    ctrl->yref_e[NMPC_IDX_Q0] = 1.0;
    for (int i = 0; i < NMPC_N_HORIZON; i++)
        ocp_nlp_cost_model_set(cap->nlp_config, cap->nlp_dims,
                               cap->nlp_in, i, "yref", ctrl->yref);
    ocp_nlp_cost_model_set(cap->nlp_config, cap->nlp_dims,
                           cap->nlp_in, NMPC_N_HORIZON, "yref", ctrl->yref_e);
    return 0;
}}

int nmpc_step(NmpcCopter_t            *ctrl,
              const NmpcMeasurement_t *meas,
              const NmpcSetpoint_t    *sp,
              double                  *u_out)
{{
    if (!ctrl || !ctrl->initialized) return -1;
    {model_name}_solver_capsule *cap = ({model_name}_solver_capsule *)ctrl->capsule;

    /* 1. pack current state -> stage-0 equality constraint */
    double x[NMPC_NX];
    nmpc_measurement_to_state(meas, x);
    ocp_nlp_constraints_model_set(cap->nlp_config, cap->nlp_dims,
                                  cap->nlp_in, cap->nlp_out, 0, "lbx", x);
    ocp_nlp_constraints_model_set(cap->nlp_config, cap->nlp_dims,
                                  cap->nlp_in, cap->nlp_out, 0, "ubx", x);

    /* 2. reference trajectory */
    build_yref(sp, ctrl->yref);
    memcpy(ctrl->yref_e, ctrl->yref, NMPC_NX * sizeof(double));
    for (int i = 0; i < NMPC_N_HORIZON; i++)
        ocp_nlp_cost_model_set(cap->nlp_config, cap->nlp_dims,
                               cap->nlp_in, i, "yref", ctrl->yref);
    ocp_nlp_cost_model_set(cap->nlp_config, cap->nlp_dims,
                           cap->nlp_in, NMPC_N_HORIZON, "yref", ctrl->yref_e);

    /* 3. SQP-RTI step */
    int status = {model_name}_acados_solve(cap);

    /* 4. extract u  (solver output is u^2 -> sqrt, mirrors main.py) */
    double u_sq[NMPC_NU];
    ocp_nlp_out_get(cap->nlp_config, cap->nlp_dims,
                    cap->nlp_out, 0, "u", u_sq);
    for (int i = 0; i < NMPC_NU; i++)
        u_out[i] = (u_sq[i] > 0.0) ? sqrt(u_sq[i]) : 0.0;

    return status;   /* 0=optimal  1=max-iter  <0=failure */
}}

void nmpc_free(NmpcCopter_t *ctrl)
{{
    if (!ctrl || !ctrl->initialized) return;
    {model_name}_solver_capsule *cap = ({model_name}_solver_capsule *)ctrl->capsule;
    {model_name}_acados_free(cap);
    {model_name}_acados_free_capsule(cap);
    ctrl->capsule     = (void *)0;
    ctrl->initialized = 0;
}}
"""

# --------------------------------------------------------------------------

CONFIG_HEADER_TEMPLATE = """\
/**
 * nmpc_copter_config.h  -  Vehicle constants for {vehicle_name}
 *
 * AUTO-GENERATED by generate_embedded.py  -  DO NOT EDIT
 */

#pragma once

#define NMPC_VEHICLE_NAME       "{vehicle_name}"
#define NMPC_VEHICLE_MOTORCOUNT {motorcount}
#define NMPC_MASS_KG            {mass_kg}
#define NMPC_GRAVITY            {gravity}
#define NMPC_MOTOR_T            {motor_T}
#define NMPC_MOTOR_MAX_OMEGA    {motor_maxOmega}
#define NMPC_UMIN               {umin}
#define NMPC_UMAX               {umax}
#define NMPC_MAX_ROT_RATE_RPS   {max_rotation_rate_rps}
#define NMPC_MAX_HORIZ_VEL_MPS  {max_horizontal_velocity_mps}
#define NMPC_MAX_VERT_VEL_MPS   {max_vertical_velocity_mps}
#define NMPC_J11  {J11}
#define NMPC_J22  {J22}
#define NMPC_J33  {J33}

/* Motor matrix  gamma = M * u_sq,  gamma=[tx ty tz Fz]  [Nm Nm Nm N] */
#define NMPC_MOTOR_MATRIX_ROWS  4
#define NMPC_MOTOR_MATRIX_COLS  {motorcount}
/* clang-format off */
static const double NMPC_MOTOR_MATRIX[4][{motorcount}] = {{
{M_rows}
}};
/* clang-format on */

static const double NMPC_MOTOR_POSITIONS[{motorcount}][2] = {{
{motortable_rows}
}};

static const double NMPC_MOTOR_DIRECTION[{motorcount}] = {{
{motordirection_row}
}};
"""

# --------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Toolchain file template  (only written for embedded targets)
# ---------------------------------------------------------------------------
TOOLCHAIN_TEMPLATE = """# toolchain_{target_id}.cmake
# AUTO-GENERATED by generate_embedded.py  -  DO NOT EDIT
#
# Prevents cmake from link-testing the cross-compiler (always fails on
# bare-metal: no OS, no startup code, no C runtime entry point).
#
# KEY: CMAKE_TRY_COMPILE_TARGET_TYPE must be set BEFORE the compiler is
# tested. Setting it here in the toolchain file (not via -D on the command
# line) guarantees cmake sees it first. The C_FLAGS are also set here so
# the compiler test object is built with the correct target flags.

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# STATIC_LIBRARY: compile test code to .o but do NOT link — linking always
# fails on bare-metal (no OS, no startup code, no libc entry point).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER   "{cc}")
set(CMAKE_AR           "{ar}" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB       "" CACHE FILEPATH "Ranlib")

# CPU/FPU flags set here so the compiler test uses the correct target.
# These are passed to every cmake-driven compilation (acados + BLASFEO).
set(CMAKE_C_FLAGS_INIT "{cmake_c_flags}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
"""

MAKEFILE_TEMPLATE = """\
# Makefile  -  {vehicle_name} NMPC  /  target: {target_id}
#
# AUTO-GENERATED by generate_embedded.py  -  DO NOT EDIT
#
# Builds two things:
#   1. acados static libs  (libacados.a  libblasfeo.a  libhpipm.a)
#   2. libnmpc.a           (nmpc_copter.c + all c_generated_code sources)
#
# After 'make' host project only needs:
#   Headers : nmpc_copter.h   (single public API header)
#   Libs    : -lnmpc  -lacados -lhpipm -lblasfeo -lm
#   Libdir  : -L$(NMPC_DIR)/libs
#
# Usage:
#   make              build everything  ->  libs/
#   make libnmpc      build libnmpc.a only  (assumes acados libs already built)
#   make clean        remove all build artefacts
#   make info         show target configuration
#
# Prerequisite: cmake and '{cc}' on PATH.

ACADOS_ROOT := {acados_root_rel}

CC          := {cc}
AR          := {ar}
CFLAGS      := {c_flags} {c_defs}
# libs/include: copied from acados at build time (libs/include/acados/utils/types.h etc.)
# $(ACADOS_ROOT)/include: direct source-tree fallback, always valid before 'make' is run
# CFLAGS      += -DACADOS_WITH_QPOASES
CFLAGS      += -I. -Ic_generated_code
CFLAGS      += -Ilibs/include
CFLAGS      += -Ilibs/include/blasfeo/include
CFLAGS      += -Ilibs/include/hpipm/include
CFLAGS      += -I$(ACADOS_ROOT)/include
CFLAGS      += -I$(ACADOS_ROOT)/include/blasfeo/include
CFLAGS      += -I$(ACADOS_ROOT)/include/hpipm/include
BUILD_DIR   := .build_acados_{target_id}
LIBS_DIR    := libs
OBJ_DIR     := .obj_nmpc_{target_id}

# --- Source files for libnmpc.a ------------------------------------------
NMPC_SRCS := nmpc_copter.c
NMPC_SRCS += c_generated_code/acados_solver_{model_name}.c
NMPC_SRCS += $(wildcard c_generated_code/{model_name}_model/*.c)

NMPC_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(NMPC_SRCS))

# -------------------------------------------------------------------------

.PHONY: all libnmpc test clean info

all: $(LIBS_DIR)/libacados.a $(LIBS_DIR)/libnmpc.a
\t@echo ""
\t@echo "=== Build complete ==="
\t@echo "Deliverables in $(LIBS_DIR)/:"
\t@echo "  libnmpc.a     <- NMPC controller"
\t@echo "  libacados.a   +"
\t@echo "  libblasfeo.a  | acados runtime"
\t@echo "  libhpipm.a    +"
\t@echo ""
\t@echo "Add to host project:"
\t@echo "  C_INCLUDES += -I$(realpath .)                  (for nmpc_copter.h)"
\t@echo "  LIBS       += -lnmpc -lacados -lhpipm -lblasfeo -lm"
\t@echo "  LIBDIR     += -L$(realpath $(LIBS_DIR))"

# --- libnmpc.a -----------------------------------------------------------

libnmpc: $(LIBS_DIR)/libnmpc.a

$(LIBS_DIR)/libnmpc.a: $(NMPC_OBJS) $(LIBS_DIR)/libacados.a
\t@mkdir -p $(LIBS_DIR)
\t$(AR) rcs $@ $(NMPC_OBJS)
\t@echo "Built $@"

$(OBJ_DIR)/%.o: %.c
\t@mkdir -p $(dir $@)
\t$(CC) $(CFLAGS) -c $< -o $@

# --- acados static libs (via cmake) --------------------------------------

$(LIBS_DIR)/libacados.a: $(BUILD_DIR)/Makefile
\tCFLAGS="$(CMAKE_C_FLAGS_TARGET)" CXXFLAGS="$(CMAKE_C_FLAGS_TARGET)" \\
\t$(MAKE) -C $(BUILD_DIR) -j$$(nproc 2>/dev/null || echo 4) blasfeo hpipm acados
\t@mkdir -p $(LIBS_DIR)
\t@# acados cmake places libs in subdirs of the build tree; use find to be robust
\t@find $(BUILD_DIR) -name "libacados.a"  -exec cp {{}} $(LIBS_DIR)/ \;
\t@find $(BUILD_DIR) -name "libblasfeo.a" -exec cp {{}} $(LIBS_DIR)/ \;
\t@find $(BUILD_DIR) -name "libhpipm.a"   -exec cp {{}} $(LIBS_DIR)/ \;
# \t@find $(BUILD_DIR) -name "libqpOASES_e.a"   -exec cp {{}} $(LIBS_DIR)/ \;
\tcp -r $(ACADOS_ROOT)/include $(LIBS_DIR)/
\t@test -f $(LIBS_DIR)/libacados.a  || (echo "ERROR: libacados.a not found"; exit 1)
\t@test -f $(LIBS_DIR)/libblasfeo.a || (echo "ERROR: libblasfeo.a not found"; exit 1)
\t@test -f $(LIBS_DIR)/libhpipm.a   || (echo "ERROR: libhpipm.a not found"; exit 1)

# CFLAGS/CXXFLAGS are exported inline so that BLASFEO's internal
# build system picks them up — cmake -DCMAKE_C_FLAGS alone is
# not sufficient because BLASFEO GENERIC runs its own sub-make.
CMAKE_C_FLAGS_TARGET := {cmake_c_flags}

$(BUILD_DIR)/Makefile: {toolchain_file}
\t@echo "Configuring acados for {target_id} ..."
\t@mkdir -p $(BUILD_DIR)
\tCFLAGS="$(CMAKE_C_FLAGS_TARGET)" \\
\tCXXFLAGS="$(CMAKE_C_FLAGS_TARGET)" \\
\tcmake -S $(ACADOS_ROOT) -B $(BUILD_DIR) \\
\t    {toolchain_flag}\\
\t    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \\
\t    -DCMAKE_C_FLAGS="$(CMAKE_C_FLAGS_TARGET)" \\
\t    -DBLASFEO_TARGET={blasfeo_target} \\
\t    -DHPIPM_TARGET={hpipm_target} \\
\t    -DACADOS_WITH_QPOASES=OFF \\
\t    -DACADOS_WITH_OPENMP=OFF \\
\t    -DACADOS_EXAMPLES=OFF \\
\t    -DBUILD_SHARED_LIBS=OFF \\
\t    -DACADOS_SILENT=ON \\
\t    -DCMAKE_BUILD_TYPE=Release

# --- x86 host test -------------------------------------------------------
#
# Compiles and runs nmpc_test_x86.c against the locally-built acados libs.
# Only meaningful when targeting 'linux' or building on the same x86 host.
# Only available for vehicle {motorcount} motors ({vehicle_name}).
#   make test
#
TEST_BIN := nmpc_test_x86
NMPC_TEST_VEHICLE_MOTORCOUNT := {motorcount}

test: $(LIBS_DIR)/libnmpc.a
ifeq ($(NMPC_TEST_VEHICLE_MOTORCOUNT),18)
\tgcc -O3 -g -o $(TEST_BIN) nmpc_test_x86.c $(CFLAGS) \\
# \t    -Llibs -lnmpc -lacados -lqpOASES_e -lhpipm -lblasfeo -lm
\t    -Llibs -lnmpc -lacados -lhpipm -lblasfeo -lm
\t@echo "[BUILD] $(TEST_BIN) OK"
\t./$(TEST_BIN)
else
\t@echo "[SKIP]  make test is only available for the 18-motor ({vehicle_name}) build."
\t@echo "        This build has $(NMPC_TEST_VEHICLE_MOTORCOUNT) motors - no reference"
\t@echo "        test vector is defined for this vehicle."
\t@exit 1
endif

# -------------------------------------------------------------------------

clean:
\trm -rf $(BUILD_DIR) $(LIBS_DIR) $(OBJ_DIR) $(TEST_BIN)

info:
\t@echo "Target      : {target_id}"
\t@echo "Description : {target_description}"
\t@echo "Compiler    : {cc}"
\t@echo "BLASFEO     : {blasfeo_target}"
\t@echo "HPIPM       : {hpipm_target}"
\t@echo "C flags     : {c_flags}"
\t@echo "C defines   : {c_defs}"
\t@echo "acados root : $(realpath $(ACADOS_ROOT))"
\t@echo "NMPC srcs   : $(NMPC_SRCS)"
{notes_echo}
"""

# --------------------------------------------------------------------------

EXAMPLE_TEMPLATE = """\
/**
 * nmpc_example_usage.c  -  Minimal integration example
 *
 * AUTO-GENERATED by generate_embedded.py  -  reference only, not compiled automatically.
 *
 * Vehicle : {vehicle_name}  /  Target : {target_id}
 *
 * The init state and first setpoint below match the regression test vector
 * (see nmpc_test_x86.c) so this file doubles as a readable worked example.
 */

#include "nmpc_copter.h"
#include <string.h>

extern void sensors_get_measurement(NmpcMeasurement_t *meas);
extern void motors_set_throttle(const double *u, int n);

static NmpcCopter_t g_nmpc;

void flight_controller_init(void)
{{
    /* Test-vector initial state: level hover at D = -0.7 m (0.7 m AGL),
     * zero velocity, hover thrust in gamma[3].
     * Replace with actual sensor fusion output on the real vehicle. */
    NmpcMeasurement_t init_meas = {{0}};
    init_meas.q[0]      = 1.0;       /* level quaternion: q0=1, q1..3=0 */
    init_meas.pos_ned[2] = -0.7;     /* 0.7 m above ground (NED Down)   */
    init_meas.gamma[3]   = 9563.736; /* hover thrust [N]                 */

    double x0[NMPC_NX];
    nmpc_measurement_to_state(&init_meas, x0);

    if (nmpc_init(&g_nmpc, x0) != 0)
        while (1);            /* handle init failure */
}}

/* Call from a {mpc_hz} Hz timer task */
void flight_controller_step(void)
{{
    NmpcMeasurement_t meas = {{0}};
    sensors_get_measurement(&meas);

    /* Setpoint matches test vector: hold position at 0.7 m AGL, level */
    NmpcSetpoint_t sp = {{0}};
    sp.pos_ned[0] =  0.0;   /* North [m]                               */
    sp.pos_ned[1] =  0.0;   /* East  [m]                               */
    sp.pos_ned[2] = -0.7;   /* Down  [m]  (negative = above ground)    */
    sp.q_ref[0]   =  1.0;   /* level quaternion                        */
    /* vel_ned and omega_ref stay zero -> position hold */

    double u[NMPC_MOTORCOUNT];
    int status = nmpc_step(&g_nmpc, &meas, &sp, u);

    if (status >= 0)         /* 0=optimal  1=max-iter, output still valid */
        motors_set_throttle(u, NMPC_MOTORCOUNT);
    /* else: solver failed -> apply safe fallback */
}}
"""

# --------------------------------------------------------------------------

TEST_TEMPLATE = """\
/**
 * nmpc_test_x86.c  -  Host-PC regression test for the NMPC solver
 *
 * AUTO-GENERATED by generate_embedded.py  -  compile and run on x86 Linux
 *
 * Vehicle : {vehicle_name}  /  Target: linux (x86-64)
 *
 * Build (inside the output directory after 'make' has produced the libs):
 *   gcc -O3 -g -o nmpc_test_x86 nmpc_test_x86.c \\
 *       -I. -Ic_generated_code -Ilibs/include \\
 *       -Ilibs/include/blasfeo/include -Ilibs/include/hpipm/include \\
 *       -Llibs -lnmpc -lacados -lhpipm -lblasfeo -lm
 *
 * Or use the generated Makefile:
 *   make test
 *
 * Expected output (vehicle={vehicle_name}, {motorcount} motors):
 *   [TEST] nmpc_init         ... OK
 *   [TEST] nmpc_step status  ... OK (0 or 1)
 *   [TEST] u range           ... OK  (all in [0..1])
 *   [TEST] test vector       ... OK  (u_sq within tolerance of expected)
 *   [PASS] All tests passed.
 */

#include "nmpc_copter.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- test vector (generated from Python reference run) ----------------- */
/*
 * state = [q0  q1  q2  q3  wx  wy  wz  N   E   D   vN  vE  vD  gx  gy  gz  Fz]
 *
 *   state[{nx}] =
 *     {{ 1.0, 0.0, -0.0, -0.0,          // quaternion (level)
 *        -0.0, 0.0, 0.0,                // omega_b  [rad/s]
 *        0.0, 0.0, -0.7,                // pos NED  [m]   (D=-0.7  -> 0.7 m AGL)
 *        -0.0, -0.0, -0.0,              // vel NED  [m/s]
 *        0.0, -0.0, -0.0, 9563.736 }};  // gamma    [Nm Nm Nm N]
 *
 * Expected u_squared output (before sqrt):
 *   u_sq[i] ~ 0.671..0.672  for all {motorcount} motors  (hover trim)
 *
 * Tolerance: each u (throttle) must be within THROTTLE_TOL of the expected value.
 */

/* Test-vector state */
static const double TV_STATE[{nx}] = {{
    /* q0..q3 */      1.0, 0.0, 0.0, 0.0,
    /* omega_b */     0.0, 0.0, 0.0,
    /* pos NED  */    0.0, 0.0, -0.7,
    /* vel NED  */    0.0, 0.0, 0.0,
    /* gamma    */    0.0, 0.0, 0.0, 9563.736
}};

/* Expected throttle output (sqrt of u_sq), one value per motor */
#define TV_EXPECTED_THROTTLE  0.8193   /* sqrt(0.6713) ~ 0.819 */
#define THROTTLE_TOL          0.05     /* ±5 % tolerance       */

/* ---- helpers ----------------------------------------------------------- */

static int check_range(const double *u, int n, double lo, double hi,
                       const char *label)
{{
    for (int i = 0; i < n; i++) {{
        if (u[i] < lo || u[i] > hi) {{
            printf("[FAIL] %s: u[%d] = %.6f  out of [%.3f .. %.3f]\\n",
                   label, i, u[i], lo, hi);
            return 0;
        }}
    }}
    printf("[TEST] %-28s ... OK\\n", label);
    return 1;
}}

static int check_near(const double *u, int n,
                      double expected, double tol, const char *label)
{{
    int ok = 1;
    for (int i = 0; i < n; i++) {{
        if (fabs(u[i] - expected) > tol) {{
            printf("[FAIL] %s: u[%d] = %.6f  expected %.6f ± %.4f\\n",
                   label, i, u[i], expected, tol);
            ok = 0;
        }}
    }}
    if (ok) printf("[TEST] %-28s ... OK\\n", label);
    return ok;
}}

/* ---- main -------------------------------------------------------------- */

int main(void)
{{
    printf("\\n=== NMPC host-PC regression test ===\\n");
    printf("Vehicle  : {vehicle_name}\\n");
    printf("Motors   : {motorcount}\\n");
    printf("NX / NU  : {nx} / {nu}\\n\\n");

    int failures = 0;

    /* ---- 1. init -------------------------------------------------------- */
    NmpcCopter_t ctrl;
    memset(&ctrl, 0, sizeof(ctrl));

    double x0[NMPC_NX];
    memcpy(x0, TV_STATE, sizeof(x0));

    int rc = nmpc_init(&ctrl, x0);
    if (rc != 0) {{
        printf("[FAIL] nmpc_init returned %d\\n", rc);
        return 1;
    }}
    printf("[TEST] %-28s ... OK\\n", "nmpc_init");

    /* ---- 2. step -------------------------------------------------------- */
    NmpcMeasurement_t meas = {{0}};
    /* fill meas from test vector */
    meas.q[0] = TV_STATE[NMPC_IDX_Q0 + 0];
    meas.q[1] = TV_STATE[NMPC_IDX_Q0 + 1];
    meas.q[2] = TV_STATE[NMPC_IDX_Q0 + 2];
    meas.q[3] = TV_STATE[NMPC_IDX_Q0 + 3];
    meas.omega_b[0] = TV_STATE[NMPC_IDX_OMEGA + 0];
    meas.omega_b[1] = TV_STATE[NMPC_IDX_OMEGA + 1];
    meas.omega_b[2] = TV_STATE[NMPC_IDX_OMEGA + 2];
    meas.pos_ned[0] = TV_STATE[NMPC_IDX_POS + 0];
    meas.pos_ned[1] = TV_STATE[NMPC_IDX_POS + 1];
    meas.pos_ned[2] = TV_STATE[NMPC_IDX_POS + 2];
    meas.vel_ned[0] = TV_STATE[NMPC_IDX_VEL + 0];
    meas.vel_ned[1] = TV_STATE[NMPC_IDX_VEL + 1];
    meas.vel_ned[2] = TV_STATE[NMPC_IDX_VEL + 2];
    meas.gamma[0]   = TV_STATE[NMPC_IDX_GAMMA + 0];
    meas.gamma[1]   = TV_STATE[NMPC_IDX_GAMMA + 1];
    meas.gamma[2]   = TV_STATE[NMPC_IDX_GAMMA + 2];
    meas.gamma[3]   = TV_STATE[NMPC_IDX_GAMMA + 3];

    /* hover setpoint: hold current altitude, level */
    NmpcSetpoint_t sp = {{0}};
    sp.pos_ned[2] = -0.7;   /* match test-vector altitude */
    sp.q_ref[0]   =  1.0;   /* level quaternion */

    double u_out[NMPC_MOTORCOUNT];
    int status = nmpc_step(&ctrl, &meas, &sp, u_out);

    if (status < 0) {{
        printf("[FAIL] nmpc_step returned status %d\\n", status);
        failures++;
    }} else {{
        printf("[TEST] %-28s ... OK (status=%d)\\n", "nmpc_step status", status);
    }}

    /* ---- 3. print raw output ------------------------------------------- */
    printf("\\nMotor throttle output (u = sqrt(u_sq)):\\n");
    for (int i = 0; i < NMPC_MOTORCOUNT; i++)
        printf("  u[%2d] = %.6f\\n", i, u_out[i]);
    printf("\\n");

    /* ---- 4. range check [0..1] ----------------------------------------- */
    if (!check_range(u_out, NMPC_MOTORCOUNT, 0.0, 1.0, "u range [0..1]"))
        failures++;

    /* ---- 5. proximity check against expected --------------------------- */
    if (!check_near(u_out, NMPC_MOTORCOUNT,
                    TV_EXPECTED_THROTTLE, THROTTLE_TOL, "test vector match"))
        failures++;

    /* ---- 6. cleanup ----------------------------------------------------- */
    nmpc_free(&ctrl);

    /* ---- result --------------------------------------------------------- */
    printf("\\n");
    if (failures == 0)
        printf("[PASS] All tests passed.\\n\\n");
    else
        printf("[FAIL] %d test(s) FAILED.\\n\\n", failures);

    return (failures == 0) ? 0 : 1;
}}
"""

# --------------------------------------------------------------------------

README_TEMPLATE = """\
# Embedded NMPC  -  {vehicle_name}  /  {target_id}

Auto-generated by `generate_embedded.py`.

## Directory layout

```
./
+-- Makefile                      builds acados static libs for {target_id}
+-- nmpc_copter.h                 API  (include this)
+-- nmpc_copter.c                 wrapper implementation
+-- nmpc_copter_config.h          vehicle constants (#defines + static arrays)
+-- nmpc_example_usage.c          integration example  (reference only)
+-- acados_ocp_{model_name}.json  OCP definition
+-- c_generated_code/             acados solver sources  (platform-independent)
|   +-- acados_solver_{model_name}.c
|   +-- acados_solver_{model_name}.h
|   +-- {model_name}_model/
|       +-- {model_name}_expl_ode_fun.c
|       +-- ...
+-- libs/                         created by 'make'  (add to .gitignore)
    +-- libacados.a
    +-- libblasfeo.a
    +-- libhpipm.a
    +-- include/
```

## Step 1 - build the static libs

```bash
cd {outdir_basename}
make           # cross-compiles acados for {target_id} using {cc}
make info      # show full target configuration
make clean     # remove build artefacts
```

Requires `cmake` and `{cc}` on PATH.
Output: `libs/libnmpc.a`, `libs/libacados.a`, `libs/libblasfeo.a`, `libs/libhpipm.a`.

## Step 1b - host-PC regression test  (linux target only)

After `make` completes, verify the solver against the reference test vector:

```bash
make test
# Expected output:
# [TEST] nmpc_init             ... OK
# [TEST] nmpc_step status      ... OK (status=0 or 1)
# [TEST] u range [0..1]        ... OK
# [TEST] test vector match     ... OK
# [PASS] All tests passed.
```

The test vector corresponds to a near-hover state (level attitude, altitude = 0.7 m AGL,
gamma[3] ≈ 9563 N·... / hover trim).  All motor throttle outputs are expected near **0.819**
(≈ `sqrt(0.671)`).  A tolerance of ±0.05 is applied.

## Step 2 - add to STM32CubeIDE / Makefile project

After `make` completes it prints the exact snippet to paste.

You only need **one header** and **four static libs** — no C sources to add to the project:

```makefile
NMPC_DIR := /path/to/{outdir_basename}

# Single include path (gives access to nmpc_copter.h)
C_INCLUDES += -I$(NMPC_DIR)

# Four libs - no C sources needed in project
LIBS   += -lnmpc -lacados -lhpipm -lblasfeo -lm
LIBDIR += -L$(NMPC_DIR)/libs
```

In STM32CubeIDE: *Properties > C/C++ Build > Settings > MCU GCC Linker*
- Library search path: `../path/to/{outdir_basename}/libs`
- Libraries: `nmpc`  `acados`  `blasfeo`  `hpipm`  `m`
- Include path (Compiler): `../path/to/{outdir_basename}`

## Step 3 - integrate in application

```c
#include "nmpc_copter.h"

NmpcCopter_t g_ctrl;

void app_init(void)
{{
    double x0[NMPC_NX] = {{0}};
    x0[NMPC_IDX_Q0] = 1.0;
    nmpc_init(&g_ctrl, x0);
}}

/* call at {mpc_hz} Hz */
void control_task(void)
{{
    NmpcMeasurement_t meas = {{ /* fill from AHRS/INS */ }};
    NmpcSetpoint_t    sp   = {{
        .pos_ned = {{0, 0, -5.0}},
        .q_ref   = {{1, 0, 0, 0}},
    }};
    double u[NMPC_MOTORCOUNT];
    int status = nmpc_step(&g_ctrl, &meas, &sp, u);
    /* u[i] in [0..1] -> ESC i */
}}
```

## Target: {target_id}

| Parameter       | Value                        |
|-----------------|------------------------------|
| Description     | {target_description}         |
| Compiler        | `{cc}`                       |
| BLASFEO target  | `{blasfeo_target}`           |
| HPIPM target    | `{hpipm_target}`             |
| C flags         | `{c_flags}`                  |
| C defines       | `{c_defs}`                   |

## Vehicle: {vehicle_name}

| Parameter          | Value                        |
|--------------------|------------------------------|
| Motors (NU)        | {motorcount}                 |
| State vector (NX)  | {nx}                         |
| Horizon stages     | {N_horizon}                  |
| Horizon time       | {Tf} s                       |
| Update rate        | {mpc_hz} Hz                  |
| Mass               | {mass_kg} kg                 |
| Motor max omega    | {motor_maxOmega} rad/s       |
| u range            | [{umin} .. {umax}]           |
| Max rotation rate  | {max_rot_rate_deg:.1f} deg/s |

{notes_md}
"""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(vehicle_id: int, target_id: str, outdir: str):
    if not os.path.isabs(outdir):
        outdir = os.path.join(PROJECT_ROOT, outdir)
    outdir = os.path.normpath(outdir)

    target = TARGETS[target_id]

    print("=" * 62)
    print("  acados Embedded Code Generator")
    print("  Vehicle : %i" % vehicle_id)
    print("  Target  : %s  (%s)" % (target_id, target["description"]))
    print("  Output  : %s" % outdir)
    print("=" * 62)

    import vehicleconfig as vc_module
    vehicle_config = vc_module.CopterConfig(vehicle=vehicle_id)
    print("Vehicle : %s  (%i motors,  %.1f kg)" % (
          vehicle_config.vehicle_name, vehicle_config.motorcount,
          vehicle_config.mass_kg))

    # --- Build & generate OCP --------------------------------------------
    ocp, model, nx, nu, ny, N_horizon, Tf = build_ocp(vehicle_config)
    mname  = model.name
    mpc_hz = vehicle_config.ocp_embedded.mpc_hz
    cfg    = vehicle_config.state_cfg

    os.makedirs(outdir, exist_ok=True)
    solver_json               = os.path.join(outdir, "acados_ocp_%s.json" % mname)
    ocp.code_export_directory = os.path.join(outdir, "c_generated_code")

    print("\nGenerating acados C code ...")
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    print("acados code generation done.")

    # Post-process: remove "#include <omp.h>" from generated C files.
    # The acados host library may have been built with OpenMP; the JSON
    # carries this flag into the generated code. On embedded targets
    # (HPIPM EMBEDDED + BLASFEO GENERIC) OpenMP is never used, so the
    # include is dead code — but arm-none-eabi-gcc has no omp.h.
    import glob
    gen_dir = ocp.code_export_directory
    c_files = glob.glob(os.path.join(gen_dir, "**", "*.c"), recursive=True)
    c_files += glob.glob(os.path.join(gen_dir, "**", "*.h"), recursive=True)
    omp_removed = 0
    for fpath in c_files:
        with open(fpath, "r", errors="replace") as cf:
            src = cf.read()
        if "#include <omp.h>" in src or "#include \"omp.h\"" in src:
            src = src.replace("#include <omp.h>", "/* #include <omp.h> (removed for embedded) */")
            src = src.replace('#include "omp.h"', '/* #include "omp.h" (removed for embedded) */')
            with open(fpath, "w") as cf:
                cf.write(src)
            omp_removed += 1
    if omp_removed:
        print("  removed omp.h include from %d generated file(s)" % omp_removed)

    # --- Format dict -----------------------------------------------------
    outdir_basename = os.path.basename(outdir)
    acados_root_rel = os.path.relpath(ACADOS_ROOT, outdir)

    notes_echo = ('\t@echo "Note        : %s"' % target["notes"]) \
                 if target.get("notes") else ""
    notes_md   = ("> **Note:** %s" % target["notes"]) \
                 if target.get("notes") else ""

    # Toolchain file: only needed for embedded (bare-metal) targets.
    # hpipm_target == EMBEDDED is the reliable signal for bare-metal.
    is_embedded = (target["hpipm_target"] == "EMBEDDED")
    if is_embedded:
        toolchain_filename = "toolchain_%s.cmake" % target_id
        toolchain_filepath = os.path.join(outdir, toolchain_filename)
        toolchain_flag_str = ("-DCMAKE_TOOLCHAIN_FILE=$(realpath %s) "
                              % toolchain_filename)
    else:
        toolchain_filename = ""
        toolchain_filepath = None
        # For PC/Linux targets: only pass the C compiler; let cmake find AR
        # via its own search path. Passing AR="ar" (relative) causes cmake to
        # resolve it relative to the build directory -> "No such file or directory".
        toolchain_flag_str = ("-DCMAKE_C_COMPILER=\"%s\" " % target["cc"])


    fmt = dict(
        # target
        target_id          = target_id,
        target_description = target["description"],
        blasfeo_target     = target["blasfeo_target"],
        hpipm_target       = target["hpipm_target"],
        cc                 = target["cc"],
        ar                 = target["ar"],
        c_flags            = target["c_flags"],
        # cmake_c_flags: used for the acados cmake build only.
        # Embedded targets define this separately (no -mcpu) to avoid
        # conflicts with BLASFEO GENERIC's internal -march flag.
        cmake_c_flags      = target.get("cmake_c_flags", target["c_flags"]),
        c_defs             = target["c_defs"],
        notes_echo         = notes_echo,
        notes_md           = notes_md,
        toolchain_file     = toolchain_filename,
        toolchain_flag     = toolchain_flag_str,
        # vehicle
        vehicle_name       = vehicle_config.vehicle_name,
        model_name         = mname,
        motorcount         = vehicle_config.motorcount,
        nx                 = nx,
        nu                 = nu,
        ny                 = ny,
        N_horizon          = N_horizon,
        Tf                 = Tf,
        mpc_hz             = mpc_hz,
        mass_kg            = "%.6g" % vehicle_config.mass_kg,
        gravity            = "%.6g" % vehicle_config.gravity_n[2],
        motor_T            = "%.6g" % vehicle_config.motor_T,
        motor_maxOmega     = "%.6g" % vehicle_config.motor_maxOmega_rad_per_sec,
        umin               = "%.6g" % vehicle_config.umin,
        umax               = "%.6g" % vehicle_config.umax,
        max_rotation_rate_rps       = "%.8f" % vehicle_config.max_rotation_rate_rps,
        max_horizontal_velocity_mps = "%.6g" % vehicle_config.max_horizontal_velocity_mps,
        max_vertical_velocity_mps   = "%.6g" % vehicle_config.max_vertical_velocity_mps,
        J11 = "%.6g" % vehicle_config.J[0, 0],
        J22 = "%.6g" % vehicle_config.J[1, 1],
        J33 = "%.6g" % vehicle_config.J[2, 2],
        M_rows             = matrix_to_c_rows(vehicle_config.M),
        motortable_rows    = matrix_to_c_rows(vehicle_config.motortable),
        motordirection_row = array_to_c_row(vehicle_config.motordirection),
        # indices
        q_index           = cfg["q_index"],
        q_index_end       = cfg["q_index_end"],
        omega_index       = cfg["omega_index"],
        omega_index_end   = cfg["omega_index_end"],
        pos3d_index       = cfg["pos3d_index"],
        pos3d_index_end   = cfg["pos3d_index_end"],
        vel3d_index       = cfg["vel3d_index"],
        vel3d_index_end   = cfg["vel3d_index_end"],
        gamma_index       = cfg["gamma_index"],
        gamma_index_end   = cfg["gamma_index_end"],
        # paths
        outdir_basename   = outdir_basename,
        acados_root_rel   = acados_root_rel,
        max_rot_rate_deg  = np.rad2deg(vehicle_config.max_rotation_rate_rps),
    )

    # --- Write files -----------------------------------------------------
    files = {
        "nmpc_copter.h":        HEADER_TEMPLATE,
        "nmpc_copter.c":        SOURCE_TEMPLATE,
        "nmpc_copter_config.h": CONFIG_HEADER_TEMPLATE,
        "nmpc_example_usage.c": EXAMPLE_TEMPLATE,
        "nmpc_test_x86.c":      TEST_TEMPLATE,
        "Makefile":             MAKEFILE_TEMPLATE,
        "README.md":            README_TEMPLATE,
    }
    if is_embedded:
        files[toolchain_filename] = TOOLCHAIN_TEMPLATE

    for filename, template in files.items():
        path = os.path.join(outdir, filename)
        with open(path, "w") as f:
            f.write(template.format(**fmt))
        print("  wrote  %s" % os.path.relpath(path))

    print("\nDone.")
    print("\nNext steps:")
    print("  cd %s" % os.path.relpath(outdir))
    print("  make           # builds libnmpc.a + libacados.a + libhpipm.a + libblasfeo.a")
    print("  make info      # show full build configuration")
    print("\nThen in host project:")
    print("  C_INCLUDES += -I%s" % os.path.relpath(outdir))
    print("  LIBS       += -lnmpc -lacados -lhpipm -lblasfeo -lm")
    print("  LIBDIR     += -L%s/libs" % os.path.relpath(outdir))
    if target.get("notes"):
        print("\nNote: %s" % target["notes"])


# ---------------------------------------------------------------------------

def list_targets():
    print("\nAvailable targets:\n")
    w = max(len(k) for k in TARGETS)
    for key, t in TARGETS.items():
        print("  %-*s  %s" % (w, key, t["description"]))
        print("  %-*s  compiler : %s" % (w, "", t["cc"]))
        print("  %-*s  BLASFEO  : %s" % (w, "", t["blasfeo_target"]))
        if t.get("notes"):
            print("  %-*s  note     : %s" % (w, "", t["notes"]))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embedded NMPC C code for a multirotor vehicle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples (run from project root with venv active):
              source env.sh
              python src/generate_embedded.py --target stm32n6
              python src/generate_embedded.py --target rpi4  --vehicle 4
              python src/generate_embedded.py --target linux --outdir build/sitl
              python src/generate_embedded.py --list-targets
        """))
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET,
        choices=list(TARGETS.keys()), metavar="TARGET",
        help="Hardware target (default: %s).  Use --list-targets to see all." % DEFAULT_TARGET)
    parser.add_argument(
        "--vehicle", type=int, default=DEFAULT_VEHICLE, choices=[4, 18],
        help="Vehicle: 4=Quadcopter  18=eVTOL  (default: %i)" % DEFAULT_VEHICLE)
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output dir (default: embedded_<vehicle>_<target> in project root)")
    parser.add_argument(
        "--list-targets", action="store_true",
        help="Print all available targets and exit.")
    args = parser.parse_args()

    if args.list_targets:
        list_targets()
        sys.exit(0)

    if args.outdir is None:
        args.outdir = "embedded_%i_%s" % (args.vehicle, args.target)

    generate(args.vehicle, args.target, args.outdir)
