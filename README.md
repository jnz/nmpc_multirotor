# nmpc_multirotor - NMPC Multirotor Drone Flight Control

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTCST.2026.3672184-blue)](https://doi.org/10.1109/TCST.2026.3672184)


![NMPC Multirotor Simulation GIF](img/nmpc_multirotor.gif)
Real-time visualization GIF (aircraft is white, NMPC prediction drawn in blue).

## About this Project

This repository contains the implementation of a
real-time capable Nonlinear Model Predictive Control (NMPC) for multirotor
aircrafts (multicopter drones).

* **~70% faster solve times** for highly redundant setups (e.g., 32 rotors).
* **Direct handling of actuator limits** within the NMPC, completely eliminating the need for a separate control allocation step.
* **Built-in compensation** for the delayed thrust response of (slow) motors.

**Hybrid Architecture for Simulation & Embedded Deployment:**
To make this project both easy to use and hardware-ready, we split the architecture:
1. **The Simulation (Python):** The 6-DoF rigid-body physics simulation (incl. obstacle simulation) and the OpenGL 3D visualization are written in Python.
2. **The Controller (C via acados):** The actual NMPC solver is powered by the [`acados`](https://github.com/acados/acados) library. `acados` generates highly optimized, real-time capable **C code**.

The generated C code has been tested to run in real-time on embedded platforms like the **NVIDIA Jetson TX2** (using the Arm Cortex-A57 CPU, there are no GPU dependencies).

# Installation

## Prerequisites

 - Linux, macOS or WSL on Windows
 - Python 3.8+
 - CMake
 - A C/C++ compiler (e.g., gcc, clang)
 - Git

For Ubuntu/Debian Linux and WSL with Ubuntu on Windows, install the
required packages:

    sudo apt update && sudo apt install -y \
        python3 python3-pip python3-venv \
        cmake build-essential git

(Optional) For the 3D visualization via Pygame + OpenGL (PyOpenGL requires
`libglu1-mesa` at runtime):

    sudo apt update
    sudo apt install libsdl2-2.0-0 libgl1 libglu1-mesa

## Clone and Setup

    git clone --recurse-submodules https://github.com/jnz/nmpc_multirotor.git
    cd nmpc_multirotor
    ./setup.sh

 - Check out the correct acados version as git submodule
 - Build acados with required options
 - Create a Python virtual environment (`venv_nmpc`)
 - Install the Python dependencies

## Running

    source env.sh   # for the virtual env.
    ./src/main.py

Note: On the first run acados will ask you to download the Tera renderer, press `y`:

    Do you wish to set up Tera renderer automatically?
    y/N? (press y to download tera or any key for manual installation)

# Simulation Controls

Use these keys to manually influence the drone:

| Key         | Command            | Direction     |
|-------------|--------------------|---------------|
| ← / →       | Lateral movement   | Left / Right  |
| ↓ / ↑       | Longitudinal       | Back / Forward|
| `j` / `k`   | Vertical thrust     | Up / Down     |
| `h` / `l`   | Yaw (rotation)      | Left / Right  |

# Note for WSL Users (Windows Subsystem for Linux)

This project runs fine on Windows with WSL2 and Ubuntu.

Windows WSL (WSLg + Mesa + D3D12) can have issues with hardware rendering on some GPUs (e.g. Intel
Arc). Workaround: Force software rendering:

    LIBGL_ALWAYS_SOFTWARE=true python src/main.py

# Citation

If you find this code useful in your research, please consider citing our paper:

> J. Zwiener, J. Stephan and C. Seiferth, "Real-Time Nonlinear Model Predictive Control of Large Multirotor Aircraft," in *IEEE Transactions on Control Systems Technology*, doi: [10.1109/TCST.2026.3672184](https://doi.org/10.1109/TCST.2026.3672184).

## BibTeX

```bibtex
@article{zwiener2026nmpc,
  author={Zwiener, Jan and Stephan, Johannes and Seiferth, Christoph},
  journal={IEEE Transactions on Control Systems Technology},
  title={Real-Time Nonlinear Model Predictive Control of Large Multirotor Aircraft},
  year={2026},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/TCST.2026.3672184}}
```

