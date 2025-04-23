# nmpc_multirotor
NMPC Multirotor Drone Flight Control

![LOGO](img/logo.png)

# 🛠️ Setup

Prerequisites

 - Linux, macOS or WSL on Windows
 - Python 3.8+
 - CMake
 - A C++ compiler (e.g., gcc, clang)
 - Git

# Clone and Setup

    git clone --recurse-submodules https://github.com/jnz/nmpc_multirotor.git
    cd nmpc_multirotor
    ./setup.sh

This will:

 - Check out the correct acados version
 - Build acados with required options
 - Create a Python virtual environment (venv_nmpc)
 - Install all dependencies

# 🚀 Running

    source env.sh
    ./src/main.py

# 🕹️ Controls

Use these keys to manually influence the drone:

| Key         | Command            | Direction     |
|-------------|--------------------|---------------|
| ← / →       | Lateral movement   | Left / Right  |
| ↓ / ↑       | Longitudinal       | Back / Forward|
| `j` / `k`   | Vertical thrust     | Up / Down     |
| `h` / `l`   | Yaw (rotation)      | Left / Right  |

- Arrow keys: Horizontal translation (XY-plane)
- `j`/`k`: Climb/descend
- `h`/`l`: Rotate yaw left/right

