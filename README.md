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

For Ubuntu/Debian Linux and WSL with Ubuntu on Windows, you can install the
required packages using:

    sudo apt update && sudo apt install -y \
        python3 python3-pip python3-venv \
        cmake \
        build-essential \
        git

(Optional) For the 3D visualization via Pygame + OpenGL (PyOpenGL requires
`libglu1-mesa` at runtime):

    sudo apt update && sudo apt install -y \
        libsdl2-2.0-0 \
        libgl1 \
        libglu1-mesa

🔧 Note for WSL Users (Windows Subsystem for Linux)

If you're running this project under WSL2 on Windows:

 - The 3D visualization (Pygame + OpenGL) may show a black screen even though no errors occur.
 - This is a known issue with WSLg + Mesa + D3D12 and some GPUs (e.g., Intel Arc).
 - Rendering does happen internally — it's just not visible due to a framebuffer issue.

Workaround: Force software rendering, by running the program like this:

    LIBGL_ALWAYS_SOFTWARE=true python src/main.py

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

