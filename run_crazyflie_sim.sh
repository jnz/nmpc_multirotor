#!/bin/bash

set -e

# 1. Define default values
CFLIB_PATH="$HOME/developer/CrazySim/crazyflie-lib-python"
PYTHON_SCRIPT="src/crazysim_bridge.py"
PYTHON_ARGS=()

# Function to display help text
show_help() {
    echo "Usage: ./run_crazysim_bridge.sh [OPTIONS] [SCRIPT_ARGS...]"
    echo ""
    echo "A wrapper script to run Crazyflie scripts in either 'sim' or 'real' mode."
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message and exit"
    echo "  --cflib-path PATH    Override the default cflib path"
    echo "                       (Default: $HOME/developer/CrazySim/crazyflie-lib-python)"
    echo "  --script FILE        Specify a different Python script to run"
    echo "                       (Default: crazysim_bridge.py)"
    echo ""
    echo "Environment Variables:"
    echo "  CF_MODE              Set to 'real' to use the official cflib from the venv."
    echo "                       Set to 'sim' (or leave unset) to use the local CrazySim cflib."
    echo ""
    echo "Example:"
    echo "  CF_MODE=real ./run_crazysim_bridge.sh --script custom_test.py --log-level DEBUG"
}

# 2. Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0 # Exit successfully after showing help
            ;;
        --cflib-path)
            CFLIB_PATH="$2"
            shift 2
            ;;
        --script)
            # Override the default Python script
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift 1
            ;;
    esac
done

# 3. Evaluate mode
MODE="${CF_MODE:-sim}"

if [[ "${MODE,,}" == "sim" ]]; then
    echo "[SIM MODE] Using cflib via PYTHONPATH..."
    echo "   Path: $CFLIB_PATH"

    if [ ! -d "$CFLIB_PATH" ]; then
        echo "Error: The cflib directory '$CFLIB_PATH' does not exist!"
        exit 1
    fi

    export PYTHONPATH="$CFLIB_PATH:$PYTHONPATH"
else
    echo "[REAL MODE] Using default cflib..."
fi

echo "Running script: $PYTHON_SCRIPT"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: '$PYTHON_SCRIPT' not found."
    exit 1
fi

# 4. Execute Python script with remaining arguments
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
