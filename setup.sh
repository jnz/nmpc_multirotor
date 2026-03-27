#!/bin/bash
set -e

ACADOS_VERSION=v0.5.3

# setup acados
echo "Setting up acados submodule..."
cd acados
git fetch --tags
git checkout $ACADOS_VERSION
git submodule update --init --recursive
mkdir -p build
cd build
cmake .. -DACADOS_WITH_QPOASES=OFF -DACADOS_WITH_OPENMP=OFF -DACADOS_EXAMPLES=OFF
make install -j4 > /dev/null
cd ../..

echo "Setting up nmpc_multirotor python project"
python3 -m venv venv_nmpc
source venv_nmpc/bin/activate
pip3 install .
pip3 install -e acados/interfaces/acados_template

echo "Setup done

Run 

> (source env.sh; ./src/main.py)

to start"

