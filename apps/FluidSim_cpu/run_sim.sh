#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALBERLA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$WALBERLA_ROOT/build-cpu"
EXE_WALBERLA_LAYOUT="$BUILD_DIR/walberla/apps/FluidSim_cpu/FluidSim_cpu"
EXE_APPS_LAYOUT="$BUILD_DIR/apps/FluidSim_cpu/FluidSim_cpu"
EXE=""
VENV="${VENV:-$PROJECT_ROOT/venv-walberla-codegen}"

NP="${NP:-1}"
PARAMS="${PARAMS:-$SCRIPT_DIR/../shared/params/FluidSim.prm}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
OMP_PLACES="${OMP_PLACES:-cores}"
BUILD_JOBS="${BUILD_JOBS:-1}"
LOG_DIR="$SCRIPT_DIR/output/logs"
LOG_FILE="$LOG_DIR/run_sim.log"

# Default CLI flags for FluidSim.
FLAGS=(
    --parallelMode inner
    --minimalLogs 1000
    --thermalLogs 100
    --initPerturb 0
    --vtkinit 0
    --timesteps 101
    --checkpointEvery 0
    --vtkevery 100
)

if [[ ! -f "$PARAMS" ]]; then
    echo "Parameter file not found: $PARAMS" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found in PATH." >&2
    exit 1
fi

CODEGEN_PY="$(command -v python3)"
if [[ -x "$VENV/bin/python" ]]; then
    CODEGEN_PY="$VENV/bin/python"
fi

# Always (re)configure to ensure FluidSim_cpu is enabled in local builds.
cmake -S "$WALBERLA_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DWALBERLA_BUILD_WITH_LTO=OFF \
    -DWALBERLA_BUILD_WITH_MPI=ON \
    -DWALBERLA_BUILD_WITH_OPENMP=ON \
    -DWALBERLA_BUILD_WITH_OPENMESH=ON \
    -DWALBERLA_BUILD_WITH_CODEGEN=ON \
    -DWALBERLA_ENABLE_SWEEPGEN=ON \
    -DWALBERLA_SWEEPGEN_MANAGED_VENV=OFF \
    -DWALBERLA_BUILD_TESTS=OFF \
    -DPython_EXECUTABLE="$CODEGEN_PY" \
    -DWALBERLA_CODEGEN_PYTHON="$CODEGEN_PY"

cmake --build "$BUILD_DIR" --target FluidSim_cpu --parallel "$BUILD_JOBS" 2>&1 | sed '/^ninja: no work to do\.?$/d'

if [[ -x "$EXE_WALBERLA_LAYOUT" ]]; then
    EXE="$EXE_WALBERLA_LAYOUT"
elif [[ -x "$EXE_APPS_LAYOUT" ]]; then
    EXE="$EXE_APPS_LAYOUT"
else
    echo "FluidSim executable not found after build." >&2
    echo "Checked: $EXE_WALBERLA_LAYOUT" >&2
    echo "Checked: $EXE_APPS_LAYOUT" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"
export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES
mpirun --allow-run-as-root -np "$NP" stdbuf -oL -eL "$EXE" "$PARAMS" "${FLAGS[@]}" "$@" > "$LOG_FILE" 2>&1
