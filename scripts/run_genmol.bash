#!/bin/bash

set -euo pipefail

N=${1:-2}
CONTAINER_PREFIX="genmol"

# Function to check if a port is in use
is_port_in_use() {
    local port=$1
    ss -ltn | awk '{print $4}' | grep -q ":$port$"
}

# Find N contiguous open ports
find_contiguous_ports() {
    local base_port=8000
    while :; do
        local ports=()
        for ((i = 0; i < N; i++)); do
            local candidate_port=$((base_port + i))
            if is_port_in_use "$candidate_port"; then
                base_port=$((candidate_port + 1))
                continue 2  # Restart search if any port in range is in use
            fi
            ports+=("$candidate_port")
        done
        echo "${ports[@]}"
        return
    done
}

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
export NUM_GPUS  # Required for parallel

PORTS=($(find_contiguous_ports))
# Track all spawned containers
CONTAINER_NAMES=()
for port in "${PORTS[@]}"; do
    CONTAINER_NAMES+=("${CONTAINER_PREFIX}-${port}")
done

# Cleanup function to stop and remove containers
cleanup() {
    echo "Cleaning up running containers..."
    for container in "${CONTAINER_NAMES[@]}"; do
        if docker ps -q -f name="^${container}$" | grep -q .; then
            echo "Stopping and removing container: $container"
            docker stop "$container" && docker rm "$container"
        fi
    done
    exit 0
}

# Run parallel processes with unique container names and assigned GPUs (round-robin)
parallel -j "$N" --link \
    'CUDA_VISIBLE_DEVICES=$(({#} % NUM_GPUS)) HOST_PORT={1} CONTAINER_NAME="genmol-{1}" make run_genmol' ::: "${PORTS[@]}"
