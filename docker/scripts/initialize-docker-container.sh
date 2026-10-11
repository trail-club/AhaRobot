#!/bin/bash
# Runs once when the container is started via docker compose up.
# Keeps the container alive so `docker exec` can attach shells to it.

set -e

WS=/app/overlay_ws
UPSTREAM=/app/upstream

# Symlink upstream packages into overlay_ws/src so colcon picks them up in one build.
# Only pkgs that overlay actually needs are linked; add more here later if required.
mkdir -p ${WS}/src
for entry in \
    astra_description:astra_description \
    astra_controller_interfaces:astra_controller_interfaces \
    astra_controller:astra_controller \
    sobits_gazebo_worlds:sobits_gazebo_worlds \
    tmc_wrs_gz_worlds:tmc_wrs_gz/tmc_wrs_gz_worlds; do
    pkg="${entry%%:*}"
    source_path="${UPSTREAM}/${entry#*:}"
    if [ -d "${source_path}" ] && [ ! -e "${WS}/src/${pkg}" ]; then
        ln -s "${source_path}" "${WS}/src/${pkg}"
        echo "[init] linked ${pkg} into overlay_ws/src"
    fi
done

# rosdep needs a per-user cache; the Dockerfile ran it as root, so init here as trail.
if [ ! -d "${HOME}/.ros/rosdep/sources.cache" ]; then
    echo "[init] running rosdep update (first time for this user)"
    rosdep update || true
fi

# Install ROS deps for all packages present under overlay_ws/src.
# Safe to re-run; rosdep is a no-op when everything is satisfied.
if [ -d "${WS}/src" ]; then
    source /opt/ros/jazzy/setup.bash
    cd ${WS}
    # The Dockerfile removes apt indexes to keep the image small. Refresh
    # them on first startup so rosdep can resolve packages installed at runtime.
    shopt -s nullglob
    apt_package_lists=(/var/lib/apt/lists/*_Packages*)
    shopt -u nullglob
    if [ ${#apt_package_lists[@]} -eq 0 ]; then
        echo "[init] refreshing apt package indexes"
        sudo apt-get update
    fi
    # These upstream dependencies serve the unused random-world manager only.
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "gz_human_sim sobits_interfaces" || \
        echo "[init] rosdep reported unresolved deps (continuing)"
fi

# Keep container alive.
tail -f /dev/null
