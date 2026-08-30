#!/bin/bash
# Runs once when the container is started via docker compose up.
# Keeps the container alive so `docker exec` can attach shells to it.

set -e

WS=/app/overlay_ws
UPSTREAM=/app/upstream

# Symlink upstream packages into overlay_ws/src so colcon picks them up in one build.
# Only pkgs that overlay actually needs are linked; add more here later if required.
mkdir -p ${WS}/src
for pkg in astra_description astra_controller_interfaces; do
    if [ -d "${UPSTREAM}/${pkg}" ] && [ ! -e "${WS}/src/${pkg}" ]; then
        ln -s "${UPSTREAM}/${pkg}" "${WS}/src/${pkg}"
        echo "[init] linked ${pkg} into overlay_ws/src"
    fi
done

# Install ROS deps for all packages present under overlay_ws/src.
# Safe to re-run; rosdep is a no-op when everything is satisfied.
if [ -d "${WS}/src" ]; then
    source /opt/ros/jazzy/setup.bash
    cd ${WS}
    rosdep install --from-paths src --ignore-src -r -y || \
        echo "[init] rosdep reported unresolved deps (continuing)"
fi

# Keep container alive.
tail -f /dev/null
