#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
# Docker Desktop mounts the Mac ssh-agent as root:root 0660; person_perception
# runs as the container user.
if [[ -S "${SSH_AUTH_SOCK:-}" && ! -w "$SSH_AUTH_SOCK" ]]; then
  sudo chown "$(id -u):$(id -g)" "$SSH_AUTH_SOCK"
fi
cd /app/overlay_ws
colcon build --symlink-install --packages-select aha_perception
source install/setup.bash
set -u

# Compose starts NoVNC and ROS concurrently. RViz must wait for its X server.
echo '[perception] waiting for NoVNC X server'
display_ready=false
for _ in {1..60}; do
  if (echo > /dev/tcp/novnc/6000) >/dev/null 2>&1; then
    display_ready=true
    break
  fi
  sleep 1
done
if [[ "$display_ready" != true ]]; then
  echo '[perception] NoVNC X server was not ready within 60 seconds' >&2
  exit 1
fi

bridge_pid=""
view_pid=""
cleanup() {
  for pid in "$view_pid" "$bridge_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  address:=0.0.0.0 port:=9090 &
bridge_pid=$!

ros2 launch aha_perception camera_view.launch.py &
view_pid=$!

echo '[perception] rosbridge and RViz started'
wait -n "$bridge_pid" "$view_pid" || true
echo '[perception] a ROS process exited; stopping the camera stack' >&2
exit 1
