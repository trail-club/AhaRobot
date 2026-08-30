FROM ros:jazzy

WORKDIR /app/

ARG DEBIAN_FRONTEND=noninteractive

# Enable Open Robotics apt repo for Gazebo Harmonic + ros_gz + gz_ros2_control
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg lsb-release ca-certificates \
 && curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      git make sudo terminator xauth \
      python3-pip python3-colcon-common-extensions python3-rosdep python3-vcstool \
      iputils-ping telnet x11-apps mesa-utils \
      '~nros-jazzy-rqt*' \
      ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-xacro \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-rviz2 \
      ros-jazzy-ros-gz \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-joint-state-broadcaster \
      ros-jazzy-joint-trajectory-controller \
      ros-jazzy-diff-drive-controller \
      ros-jazzy-forward-command-controller \
      gz-harmonic \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# rosdep bootstrap for later "rosdep install" inside the mounted workspace
RUN rosdep update

# Non-root user matching host uid/gid
ARG USER_ID
ARG GROUP_ID
ARG USERNAME=trail

RUN getent group ${GROUP_ID} >/dev/null 2>&1 || groupadd -g ${GROUP_ID} ${USERNAME}
RUN if ! getent passwd ${USER_ID} >/dev/null 2>&1; then \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}; \
    else \
        existing_user=$(getent passwd ${USER_ID} | cut -d: -f1); \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $existing_user 2>/dev/null || true; \
    fi

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/${USERNAME}/.local/share && \
    chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME} && \
    chown ${USER_ID}:${GROUP_ID} /app

COPY ./docker/scripts/ /app/docker/scripts/
RUN chmod +x /app/docker/scripts/*.sh

USER ${USERNAME}

# Source ROS env from both .bashrc (interactive) and .profile (bash -lc)
RUN echo 'source /app/docker/scripts/initialize-bash-script.sh' >> ~/.bashrc && \
    echo 'source /app/docker/scripts/initialize-bash-script.sh' >> ~/.profile

ENTRYPOINT []
CMD ["/bin/bash"]
