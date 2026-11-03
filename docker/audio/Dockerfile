# 音声コンテナ (aha_audio): ROS 2 Jazzy + PyTorch (CUDA 12.8) + NeMo / Haas / Qwen3-ASR / Kokoro.
# 開発コンテナ (docker/Dockerfile) とは分けて、重い ML 依存をこちらに閉じ込める。
FROM ros:jazzy

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions python3-yaml \
      build-essential cmake git curl sudo openssh-client \
      libportaudio2 libasound2-plugins alsa-utils pulseaudio-utils \
      libsndfile1 ffmpeg sox espeak-ng \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 専用イメージなのでシステム Python に直接入れる。ros2 run の shebang (/usr/bin/python3) から
# そのまま import できるようにするため venv は使わない。apt 版 (numpy 等) は pip から消せないので
# --ignore-installed で /usr/local 側に入れ、apt 版より優先させる。
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
COPY docker/audio/requirements.txt docker/audio/constraints.txt /tmp/audio/
RUN pip install --ignore-installed -c /tmp/audio/constraints.txt \
      torch torchaudio -r /tmp/audio/requirements.txt \
 && pip freeze > /opt/audio-requirements.lock

COPY docker/audio/asound.conf /etc/asound.conf

# Non-root user matching host uid/gid (docker/Dockerfile と同じ方式)
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

# モデルキャッシュ (named volume をここにマウント。初回マウント時に所有者が引き継がれる)
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p ${HF_HOME} /home/${USERNAME}/.local/share \
 && chown -R ${USER_ID}:${GROUP_ID} ${HF_HOME} /home/${USERNAME} /app

COPY ./docker/scripts/ /app/docker/scripts/
RUN chmod +x /app/docker/scripts/*.sh

USER ${USERNAME}
RUN echo 'source /app/docker/scripts/initialize-bash-script.sh' >> ~/.bashrc && \
    echo 'source /app/docker/scripts/initialize-bash-script.sh' >> ~/.profile

ENTRYPOINT []
CMD ["/bin/bash"]
