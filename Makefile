USER_ID ?= $(shell id -u)
GROUP_ID ?= $(shell id -g)

IMAGE ?= trail/aharobot-jazzy

.PHONY: build
build:
	docker build \
		-t $(IMAGE) \
		--build-arg USER_ID="$(USER_ID)" \
		--build-arg GROUP_ID="$(GROUP_ID)" \
		-f ./docker/Dockerfile \
		.

.PHONY: up
up:
	# Simple `up`: base compose only. For NoVNC / GPU / WSL, use ./run_docker_container.py
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) \
		docker compose --compatibility -p aharobot -f ./docker/docker-compose.yml up -d

.PHONY: shell
shell:
	docker exec -it aharobot_aha_project_1 bash

.PHONY: down
down:
	docker compose --compatibility -p aharobot \
		-f ./docker/docker-compose.yml \
		-f ./docker/docker-compose.gpu.yml \
		-f ./docker/docker-compose.wsl-novnc.yml \
		-f ./docker/docker-compose.audio.yml \
		--profile darwin --profile linux down --remove-orphans

.PHONY: ws-build
ws-build:
	docker exec -it aharobot_aha_project_1 bash -lc \
		"cd /app/overlay_ws && colcon build --symlink-install --packages-up-to aha_bringup"

.PHONY: sim
sim:
	docker exec -it aharobot_aha_project_1 bash -lc \
		"source /app/overlay_ws/install/setup.bash && ros2 launch aha_bringup sim.launch.py"

.PHONY: clean
clean:
	rm -rf overlay_ws/build overlay_ws/install overlay_ws/log

# ---- audio (aha_perception audio stack) container ----------------------------
AUDIO_IMAGE ?= trail/aharobot-audio
AUDIO_COMPOSE = USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) AUDIO_IMAGE=$(AUDIO_IMAGE) \
	docker compose --compatibility -p aharobot -f ./docker/docker-compose.audio.yml
AUDIO_EXEC = $(AUDIO_COMPOSE) exec aha_audio

.PHONY: audio-build
audio-build:
	docker build \
		-t $(AUDIO_IMAGE) \
		--build-arg USER_ID="$(USER_ID)" \
		--build-arg GROUP_ID="$(GROUP_ID)" \
		-f ./docker/audio/Dockerfile \
		.

.PHONY: audio-lock
audio-lock:
	# イメージ内で解決された Python パッケージの全バージョンを記録する
	docker run --rm $(AUDIO_IMAGE) cat /opt/audio-requirements.lock > docker/audio/requirements.lock

.PHONY: audio-up
audio-up:
	$(AUDIO_COMPOSE) up -d

.PHONY: audio-down
audio-down:
	$(AUDIO_COMPOSE) down

.PHONY: audio-shell
audio-shell:
	$(AUDIO_EXEC) bash

.PHONY: audio-ws-build
audio-ws-build:
	$(AUDIO_EXEC) bash -lc \
		"cd /app/overlay_ws && colcon build --symlink-install --packages-up-to aha_perception"

# マイクの入力元 (PulseAudio source 名)。例: make audio-run PULSE_SOURCE=alsa_input.usb-xxx
# 一覧は `make audio-shell` 内で `pactl list sources short`
PULSE_SOURCE ?=

.PHONY: audio-run
audio-run:
	$(AUDIO_COMPOSE) exec $(if $(PULSE_SOURCE),-e PULSE_SOURCE=$(PULSE_SOURCE)) aha_audio bash -lc \
		"source /app/overlay_ws/install/setup.bash && ros2 launch aha_perception audio.launch.py"
