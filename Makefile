USER_ID ?= $(shell id -u)
GROUP_ID ?= $(shell id -g)

IMAGE ?= trail/aharobot-jazzy
CONTAINER ?= aharobot_aha_project_1
FORWARD_PORT ?= 8765
FORWARD_HOST_PORT ?= 1$(FORWARD_PORT)
AHA_GUI_HOST ?= $(if $(filter Darwin,$(shell uname -s)),0.0.0.0,127.0.0.1)

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
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) AHA_GUI_HOST=$(AHA_GUI_HOST) \
		docker compose --compatibility -p aharobot -f ./docker/docker-compose.yml up -d

.PHONY: shell
shell:
	docker exec -it $(CONTAINER) bash

.PHONY: forward
forward:
	# macOS: reach a port of the host-network dev container from the Mac.
	# The Mac-side port must differ, because Docker binds it inside the VM too.
	docker run -d --rm --name aharobot_forward_$(FORWARD_PORT) \
		-p 127.0.0.1:$(FORWARD_HOST_PORT):$(FORWARD_PORT) \
		-v "$(CURDIR)/docker/scripts:/scripts:ro" \
		--entrypoint python3 $(IMAGE) /scripts/forward_port.py $(FORWARD_PORT)
	@echo "Open http://localhost:$(FORWARD_HOST_PORT)/"

.PHONY: down
down:
	-docker rm -f $$(docker ps -aq --filter name=aharobot_forward_) 2>/dev/null
	docker compose --compatibility -p aharobot \
		-f ./docker/docker-compose.yml \
		-f ./docker/docker-compose.gpu.yml \
		-f ./docker/docker-compose.wsl-novnc.yml \
		-f ./docker/docker-compose.audio.yml \
		--profile darwin --profile linux down --remove-orphans

.PHONY: ws-build
ws-build:
	docker exec -it $(CONTAINER) bash -lc \
		"cd /app/overlay_ws && colcon build --symlink-install --packages-up-to aha_bringup"

.PHONY: test
test:
	pre-commit run --all-files
	bash tools/perception/macos/test.sh
	docker exec $(CONTAINER) bash -lc \
		"cd /app/overlay_ws && colcon build --symlink-install --base-paths src ../tests/ros && source install/setup.bash && colcon test --base-paths src ../tests/ros --packages-skip astra_description astra_controller astra_controller_interfaces sobits_gazebo_worlds tmc_wrs_gz_worlds --event-handlers console_direct+ && colcon test-result --verbose"

.PHONY: sim
sim:
	docker exec -it $(CONTAINER) bash -lc \
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

# 推論先 (dgx | trailpc | local) と SSH のユーザー名。例: make audio-run AUDIO_TARGET=trailpc SSH_USER=xxx
AUDIO_TARGET ?= dgx
SSH_USER ?=

# マイクの入力元 (PulseAudio source 名)。例: make audio-run PULSE_SOURCE=alsa_input.usb-xxx
# 一覧は `make audio-shell` 内で `pactl list sources short`
PULSE_SOURCE ?=

.PHONY: audio-run
audio-run:
	$(AUDIO_COMPOSE) exec $(if $(PULSE_SOURCE),-e PULSE_SOURCE=$(PULSE_SOURCE)) aha_audio bash -lc \
		"source /app/overlay_ws/install/setup.bash && ros2 launch aha_perception audio.launch.py target:=$(AUDIO_TARGET) ssh_user:=$(SSH_USER)"
