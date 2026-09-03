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
