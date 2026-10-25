# The NVIDIA image supports the DGX Spark's ARM64 GB10 GPU.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY tools/perception/macos/sam31/pyproject.toml tools/perception/macos/sam31/uv.lock /app/tools/perception/macos/sam31/
COPY upstream/sam3 /app/upstream/sam3
ENV UV_PROJECT_ENVIRONMENT=/opt/aharobot-sam31-venv
RUN uv venv --python python3 --system-site-packages "$UV_PROJECT_ENVIRONMENT" \
 && uv sync --project tools/perception/macos/sam31 --locked --inexact --no-python-downloads \
 && "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch, torchvision, triton, sam3"
ENV PATH="/opt/aharobot-sam31-venv/bin:$PATH"
ENV PYTHONPATH=/app/overlay_ws/src/aha_perception
ENV PYTORCH_KERNEL_CACHE_PATH=/cache/torch/kernels
ENV TRITON_CACHE_DIR=/cache/triton
ENTRYPOINT ["python", "-u", "-m", "aha_perception.sam31_worker"]
