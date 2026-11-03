# ASR and TTS inference for aha_perception.audio over an SSH stdio channel.
# The NVIDIA image supports both the DGX Spark's ARM64 GB10 and the x86-64 RTX 50 series.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      git libsndfile1 ffmpeg sox espeak-ng \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY tools/perception/audio_worker/pyproject.toml tools/perception/audio_worker/uv.lock /app/tools/perception/audio_worker/
ENV UV_PROJECT_ENVIRONMENT=/opt/aharobot-audio-venv
# --compile-bytecode: NeMo 2.7.3 corrupts Python's shared b" " when its CUDA-graph
# decoder is compiled at run time, and later .pyc files cache the damage.
RUN uv venv --python python3 --system-site-packages "$UV_PROJECT_ENVIRONMENT" \
 && uv sync --project tools/perception/audio_worker --locked --inexact --no-python-downloads --compile-bytecode
ENV PATH="/opt/aharobot-audio-venv/bin:$PATH"
ENV PYTHONPATH=/app/overlay_ws/src/aha_perception
ENV PYTORCH_KERNEL_CACHE_PATH=/cache/torch/kernels
ENV TRITON_CACHE_DIR=/cache/triton
ENV HF_HOME=/cache/huggingface
ENV HOME=/cache
ENTRYPOINT ["python", "-u", "-m", "aha_perception.audio.worker"]
