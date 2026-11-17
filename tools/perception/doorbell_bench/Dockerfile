# Doorbell / knock detection benchmark. Same CUDA base as the audio worker, so
# both the RTX 50 series (sm_120) and the DGX Spark run it.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY tools/perception/doorbell_bench/pyproject.toml tools/perception/doorbell_bench/uv.lock /app/tools/perception/doorbell_bench/
ENV UV_PROJECT_ENVIRONMENT=/opt/doorbell-bench-venv
RUN uv venv --python python3 --system-site-packages "$UV_PROJECT_ENVIRONMENT" \
 && uv sync --project tools/perception/doorbell_bench --locked --inexact --no-python-downloads

# Model code that is not on PyPI, at fixed commits (weights are fetched at run time).
RUN git clone -q https://github.com/fschmid56/PretrainedSED /opt/third_party/PretrainedSED \
 && git -C /opt/third_party/PretrainedSED checkout -q 1aa47e482f7e89904cba2338999345025d8b4e36 \
 && git clone -q https://github.com/fschmid56/EfficientAT /opt/third_party/EfficientAT \
 && git -C /opt/third_party/EfficientAT checkout -q a425fdce92572e602a1d5634799bd9f1f2efa806

ENV PATH="/opt/doorbell-bench-venv/bin:$PATH"
ENV THIRD_PARTY=/opt/third_party
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV HOME=/cache
WORKDIR /app/tools/perception/doorbell_bench
