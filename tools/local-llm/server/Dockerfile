# TensorFold 0.6.5 (2026-10-03) on NVIDIA PyTorch for DGX Spark (GB10)
# [grammar] (xgrammar) enables JSON schema output
FROM nvcr.io/nvidia/pytorch:26.07-py3
RUN python -m pip install --no-cache-dir "tensorfold[grammar] @ git+https://github.com/ashhart/TensorFold.git@609ca41"
COPY keysync.sh /usr/local/bin/keysync.sh
