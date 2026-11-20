# syntax=docker/dockerfile:1
# Image for laundry-bot GPU runs on RunPod (or any NVIDIA machine).
# It holds software only - no code, no secrets. Each run downloads the code
# from GitHub at the exact commit the launcher asks for (see bootstrap.sh).
#
# CUDA 12.8 so one image runs on everything from an RTX A4000 to an RTX 5090.
ARG BASE=nvidia/cuda:12.8.1-runtime-ubuntu24.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git ca-certificates curl \
        libegl1 libgl1 libgles2 libglvnd0 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Let the GL loader find NVIDIA's EGL driver (the driver itself is mounted in
# by the NVIDIA container runtime at start), for GPU rendering of camera images.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt requirements-cloud.txt /tmp/
# `extra_ca` is only used for local test builds behind a TLS-inspecting proxy;
# a build secret is never stored in the image, and GitHub's builds don't set it.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-cloud.txt

COPY docker/bootstrap.sh /opt/laundry/bootstrap.sh
WORKDIR /workspace
CMD ["bash", "/opt/laundry/bootstrap.sh"]
