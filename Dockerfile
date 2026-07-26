FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# torch==2.1.0 (pinned in requirements.txt) predates Blackwell (sm_120) support;
# install the current cu128 wheels instead, same as the nika image.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# accelerate/deepspeed/torchac/timm/pytorch-msssim intentionally unpinned here
# (their pinned versions in requirements.txt were tested against torch==2.1.0).
# Only deepspeed's pure-python flops_profiler is actually used, so skip building
# its compiled ops (also avoids needing a GPU present at image-build time).
ENV DS_BUILD_OPS=0
RUN pip install --no-cache-dir \
    accelerate \
    deepspeed \
    pytorch-msssim \
    timm \
    torchac \
    numpy \
    pyyaml

EXPOSE 6006
