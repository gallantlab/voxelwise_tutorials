#!/bin/bash 
docker run --rm repronim/neurodocker:latest generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --env "DEBIAN_FRONTEND=noninteractive" \
    --run "chmod 777 /tmp" \
    --install build-essential git ca-certificates netbase\
    --miniconda \
    version="py311_24.4.0-0" \
    conda_install="python=3.11 gxx_linux-64 notebook numpy=1.26.4 git-annex" \
    --run "chmod 777 /opt/miniconda-py311_24.4.0-0/share" \
    --user nonroot \
    --run "pip install Pillow==9.5.0" \
    --workdir /home/nonroot/voxelwise_tutorials \
    --run "pip install voxelwise_tutorials" \
    --run "git clone --depth 1 https://github.com/gallantlab/voxelwise_tutorials.git" \
    --run "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" \
    --run "git config --global user.email 'you@example.com'" \
    --run "git config --global user.name 'Your Name'" \
    > cpu.Dockerfile