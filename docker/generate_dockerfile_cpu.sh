#!/bin/bash 
docker run --rm repronim/neurodocker:latest generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --env "DEBIAN_FRONTEND=noninteractive" \
    --run "chmod 777 /tmp" \
    --install build-essential git ca-certificates netbase\
    --miniconda \
    version="py311_24.4.0-0" \
    conda_install="gxx_linux-64 notebook jupyterlab numpy git-annex ipywidgets" \
    --run "chmod 777 /opt/miniconda-py311_24.4.0-0/share" \
    --run "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" \
    --user voxelwise \
    --workdir /home/voxelwise \
    --run "git clone https://github.com/gallantlab/voxelwise_tutorials.git --depth 1" \
    --run "python -m pip install voxelwise_tutorials" \
    --run "git config --global user.email 'you@example.com'" \
    --run "git config --global user.name 'Your Name'" \
    --workdir /home/voxelwise/voxelwise_tutorials/tutorials/notebooks/shortclips \
    > cpu.Dockerfile