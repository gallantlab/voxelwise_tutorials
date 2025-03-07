#!/bin/bash 
# Check if an argument is passed. If not, explain how to use it. 
# First argument is the CUDA version.
# Second optional argument is the ubuntu version.
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CUDA_VERSION> [<OS_VERSION>]"
    echo "Example: $0 10.2 18.04"
    echo "If OS_VERSION is not provided, it defaults to 18.04."
    exit 1
fi
CUDA_VERSION="$1"
# Check if the second argument is provided, if not set it to 18.04
if [ $# -eq 2 ]; then
    OS_VERSION="$2"
else
    OS_VERSION="18.04"
fi
# Check if the CUDA version is valid
if [[ ! "$CUDA_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid CUDA version format. Please use the format X.Y"
    exit 1
fi

BASE_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION}"
echo "Using CUDA version: $CUDA_VERSION"
echo "Using OS version: ubuntu-$OS_VERSION"
echo "Using base image: $BASE_IMAGE"

CUDA_VERSION_SHORT=$(echo "${CUDA_VERSION}" | cut -d'.' -f1-2 | tr -d '.')
echo "Using CUDA version short: $CUDA_VERSION_SHORT"

CONDA_VERSION="py310_25.1.1-2"
echo "Using conda version: $CONDA_VERSION"

# Generate the Dockerfile using Neurodocker
docker run --rm repronim/neurodocker:latest generate docker \
    --pkg-manager apt \
    --base-image $BASE_IMAGE \
    --env "DEBIAN_FRONTEND=noninteractive" \
    --run "chmod 777 /tmp" \
    --install build-essential git ca-certificates netbase\
    --miniconda \
    version="$CONDA_VERSION" \
    conda_install="gxx_linux-64 notebook jupyterlab numpy<2 git-annex ipywidgets" \
    --run "chmod 777 /opt/miniconda-$CONDA_VERSION/share" \
    --run "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu"$CUDA_VERSION_SHORT \
    --user voxelwise \
    --workdir /home/voxelwise \
    --run "git clone https://github.com/gallantlab/voxelwise_tutorials.git --depth 1" \
    --run "python -m pip install voxelwise_tutorials" \
    --run "git config --global user.email 'you@example.com'" \
    --run "git config --global user.name 'Your Name'" \
    --workdir /home/voxelwise/voxelwise_tutorials/tutorials/notebooks/shortclips \
    > gpu-cu"$CUDA_VERSION_SHORT".Dockerfile