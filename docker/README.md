# Using the Dockerfiles

Although the easiest way to run the tutorials is to use Google Colab, we also provide Dockerfiles for those who prefer to run the tutorials locally.

We provide two versions of the Dockerfile: one for CPU and one for GPU. You should use the CPU version if you do not have a compatible Nvidia GPU or if you prefer to run the tutorials on a CPU. The GPU version is recommended for those with a compatible Nvidia GPU, as it will significantly speed up the fitting of voxelwise encoding models.

## CPU Version
The CPU version of the Dockerfile is designed to run on any machine with a CPU. It does not require any special hardware or drivers, making it suitable for a wide range of environments. Note, however, that the CPU version will be significantly slower when fitting voxelwise encoding models compared to the GPU version.

### Prerequisites
- Install Docker on your machine. You can find instructions for your operating system [here](https://docs.docker.com/get-docker/).

### Build the Docker image
To build the CPU version, run the following command from the current `voxelwise_tutorials/docker` directory:

```bash
docker build --tag voxelwise_tutorials --file cpu.Dockerfile . 
```

This will create a Docker image named `voxelwise_tutorials` based on the `cpu.Dockerfile`.

### Run the Docker container
To run the CPU version, use the following command:

```bash
docker run --rm -it --name voxelwise_tutorials --publish 8888:8888 voxelwise_tutorials jupyter-lab --ip 0.0.0.0 
```

This command will start a Docker container from the `voxelwise_tutorials` image, mapping port 8888 on your host machine to port 8888 in the container. The `--rm` flag ensures that the container is removed after it is stopped, and the `-it` flag allows you to interact with the container. Note that all data that will be downloaded during the tutorial will be stored in the container, and it will be removed when you stop the container. 

You should see output similar to the following:

```
To access the server, open this file in a browser:
    file:///home/voxelwise/.local/share/jupyter/runtime/jpserver-7-open.html
Or copy and paste one of these URLs:
    http://f4cb3fce5844:8888/lab?token=73d9628b0e8839023e3409945f06b9ddbdedde95fe630e00
    http://127.0.0.1:8888/lab?token=73d9628b0e8839023e3409945f06b9ddbdedde95fe630e00
```
The URL will contain a token that you can use to access the Jupyter Lab interface. Open your web browser and navigate to `http://127.0.0.1:8888/lab?token=<YOUR_TOKEN>`. This will open the Jupyter Lab interface, where you can start working with the tutorials.

## GPU Version
The GPU version of the Dockerfile is designed to take advantage of Nvidia GPUs for faster computation. This version is recommended if you have a compatible Nvidia GPU and the necessary drivers installed. The GPU version will significantly speed up the fitting of voxelwise encoding models.

## Prerequisites
- Install Docker on your machine. You can find instructions for your operating system [here](https://docs.docker.com/get-docker/).
- Install the Nvidia Container Toolkit. You can find instructions for your operating system [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


## Create a Dockerfile for your version of CUDA

You will need to create a Dockerfile for your version of CUDA. We provide a bash script that will do this for you. The bash script uses [neurodocker](https://www.repronim.org/neurodocker/) to quickly create a Dockerfile. The bash script is called `generate_dockerfile_gpu.sh` and can be found in this directory. To use the script, run the following command:

```bash
bash generate_dockerfile_gpu.sh CUDA_VERSION [<OS_VERSION>]
```

Replace `CUDA_VERSION` with the version of CUDA you have installed (e.g., `10.2`). The optional `<OS_VERSION>` argument allows you to specify the version of Ubuntu you want to use (e.g., `22.04`). If you do not specify an OS version, the script will default to `18.04`.

For example, to create a Dockerfile with CUDA 10.2 and Ubuntu 18.04, run the following command:

```bash
bash generate_dockerfile_gpu.sh 10.2 18.04
```
This will create a Dockerfile named `gpu-cu102.Dockerfile` in the current directory. You can then use this Dockerfile to build the GPU version of the tutorials.

> [!IMPORTANT]
> Make sure to use the correct version of CUDA that matches your Nvidia driver. Also note that not all CUDA versions are available in the latest Ubuntu images, so make sure to select both the appropriate CUDA and Ubuntu versions.

## Build the Docker image
To build the GPU version, run the following command, replacing `gpu-cu102.Dockerfile` with the name of the Dockerfile you created in the previous step:

```bash
docker build --tag voxelwise_tutorials --file gpu-cu102.Dockerfile . 
```

This will create a Docker image named `voxelwise_tutorials` based on the your GPU Dockerfile.

## Run the Docker container
To run the GPU version, use the following command:

```bash
docker run --rm -it --gpus all --name voxelwise_tutorials --publish 8888:8888 voxelwise_tutorials jupyter-lab --ip 0.0.0.0 
```

Then you can follow the same steps as in the CPU version to access the Jupyter Lab interface.