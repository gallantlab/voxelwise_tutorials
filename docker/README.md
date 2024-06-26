# Using the Dockerfiles

To build the CPU version, run 
```
docker build --tag voxelwise_tutorials --file cpu.Dockerfile . 
```
and run with 
```
docker run --rm -it  --name voxelwise_tutorials  --publish 8888:8888     voxelwise_tutorials     jupyter-notebook --ip 0.0.0.0 
```

To build the GPU version, make sure to have Nvidia Driver and CUDA installed. Check the CUDA version and build with 
```
docker build --build-arg CUDA_VERSION=[version here] --tag voxelwise_tutorials --file gpu.Dockerfile . 
```
Run with
```
docker run --rm -it  --gpus all --name voxelwise_tutorials  --publish 8888:8888     voxelwise_tutorials     jupyter-notebook --ip 0.0.0.0 
```

Once the container is running, go to `http://localhost:8888/` (or the IP address of the host server) and enter the token from the terminal.