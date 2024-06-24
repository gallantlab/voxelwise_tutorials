# Using the Dockerfiles

To build the CPU version, run 
```
docker build --tag notebook --file cpu.Dockerfile .
```
and run with 
```
docker run --rm -it  --name notebook  --publish 8888:8888     notebook     jupyter-notebook --ip 0.0.0.0 --allow-root
```

To build the GPU version, make sure to have Nvidia Driver and CUDA installed. Check the CUDA version and build with 
```
docker build --build-arg CUDA_VERSION=[version here] --tag notebook --file gpu.Dockerfile .
```
Run with
```
docker run --rm -it  --gpus all --name notebook  --publish 8888:8888     notebook     jupyter-notebook --ip 0.0.0.0 --allow-root
```

Once the container is running, go to `http://localhost:8888/` (or the IP address of the host server) and enter the token from the terminal.