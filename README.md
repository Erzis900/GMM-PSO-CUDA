# GMM-PSO-CUDA
 
This repository is based on [constraintsGM](https://github.com/LRMPUT/constraintsGM). Credits to [@dominikbelter](https://github.com/dominikbelter).

The project was done on Ubuntu 24.04.

## CUDA Installation
Follow the instructions:
https://developer.nvidia.com/cuda-downloads?target_os=Linux

## Building
```
cd build
cmake ..
cmake --build .
```

## Configuration

The configuration file is located in `resouces/configGlobal.xml`.
You can set the population size, number of Gaussians, maximum number of iterations, etc.

## Running
```
cd build/bin
./GaussianMixture
```