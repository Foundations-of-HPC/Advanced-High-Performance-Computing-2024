 
# How to compile a code using Openmp on GPUs

OpenMP offloading is supported by different compilers. 
Here a list of the mosto common compilers used:

## Compilers

- **Nvidia Compiler**: HPC standard develompemnt Kit 
- **clang**  https://www.openmp.org/resources/refguides/
- **gcc** https://www.openmp.org/spec-html/5.2/openmp.html

## Nvidia HPC-SDK

NVIDIA HPC SDK

The NVIDIA HPC Software Development Kit (SDK) includes the proven compilers, libraries and software tools essential to maximizing developer productivity and the performance and portability of HPC applications.

```
nvc -mp=gpu -gpu=cc75  -o program program.c
```
You may need to set certain environment variables to control OpenMP offloading behavior:
- OMP_TARGET_OFFLOAD=MANDATORY: Ensures the program fails if GPU offloading isn’t supported.
- OMP_NUM_TEAMS=<value> and OMP_TEAM_LIMIT=<value>: Control GPU parallelism.
- OMP_DISPLAY_ENV=true: Displays OpenMP runtime information for debugging.



## clang

```
clang -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75 -o prog.x prog.c
```
- -fopenmp: Enables OpenMP support.
- -fopenmp-targets=nvptx64: Specifies NVIDIA GPUs as the offloading target.
- -Xopenmp-target -march=sm_75: Sets the GPU architecture (e.g., sm_75 for Compute Capability 7.5).

## gcc

```
gcc -fopenmp -foffload=nvptx-none -foffload-options="-march=sm_75" -o program program.c
```
- -fopenmp: Enables OpenMP support.
- -foffload=nvptx-none: Targets NVIDIA GPUs using the NVPTX backend.
- -foffload-options="-march=sm_75": Specifies the GPU architecture (e.g., sm_75 for Compute Capability 7.5).

## How to use CINECA LEONARDO

On Leonardo it is availble the NVIDIA HPC SDK version 24.3

```
module load nvhpc/24.3 openmpi/4.1.6--nvhpc--24.3
```

Compilation Flags

```
OMP_FLAG=-mp=multicore,gpu    # activate Openmp on CPU and GPU
GPU_FLAG=-gpu=cc80 -target=gpu -gpu=nomanaged       
# Unmanaged requires explicit allocations, i.e., without relying on the automatic management of CUDA mapped memory.
# -gpu=ccnative se non si conosce la compute-capability della GPU
```                                                                                           
compilation command
```
nvc  -O3 -fast -Minfo=all -v -Mneginfo $(GPU_FLAG) $(OMP_FLAG)
```

To activate  debuging support
```
GPU_FLAG_DEBUG=-gpu=ccnative,debug,lineinfo -target=gpu -gpu=nomanaged
```