## Day 1: **CUDA** exercises

You find proposed exercises in the `hands-on` directory:
```bash
cd hands-on
```

There you will find 4 exercises 

```bash
.
├── 01-arrayInit
├── 02-vectorSum
├── 03-twoMatrixMultiplication
└── 04-stream
```

and for each exercise a `Makefile` is presented to help you compile. You can use both C/C++ and Fortran for CUDA hands-on session. For C/C++ exercises you can choose **GNU** or **NVHPC** environment. If you are a **Fortran** user you have to load **NVHPC** environment in order to use CUDA-FORTRAN.


### Load Environment Modules

You can load and set your environment through module:
```bash
module load nvhpc
```

This will set all environment variables (PATH and LD_LIBRARY_PATH) to the NVIDIA HPC SDK which will provide you with compiler and profiling tools for the hands-on sessions.

You can also load, list, inspect or purge the current state of loaded modules through the followin commands:
```bash
module avail # show the full list of available modules
module list # list currently loaded modules
module show <modulename> # inspect setup of a module
module purge # completely unload all loaded modules
```

