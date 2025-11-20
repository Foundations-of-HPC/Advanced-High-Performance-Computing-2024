# Introduction to OpenACC

## Getting started

```bash
.
├── HandsOn
│   └── 00-laplace2d_serial
├── Solution
│   ├── 00-laplace2d_OpenMP_CPU
│   ├── 01-laplace2d_OpenACC_parallel
│   ├── 02-laplace2d_OpenACC_structured_data
│   ├── 03-laplace2d_OpenACC_unstructured_data
│   └── 04-laplace2d_OpenMP_optimized
└── SourceCode
    ├── OpenACC
    └── OpenMP
```


## To make it easy for you to get started with GitLab, here's a list of recommended next steps.


## Presentation
- Introduction to OpenX


## Checkpoints 

Leonardo cluster will be used for this course. You can connect to a leonardo login node using SSH connection:

```bash 
ssh user.name@login.leonardo.cineca.it
```

## Interactive
 
You can request a GPU resource on a **compute node** to run it.

```bash
salloc -N 1 -t 04:00:00 --ntasks-per-node=32 --gres=gpu:1 -p boost_usr_prod -A user_name
```

## Exercises

A skeleton of exercises are provided.  
At the end of course the **solutions** of all exercises are published in the repository.

#### Load NVIDIA module 

```bash
module load nvhpc
```

## Authors and acknowledgment
Although, slides are modified but it was inspired from various sources.

