[[_TOC_]]

# Trieste GPU Computing School 2025

An intensive hands-on school focused on modern GPU computing for scientific and high-performance computing - Trieste (Italy) 2025
Instructors (from CINECA):
- Nitin Shukla
- Miacheal Redenti

## Access to Leonardo (Training Accounts)

<details>
  <summary><strong>⚠️ Access & Credential Warnings (click to expand)</strong></summary>

**Training usernames**: You should have received login credentials (username and passwords) via email. These are fixed and can not be changed.

**Validity**: until end of day Sunday 23 Nov 2025 (access disabled starting Monday 24 Nov).  

<div align="left">⚠️ Before receiving credentials each student must complete the <a href="https://eventi.cineca.it/en/hpc/reserved-course/20251117">privacy & responsibility form</a>
</div>

<div align="left">⚠️ Fail2ban: 3 wrong password attempts from the same public IP blocks that IP for ~20 minutes (affects everyone behind the same NAT). Ask for help before retrying. 
</div>


You can connect to a Leonardo login node using SSH connection:
```bash
ssh a08traXX@login.leonardo.cineca.it
```

</details>

## Hands-on Session

<details>
  <summary>Setup + Resource Allocation (click to expand)</summary>

Hands on sessions will be held on **Leonardo** cluster at CINECA. For a detailed guide on Leonardo cluster see [here](https://docs.hpc.cineca.it/hpc/leonardo.html).

### Clone repository

```bash
git clone https://gitlab.hpc.cineca.it/training/trieste-gpu-computing.git
```

### Allocating resources 

Please complete and compile the exercises on **login node**. Once you have completed the exercise you can request a GPU resource on a **compute node** to run it. If you want to run on a dedicated GPU, ask the scheduler for GPU resourses with:

```shell
srun -X -t 10 -N 1 --ntasks-per-node 4 --mem=8gb --gres=gpu:1 -p boost_usr_prod -A tra25_gputs --reservation=s_tra_gputs --pty /usr/bin/bash
```

or 

```shell
salloc -N1 --cpus-per-task=1 --ntasks-per-node=8 -A tra25_gputs --reservation=s_tra_gputs -t 00:10:00 -p boost_usr_prod  --gres=gpu:1
```

Within this command you reserve a GPU on a compute node for 10 minutes with 4 CPU cores and 8GB of memory 

In order to avoid to repeat the `srun` command you can source the `get_gpu` file located in the root directory of repository and an alias to the `srun` command is created.  Therefore, first:

```shell
source get_gpu
```

Once you have sourced the `get_gpu` file you can simply reserve a GPU with the following alias:

```shell
get_gpu
```

**NB**: Please do **NOT** use **login node** to run your exercises but use **compute node**.
login node is only used to complete source code and compile it.

</details>



➡️ See the **Day 1: CUDA** exercises: [cuda/README.md](cuda/README.md)

➡️ See the **Day 2: OpenACC** exercises: [openacc/README.md](openacc/README.md)
