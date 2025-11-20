/*
 * Created on Thu Dec 19 2024
 *
 * This file is part of the exercises for the Lectures on
 * Foundations of High Performance Computing
 * given at
 *     Master in HPC and
 *     Master in Data Science and Scientific Computing
 *
 * @ SISSA, ICTP and University of Trieste
 *
 * contact: taffoni@oats.inaf.it
 *
 *
 *
 *
 * The MIT License (MIT)
 * Copyright (c) 2024 Taffoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <omp.h>
#include <stdio.h>

int main() {
    int num_devices = omp_get_num_devices();
    printf("Number of devices available: %d\n", num_devices);

    #pragma omp target device(0)
    {
        printf("Running on device: %d\n", omp_get_device_num());
    }

    #pragma omp target device(1)
    {
        printf("Running on device: %d\n", omp_get_device_num());
    }

    #pragma omp target
    {
        int max_threads = omp_get_max_threads();
        printf("Maximum threads per team: %d\n", max_threads);
    }
    
#pragma omp target teams
{
    if(  omp_get_team_num() == 1 )     printf("Maximum teams: %d\n", omp_get_num_teams());
}

        #pragma omp target teams num_teams(4) thread_limit(8) device(0)
    {
        int device_num = omp_get_device_num();
        int num_teams = omp_get_num_teams();
        int team_num = omp_get_team_num();
        int num_threads = omp_get_num_threads();
        int thread_num = omp_get_thread_num();

        #pragma omp parallel
        {
            printf("Device: %d, Team: %d/%d, Thread: %d/%d\n",
                   device_num, team_num, num_teams, thread_num, num_threads);
        }
    }

    return 0;
}