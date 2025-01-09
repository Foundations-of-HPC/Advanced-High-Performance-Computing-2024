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
#include <stdio.h>
#include <stdlib.h>
#pragma omp declare target
double mean_value(double* array, size_t array_size){
double sum = 0.0;
for(size_t i=0; i<array_size; ++i)
sum += array[i];
return sum/array_size;
}
#pragma omp end declare target

void rand_init(double* array, size_t array_size)
{
srand((unsigned) 12345900);
for (size_t i=0; i<array_size; ++i)
array[i] = 2.*((double)rand()/RAND_MAX -0.5);
}


void iterate(double* array, size_t array_size, size_t cell_size){
double local_mean;
#pragma omp target teams distribute parallel for simd
for (size_t i = cell_size/2; i< array_size-cell_size/2; ++i)
{
local_mean = mean_value(&array[i-cell_size/2], cell_size);
if (local_mean < 0.)
array[i] += 0.1;
else if (local_mean > 0.)
array[i] -= 0.1;
}
}

int main(void){
size_t num_cols = 50000;
size_t num_rows = 3000;
double* table = (double*) malloc(num_rows*num_cols*sizeof(double));
double* mean_values = (double*) malloc(num_rows*sizeof(double));
// We initialize the first row with random values between -1 and 1
rand_init(table, num_cols);
printf("...\n");
#pragma omp target enter data map(to:table[0:num_rows*num_cols])
for (size_t i=1; i<num_rows; ++i)
iterate(&table[i*num_cols], num_cols, 32);

#pragma omp target teams distribute parallel for simd map(from:mean_values[0:num_rows])
{
for (size_t i=0; i<num_rows; ++i)
mean_values[i] = mean_value(&(table[i*num_cols]), num_cols);
}
#pragma omp target exit data map(delete:table)
for (size_t i=0; i<10; ++i)
printf("Mean value of row %6d=%10.5f\n", i, table[i]);
printf("...\n");
for (size_t i=num_rows-10; i<num_rows; ++i)
printf("Mean value of row %6d=%10.5f\n", i, table[i]);
return 0;
}