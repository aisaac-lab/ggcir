#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

const int BLOCK_SIZE = 256;

__global__ void device_dft_idft(double *d_re, double *d_im, double *d_temp_re, double *d_temp_im, int flag, int num)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int j;
  for(j=0; j<num; j++){
    d_temp_re[i] += d_re[j]*cos(2*PI*i*j/num) + flag*d_im[j]*sin(2*PI*i*j/num);
    d_temp_im[i] += -flag*d_re[j]*sin(2*PI*i*j/num) + d_im[j]*cos(2*PI*i*j/num);

    // d_temp_re[i] += (float)i;
    // d_temp_im[i] += (float)i;
  }
  if(flag == IDFT){
    d_temp_re[i] /= num;
    d_temp_im[i] /= num;
  }
}

void launch_kernel(double *h_re, double *h_im, int num, int flag)
{
  //This function launches the dft kernel.
  //Please modify this function for dft kernel .
  //You need to allocate the device memory and so on in this function.

  //device_dft_idft<<<blockDim, threadDim>>>(argment1, argment2, etc...);
  int i;
  double *d_re, *d_im;
  double *d_temp_re, *d_temp_im;
  double *h_temp_re, *h_temp_im;
  dim3 dim_grid(num/BLOCK_SIZE, 1, 1), dim_block(num, 1, 1);

  cudaMalloc((void **)&d_temp_re, sizeof(double) * num);
  cudaMalloc((void **)&d_temp_im, sizeof(double) * num);
  cudaMalloc((void **)&d_re, sizeof(double) * num);
  cudaMalloc((void **)&d_im, sizeof(double) * num);

  if((h_temp_re = (double*)malloc(sizeof(double)*num)) == NULL){
    fprintf(stderr, "Allocationerror!\n");
  }

  if((h_temp_im = (double*)malloc(sizeof(double)*num)) == NULL){
    fprintf(stderr, "Allocationerror!\n");
    free(h_temp_re);
  }

  for(i=0; i<num; i++){
    h_temp_re[i] = h_temp_im[i] = 0.0;
  }

  cudaMemcpy(d_temp_re, h_temp_re, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_temp_im, h_temp_im, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_re, h_re, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_im, h_im, sizeof(double) * num, cudaMemcpyHostToDevice);

  // device_dft_idft<<<dim_grid, dim_block>>>(d_re, d_im, d_temp_re, d_temp_im, flag, num);
  // cudaMemcpy(h_re, d_temp_re, sizeof(double) * num, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_im, d_temp_im, sizeof(double) * num, cudaMemcpyDeviceToHost);

  for(i=0; i<num; i++){
    for(int j=0; j<num; j++){
      h_temp_re[i] += h_re[j]*cos(2*PI*i*j/num) + flag*h_im[j]*sin(2*PI*i*j/num);
      h_temp_im[i] += -flag*h_re[j]*sin(2*PI*i*j/num) + h_im[j]*cos(2*PI*i*j/num);
    }
    if(flag == IDFT){
      h_temp_re[i] /= num;
      h_temp_im[i] /= num;
    }
  }

  for(i=0; i<num; i++){
    h_re[i] = h_temp_re[i];
    h_im[i] = h_temp_im[i];
  }

  cudaFree(d_re);
  cudaFree(d_im);
  cudaFree(d_temp_re);
  cudaFree(d_temp_im);

  free(h_temp_re);
  free(h_temp_im);
}
