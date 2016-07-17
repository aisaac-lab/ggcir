#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

const int BLOCK_SIZE = 128;

__global__ void device_dft_idft(double *d_re, double *d_im, double *d_temp_re, double *d_temp_im, int flag, int num)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  for(j=0; j<num; j++){
    d_temp_re[i] += d_re[j]*cos(2*PI*i*j/num) + flag*d_im[j]*sin(2*PI*i*j/num);
    d_temp_im[i] += -flag*d_re[j]*sin(2*PI*i*j/num) + d_im[j]*cos(2*PI*i*j/num);
  }
  if(flag == IDFT){
    d_temp_re[i] /= num;
    d_temp_im[i] /= num;
  }
}

void launch_kernel(double *h_re, double *h_im, int num, int flag)
{
  int i;
  double *d_re, *d_im;
  double *d_temp_re, *d_temp_im;
  double *h_temp_re, *h_temp_im;
  dim3 dim_grid(num/BLOCK_SIZE, 1, 1), dim_block(BLOCK_SIZE, 1, 1);

  cudaMalloc((void **)&d_temp_re, sizeof(double) * num);
  cudaMalloc((void **)&d_temp_im, sizeof(double) * num);
  cudaMalloc((void **)&d_re, sizeof(double) * num);
  cudaMalloc((void **)&d_im, sizeof(double) * num);

  cudaHostAlloc((void **)&h_temp_re, sizeof(double)*num, cudaHostAllocDefault);
  cudaHostAlloc((void **)&h_temp_im, sizeof(double)*num, cudaHostAllocDefault);

  for(i=0; i<num; i++){
    h_temp_re[i] = h_temp_im[i] = 0.0;
  }

  cudaMemcpy(d_temp_re, h_temp_re, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_temp_im, h_temp_im, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_re, h_re, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_im, h_im, sizeof(double) * num, cudaMemcpyHostToDevice);

  device_dft_idft<<<dim_grid, dim_block>>>(d_re, d_im, d_temp_re, d_temp_im, flag, num);
  cudaMemcpy(h_re, d_temp_re, sizeof(double) * num, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_im, d_temp_im, sizeof(double) * num, cudaMemcpyDeviceToHost);

  cudaFree(d_re);
  cudaFree(d_im);
  cudaFree(d_temp_re);
  cudaFree(d_temp_im);

  cudaFreeHost(h_temp_re);
  cudaFreeHost(h_temp_im);
}
