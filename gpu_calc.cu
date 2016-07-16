#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

__global__ void device_dft_idft()
{
  //This kernel calculates the dft and idft on GPU.
  //Please modify this kernel!!
}

void launch_kernel(double *re, double *im, int num, int flag)
{
  //This function launches the dft kernel.
  //Please modify this function for dft kernel .
  //You need to allocate the device memory and so on in this function.

  //device_dft_idft<<<blockDim, threadDim>>>(argment1, argment2, etc...);
  int i, j;
  double *temp_re, *temp_im;

  if((temp_re = (double*)malloc(sizeof(double)*num)) == NULL){
    fprintf(stderr, "Allocationerror!\n");
    return 1;
  }

  if((temp_im = (double*)malloc(sizeof(double)*num)) == NULL){
    fprintf(stderr, "Allocationerror!\n");
    free(temp_re);
    return 1;
  }

  for(i=0; i<num; i++){
    temp_re[i] = temp_im[i] = 0.0;
  }

  for(i=0; i<num; i++){
    for(j=0; j<num; j++){
      temp_re[i] += re[j]*cos(2*PI*i*j/num) + flag*im[j]*sin(2*PI*i*j/num);
      temp_im[i] += -flag*re[j]*sin(2*PI*i*j/num) + im[j]*cos(2*PI*i*j/num);
    }
  if(flag == IDFT){
    temp_re[i] /= num;
    temp_im[i] /= num;
  }
  }

  for(i=0; i<num; i++){
    re[i] = temp_re[i];
    im[i] = temp_im[i];
  }

  free(temp_re);
  free(temp_im);

  return 0;
}
