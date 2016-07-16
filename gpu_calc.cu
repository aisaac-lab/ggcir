#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

__global__ void device_dft_idft()
{
  //This kernel calculates the dft and idft on GPU.
  //Please modify this kernel!!
}

void launch_kernel(double *h_re, double *h_im, int num, int flag)
{
  //This function launches the dft kernel.
  //Please modify this function for dft kernel .
  //You need to allocate the device memory and so on in this function.

  //device_dft_idft<<<blockDim, threadDim>>>(argment1, argment2, etc...);
}



