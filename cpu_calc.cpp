/**************************************/
/* You don't have to change this code */
/* Please modify the "gpu_calc.cu"    */
/**************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

void dft_swap(double *re, double *im, int num)
{
  int i;

  for(i=0; i<num/2; i++){
    swap(&re[i], &re[num/2 + i]);
    swap(&im[i], &im[num/2 + i]);
  }
}

void swap(double *a, double *b)
{
  double temp;

  temp = *a;
  *a = *b;
  *b = temp;
}

int dft_idft(double *re, double *im, int num, int flag)
{
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

void launch_cpu(double *re, double *im, int num, int flag)
{
  dft_idft(re, im, num, flag);
  
}
