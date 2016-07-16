/**************************************/
/* You don't have to change this code */
/* Please modify the "gpu_calc.cu"    */
/**************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <malloc.h>
#include <time.h>
#include "calculation.h"
#include "toolkit.h"
#include "timer.h"

void cpu_function(double *re, double *im, int num, int flag, wtime_t &time)
{
  dft_swap(re, im, num);
  start_timer();
  launch_cpu(re, im, num, flag); //launch_function in cpu_calc.cpp to start the dft function
  stop_timer();
  time = elapsed_millis();
  dft_swap(re, im, num);
}

void gpu_function(double *re, double *im, int num, int flag, wtime_t &time)
{
  dft_swap(re, im, num);
  start_timer();
  launch_kernel(re, im, num, flag); //launch_kernel in gpu_calc.cu to start the dft function
  stop_timer();
  time = elapsed_millis();
  dft_swap(re, im, num);
}

int main(int argc, char **argv)
{
  double *gpu_re, *gpu_im;
  double *cpu_re, *cpu_im;
  wtime_t gpu_dft, gpu_idft;
  wtime_t cpu_dft, cpu_idft;
  double dft_res, idft_res;
  int num;
  int i;
  FILE *fp;
  char filename[100];

  initialize(num, filename, argc, argv);

  if((fp = fopen(filename, "r")) == NULL){
	fprintf(stderr, "file open failur\n");
  }

  gpu_re = (double *)malloc(sizeof(double) * num);
  gpu_im = (double *)malloc(sizeof(double) * num);
  cpu_re = (double *)malloc(sizeof(double) * num);
  cpu_im = (double *)malloc(sizeof(double) * num);

  for(i=0; i<num; i++){
  	fscanf(fp, "%lf", &gpu_re[i]);
  	cpu_re[i] = gpu_re[i];
    gpu_im[i] = 0;
    cpu_im[i] = 0;
  }

  printf("\nthe number of elements : %d\n", num);
  printf("the name of dataset    : %s\n\n", filename);

  create_timer();

  cpu_function(cpu_re, cpu_im, num, DFT, cpu_dft);
  gpu_function(gpu_re, gpu_im, num, DFT, gpu_dft);
  dft_res = check(cpu_re, gpu_re, num);
  cpu_function(cpu_re, cpu_im, num, IDFT, cpu_idft);
  gpu_function(gpu_re, gpu_im, num, IDFT, gpu_idft);
  idft_res = check(cpu_re, gpu_re, num);

  print_time(cpu_dft, cpu_idft, gpu_dft, gpu_idft);
  print_result(dft_res, idft_res);

  destroy_timer();
  fclose(fp);
  free(gpu_re);
  free(gpu_im);
  free(cpu_re);
  free(cpu_im);
  return 0;
}

