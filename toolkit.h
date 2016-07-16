#ifndef __TOOLKIT_H_INCLUDED__
#define __TOOLKIT_H_INCLUDED__

#include "timer.h"

void initialize(int &num, char *filename, int argc, char **argv);
void print_time(wtime_t cpu_dft, wtime_t cpu_idft, wtime_t gpu_dft, wtime_t gpu_idft);
void print_result(double dft_res, double idft_res);
double check(double *cpu_data, double *gpu_data, int num);

#endif /* __TOOLKIT_H_INCLUDED__ */
