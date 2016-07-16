/**************************************/
/* You don't have to change this code */
/* Please modify the "gpu_calc.cu"    */
/**************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415926535897
#define N 1024

void data_1()
{
  int i;
  FILE *fp;
  char *filename = "data_1.txt";
  
  if((fp = fopen(filename, "w")) == NULL){
	fprintf(stderr, "file open failur\n");
	return;
  }
  double tmp;  

  for(i = 0; i < N; i++){
	tmp = sin(i*PI/180);
	fprintf(fp, "%lf\n", tmp);
	printf("%lf\n", tmp);
  }

  fclose(fp);
}

void data_2()
{
  int i;
  FILE *fp;
  char *filename = "data_2.txt";
  
  if((fp = fopen(filename, "w")) == NULL){
	fprintf(stderr, "file open failur\n");
	return;
  }
  double tmp;  

  for(i = 0; i < N; i++){
	tmp = tan(i*PI/180);
	fprintf(fp, "%lf\n", tmp);
	printf("%lf\n", tmp);
  }

  fclose(fp);
}


void data_3()
{
  double alfa = 8.0 * atan(1.0) / N;
  int i;
  FILE *fp;
  char *filename = "data_3.txt";

  if((fp = fopen(filename, "w")) == NULL){
	fprintf(stderr, "file open failur\n");
	return;
  }
  double tmp;  

  for(i = 0; i < N; i++){
	tmp = 10.0 * cos(i * alfa * 10)
	  +10.0 * cos(i * alfa * 20)
	  +10.0 * sin(i * alfa * 30);
	fprintf(fp, "%lf\n", tmp);
	printf("%lf\n", tmp);
  }

  fclose(fp);
}

int main()
{

  data_1();
  data_2();
  data_3();

  return 0;
}
