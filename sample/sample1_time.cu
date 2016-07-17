#include <stdio.h>
#include <stdlib.h>
#include "header.h"             // 定数の定義，プロトタイプ宣言などが記述されている

int main(int argc, char **argv) {
  float *h_A, *h_B, *h_C;       // ホスト側の変数
  float *d_A, *d_B, *d_C;       // デバイス側の変数
  float result;                 // 計算結果
  dim3 dim_grid(LENGTH/BLOCK_SIZE, 1, 1), dim_block(BLOCK_SIZE, 1, 1); // Step 4で使用
  cudaEvent_t start, stop;
  float elapsed_time;

  // Step 0. タイマーを作る
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Step 1. ホスト側で問題データを生成する
  h_A = (float *)malloc(sizeof(float) * LENGTH);
  h_B = (float *)malloc(sizeof(float) * LENGTH);
  h_C = (float *)malloc(sizeof(float) * LENGTH);
  for (int i = 0; i < LENGTH; ++i) {
    h_A[i] = 1.0f; h_B[i] = 2.0f; h_C[i] = 0.0f;
  }

  // Step 1.1. ホスト側で計算
  cudaEventRecord(start, 0);
  Sample1Host(h_A, h_B, h_C, LENGTH);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  result = 0.0f;
  for (int i = 0; i < LENGTH; ++i) result += h_C[i];
  result /= (float)LENGTH;
  printf("CPU: result = %f, time = %f [msec]\n", result, elapsed_time);

  // Step 2. デバイス側で必要な分だけメモリ領域を確保する
  cudaMalloc((void **)&d_A, sizeof(float) * LENGTH);
  cudaMalloc((void **)&d_B, sizeof(float) * LENGTH);
  cudaMalloc((void **)&d_C, sizeof(float) * LENGTH);

  // Step 3. ホストからデバイスへ問題データをコピーする
  cudaMemcpy(d_A, h_A, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);

  // Step 4. ホストがカーネルを呼び出す
  cudaEventRecord(start, 0);
  Sample1Kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Step 5. デバイスが処理を終えたら，デバイスからホストへデータをコピーする
  cudaMemcpy(h_C, d_C, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);

  // Step 6. デバイス側で確保していたメモリ領域を解放する
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  // Step 7. デバイスが計算した結果をホストが処理する
  result = 0.0f;
  for (int i = 0; i < LENGTH; ++i) result += h_C[i];
  result /= (float)LENGTH;
  printf("GPU: result = %f, time = %f [msec]\n", result, elapsed_time);

  // Step 8. ホスト側のメモリ解放などの後処理をする
  free(h_A); free(h_B); free(h_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
