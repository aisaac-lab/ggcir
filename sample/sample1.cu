#include <stdio.h>
#include <stdlib.h>
#include "header.h"             // 定数の定義，プロトタイプ宣言などが記述されている

int main(int argc, char **argv) {
  float *h_A, *h_B, *h_C;       // ホスト側の変数
  float *d_A, *d_B, *d_C;       // デバイス側の変数
  float result = 0.0f;          // 計算結果
  dim3 dim_grid(LENGTH/BLOCK_SIZE, 1, 1), dim_block(BLOCK_SIZE, 1, 1); // Step 4で使用

  // Step 1. ホスト側で問題データを生成する
  h_A = (float *)malloc(sizeof(float) * LENGTH);
  h_B = (float *)malloc(sizeof(float) * LENGTH);
  h_C = (float *)malloc(sizeof(float) * LENGTH);
  for (int i = 0; i < LENGTH; ++i) {
    h_A[i] = 1.0f; h_B[i] = 2.0f; h_C[i] = 0.0f;
  }

  // Step 2. デバイス側で必要な分だけメモリ領域を確保する
  cudaMalloc((void **)&d_A, sizeof(float) * LENGTH);
  cudaMalloc((void **)&d_B, sizeof(float) * LENGTH);
  cudaMalloc((void **)&d_C, sizeof(float) * LENGTH);

  // Step 3. ホストからデバイスへ問題データをコピーする
  cudaMemcpy(d_A, h_A, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);

  // Step 4. ホストがカーネルを呼び出す
  Sample1Kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C);

  // Step 5. デバイスが処理を終えたら，デバイスからホストへデータをコピーする
  cudaMemcpy(h_C, d_C, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);

  // Step 6. デバイス側で確保していたメモリ領域を解放する
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  // Step 7. デバイスが計算した結果をホストが処理する
  for (int i = 0; i < LENGTH; ++i) result += h_C[i];
  result /= (float)LENGTH;
  printf("result = %f\n", result);

  // Step 8. ホスト側のメモリ解放などの後処理をする
  free(h_A); free(h_B); free(h_C);

  return 0;
}
