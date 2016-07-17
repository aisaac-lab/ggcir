__global__ void Sample1Kernel(float *d_A, float *d_B, float *d_C) {
  // Step 1. 自身のCUDAスレッドIDを計算する
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // Step 2. CUDAスレッドIDを用いてグローバルメモリからデータを読み込み，計算する
  d_C[thread_id] = d_A[thread_id] + d_B[thread_id];
}
