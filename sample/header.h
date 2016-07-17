const int LENGTH = 1024 * 1024;
const int LENGTH_X = 1024;
const int LENGTH_Y = 1024;
const int BLOCK_SIZE = 256;

// For sample1
__global__ void Sample1Kernel(float *d_A, float *d_B, float *d_C);
__host__ void Sample1Host(float *h_A, float *h_B, float *h_C, int length);
