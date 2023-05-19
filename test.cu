#include <iostream>
#include <dirent.h>
#include <cuda_runtime.h>
#define CHECK(call) \
do \
{ \
 const cudaError_t error_code = call; \
 if (error_code != cudaSuccess) \
 { \
 printf("CUDA Error:\\n"); \
 printf(" File: %s\\n", __FILE__); \
 printf(" Line: %d\\n", __LINE__); \
 printf(" Error code: %d\\n", error_code); \
 printf(" Error text: %s\\n", cudaGetErrorString(error_code)); \
 exit(1); \
 } \
} while (0)

int main() {
    // 输入数据
    const int lines = 1020;     // 行数
    const int samples = 1020;   // 列数
    const int bands = 18;      // 波段数
    const int new_lines = lines * 3;     // 行数
    const int new_samples = samples * 3;   // 列数
	float *d_input;
    CHECK(cudaMalloc(&d_input, new_lines * sizeof(float)));
}