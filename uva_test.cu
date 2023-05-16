#include "cuda_runtime.h"
#include <stdio.h>

__global__ void increment_kernel(int *data, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}
__global__ void kernel(float *data) {
    int idx = threadIdx.x;
    data[idx] = idx;
}
int main() {
    // int n = 10;
    // size_t size = n * sizeof(int);

    // // 分配UVA内存
    // int *data;
    // cudaHostAlloc(&data, size, cudaHostAllocMapped);

    // // 初始化数据
    // for (int i = 0; i < n; i++) {
    //     data[i] = i;
    // }

    // // 调用核函数
    // dim3 block(32);
    // dim3 grid((n + block.x - 1) / block.x);
    // increment_kernel<<<grid, block>>>(data, n);

    // // 等待设备完成
    // cudaDeviceSynchronize();

    // // 打印结果
    // for (int i = 0; i < n; i++) {
    //     printf("%d ", data[i]);
    // }
    // printf("\n");

    // // 释放内存
    // cudaFreeHost(data);

    // float *a_h, *b_h; // host data
    // float *a_d; // device data
    // int N = 14;
    // uint4 *u4;
    // // register u8 r = EXTRACT(N);
    // size_t size = N * sizeof(float);

    // // allocate memory on host
    // cudaHostAlloc((void **) &a_h, size, cudaHostAllocMapped);
    // cudaHostAlloc((void **) &b_h, size, cudaHostAllocMapped);

    // // get device pointer
    // cudaHostGetDevicePointer((void **) &a_d, (void *) a_h, 0);

    // // initialize data on host
    // for (int i = 0; i < N; i++) {
    //     a_h[i] = 10.0f + i;
    //     b_h[i] = 0.0f;
    // }

    // // copy data from host to device
    // cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    // // copy data from device to host
    // cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost);

    // // print result
    // for (int i = 0; i < N; i++) {
    //     printf("%d %g %g\n", i, a_h[i], b_h[i]);
    // }

    // // free memory on host
    // cudaFreeHost(a_h);
    // cudaFreeHost(b_h);
    float *h_data;
    float *d_data;
    int N = 10;
    size_t size = N * sizeof(float);

    // 使用cudaHostAllocMapped参数分配内存
    cudaHostAlloc((void**)&h_data, size, cudaHostAllocMapped);
    // 获取设备指针
    cudaHostGetDevicePointer((void**)&d_data, (void*)h_data, 0);

    kernel<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    cudaFreeHost(h_data);

    return 0;
}
