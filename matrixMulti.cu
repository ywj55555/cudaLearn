#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 512 // 矩阵A的行数
#define K 512 // 矩阵A的列数，矩阵B的行数
#define N 512 // 矩阵B的列数
#define TILE_WIDTH 16 // 分块矩阵的大小

// 初始化矩阵，随机生成0-10之间的浮点数
void initial(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (float)(rand() % 10 + 1);
    }
}

// array_A, array_B, array_C是显存指针，M_p, K_p, N_p是矩阵维度
__global__ void multiplicateMatrixOri(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
    // 这里我们划分的block和grid是二维的，分别计算线程的二维索引（x方向和y方向的索引）
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // row number
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // col number

    if (ix < N_p && iy < M_p) // 筛选线程，每个线程计算C中的一个元素，线程的xy索引与C的元素位置索引对应
    {
        float sum = 0;
        for (int k = 0; k < K_p; k++) // C中的某个元素为A中对应行和B中对应列向量的乘积和。
        {
            sum += array_A[iy * K_p + k] * array_B[k * N_p + ix];
        }
        array_C[iy * N_p + ix] = sum;
    }
}

// 共享矩阵能否优化，我认为看行列是否大于 全局内存的访问长度，如128 / 3 = 32, 然后再加上 读取共享内存的开销，应该需要大于某个阈值，
// 核函数，计算矩阵乘法：C = A * B，使用分块和共享内存优化
// array_A, array_B, array_C是显存指针，M_p, K_p, N_p是矩阵维度
__global__ void multiplicateMatrix(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
    // 这里我们划分的block和grid是二维的，分别计算线程的二维索引（x方向和y方向的索引）
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // row number
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // col number

    // 显式声明共享内存a，b子矩阵块
    __shared__ float shareA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shareB[TILE_WIDTH][TILE_WIDTH];

    float sum = 0;
    for (int m = 0; m < (K_p - 1) / TILE_WIDTH + 1; m++) // 循环遍历所有分块
    {
        // load data from global memory to shared memory
        if (iy < M_p && m * TILE_WIDTH + threadIdx.x < K_p) // 筛选有效线程，避免越界
        {
            shareA[threadIdx.y][threadIdx.x] = array_A[iy * K_p + m * TILE_WIDTH + threadIdx.x]; // 将A矩阵的一行分块拷贝到共享内存中
        }
        else
        {
            shareA[threadIdx.y][threadIdx.x] = 0; // 越界的线程赋值为0，不影响计算结果
        }

        if (ix < N_p && m * TILE_WIDTH + threadIdx.y < K_p) // 筛选有效线程，避免越界
        {
            shareB[threadIdx.y][threadIdx.x] = array_B[(m * TILE_WIDTH + threadIdx.y) * N_p + ix]; // 将B矩阵的一列分块拷贝到共享内存中
        }
        else
        {
            shareB[threadIdx.y][threadIdx.x] = 0; // 越界的线程赋值为0，不影响计算结果
        }

        __syncthreads(); // 需要等所有线程块都拷贝完成后再进行计算

        for (int i = 0; i < TILE_WIDTH; i++) // 计算C中的一个元素为A中对应行和B中对应列向量的乘积和。
        {
            sum += shareA[threadIdx.y][i] * shareB[i][threadIdx.x];
        }

        __syncthreads(); // 需要等所有线程块都计算完成后再进行下一轮循环
    }

    if (ix < N_p && iy < M_p) // 筛选
        {
        array_C[iy * N_p + ix] = sum; // 将计算结果写回C矩阵中
    }
}

// 主函数
int main(int argc, char **argv)
{
    int Axy = M * K; // 矩阵A的元素个数
    int Bxy = K * N; // 矩阵B的元素个数
    int Cxy = M * N; // 矩阵C的元素个数

    float *h_A, *h_B, *h_C; // 在CPU上分配内存

    h_A = (float *)malloc(Axy * sizeof(float));
    h_B = (float *)malloc(Bxy * sizeof(float));
    h_C = (float *)malloc(Cxy * sizeof(float));

    initial(h_A, Axy); // 初始化矩阵A
    initial(h_B, Bxy); // 初始化矩阵B

    float *d_A, *d_B, *d_C; // 在GPU上分配显存

    cudaMalloc((void **)&d_A, Axy * sizeof(float));
    cudaMalloc((void **)&d_B, Bxy * sizeof(float));
    cudaMalloc((void **)&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice); // 将CPU上初始化的A值拷贝到GPU上
    cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice); // 将CPU上初始化的B值拷贝到GPU上

    int dimx = TILE_WIDTH; // 划分block的x方向大小
    int dimy = TILE_WIDTH; // 划分block的y方向大小
    dim3 block(dimx, dimy); // 定义block为二维结构
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y); // 定义grid为二维结构

    multiplicateMatrix<<<grid, block>>>(d_A, d_B, d_C, M, K, N); // 调用核函数

    cudaMemcpy(h_C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost); // 将GPU上的计算结果拷贝回CPU

    cudaFree(d_A); // 释放GPU显存资源
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A); // 释放CPU内存资源
    free(h_B);
    free(h_C);

    return 0;
}
