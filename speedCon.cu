#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__shared__ int sharedArray[64];

__global__ void add(int *a, const int *b){
    // const u32 tid =(blockIdx.x* blockDim.x)+ threadIdx.x;
    int i = blockIdx.x;
    a[i] += b[i];
    sharedArray[i] = 2 * a[i];
    if (i == 1)
    {
        printf("Value of shared_var: %d\n", sharedArray[i]);
    }
}

__global__ void add2(int *a, const int *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 0 3 
    a[i] += b[i]; // 3 + 2 = 5
    // sharedArray[i] = 3 * a[i];
    // if (i == 1)
    // {
    //     printf("Value of shared_var: %d\n", sharedArray[i]);
    // }
}

int main(){
     const int N = 64; // number of elements
         int *a, *b, *temp, i;
       // malloc HOST memory for temp
       temp = new int [N];
           // malloc DEVICE memory for a, b
        cudaMalloc(&a, N*sizeof(int));
        cudaMalloc(&b, N*sizeof(int));
            // set a's values: a[i] = i
            for(i=0;i<N;i++) temp[i] = i;
        cudaMemcpy(a, temp, N*sizeof(int), cudaMemcpyHostToDevice);
             // set b's values: b[i] = 2*i
             for(i=0;i<N;i++) temp[i] = 2*i;
        cudaMemcpy(b, temp, N*sizeof(int), cudaMemcpyHostToDevice);
            // calculate a[i] += b[i] in GPU
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
        // add<<<N,1>>>(a, b); // 已经修改a了！！
        // cudaEventRecord(stop);
        float elapsedTime;
        cudaFuncAttributes attr;
        cudaError_t err = cudaFuncGetAttributes(&attr, add);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get kernel attributes (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printf("add Shared memory size: %zu bytes\n", attr.sharedSizeBytes);

        err = cudaFuncGetAttributes(&attr, add2);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get kernel attributes (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printf("add2 Shared memory size: %zu bytes\n", attr.sharedSizeBytes);

        // cudaEventElapsedTime(&elapsedTime,start,stop);

        // printf("Time to generate: %.6f ms\n", elapsedTime);
        // cudaEventRecord(start);
        // add2<<<2,32>>>(a, b);
        // cudaEventRecord(stop);
        // cudaEventElapsedTime(&elapsedTime,start,stop);
        printf("Time to generate: %.6f ms\n", elapsedTime);
        // show a's values
        cudaMemcpy(temp, a, N*sizeof(int), cudaMemcpyDeviceToHost);
            //for(i=0;i<N;i++){
            // cout << temp[i] << endl;
            // }
        // free HOST & DEVICE memory
        delete [] temp;
        cudaFree(a);
        cudaFree(b);
}

