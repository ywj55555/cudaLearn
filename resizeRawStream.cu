#include <iostream>
#include <dirent.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include<opencv2/imgproc/types_c.h> 
// 来一个宏定义 调用api看看是否成功，！！!
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

// #include<opencv2/opencv.hpp>
#define STREAM_NUM 4
cv::Mat getRgb(float *rawdata){
    int lines = 1020 * 3;
    int samples = 1020 * 3;
    int channels = 18;
    cv::Mat rgb_img,tmp_channel;
//    cv::Mat A(lines, samples, CV_32FC(18), rawdata);
//    std::vector<cv::Mat> spect_Channels(18);
//    cv::split(A, spect_Channels);
    std::vector<cv::Mat> rgb_channels;
//    float *tmp=rawdata;
//    rgb_channels.push_back(spect_Channels[10]);
//    rgb_channels.push_back(spect_Channels[7]);
//    rgb_channels.push_back(spect_Channels[1]);
//    time_t start_time = clock();
    for(int i =0 ;i<lines*samples*channels;i++){

        if(rawdata[i]<0)rawdata[i]=0;
        if(isnan(rawdata[i])){
            rawdata[i]=0;
        }
    }
//    time_t  end_time = clock();
//    std::cout << (end_time - start_time)/CLOCKS_PER_SEC  << std::endl;
    cv::Mat dst = cv::Mat(lines, samples, CV_8UC1);
    tmp_channel = cv::Mat(lines, samples, CV_32FC1,rawdata+1*lines*samples);
    cv::normalize(tmp_channel,dst,0,255,CV_MINMAX,CV_8UC1);
    rgb_channels.push_back(dst);

    dst = cv::Mat(lines, samples, CV_8UC1);
    tmp_channel = cv::Mat(lines, samples, CV_32FC1,rawdata+7*lines*samples);
    cv::normalize(tmp_channel,dst,0,255,CV_MINMAX,CV_8UC1);
    rgb_channels.push_back(dst);

    dst = cv::Mat(lines, samples, CV_8UC1);
    tmp_channel = cv::Mat(lines, samples, CV_32FC1,rawdata+10*lines*samples);
    cv::normalize(tmp_channel,dst,0,255,CV_MINMAX,CV_8UC1);
    rgb_channels.push_back(dst);

    cv::merge(rgb_channels, rgb_img);
    return rgb_img;
}


__global__ void trilinearInterpolation(float *input, float *output, int width, int height, int depth, int newWidth, int newHeight, int newDepth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < newWidth && y < newHeight && z < newDepth) {
        float xRatio = ((float)(width - 1)) / newWidth;
        float yRatio = ((float)(height - 1)) / newHeight;
        float zRatio = ((float)(depth - 1)) / newDepth;

        int x1 = (int)(x * xRatio);
        int y1 = (int)(y * yRatio);
        int z1 = (int)(z * zRatio);

        float xDiff = (x * xRatio) - x1;
        float yDiff = (y * yRatio) - y1;
        float zDiff = (z * zRatio) - z1;

        int index = z1 * width * height + y1 * width + x1;

        float c000 = input[index];
        float c001 = input[index + width * height];
        float c010 = input[index + width];
        float c011 = input[index + width * height + width];
        float c100 = input[index + 1];
        float c101 = input[index + width * height + 1];
        float c110 = input[index + width + 1];
        float c111 = input[index + width * height + width + 1];

        output[z * newWidth * newHeight + y * newWidth + x] =
            c000 * (1 - xDiff) * (1 - yDiff) * (1 - zDiff) +
            c100 * xDiff * (1 - yDiff) * (1 - zDiff) +
            c010 * (1 - xDiff) * yDiff* (1 - zDiff) +
            c001* (1-xDiff)*(1-yDiff)*zDiff+
            c101*xDiff*(1-yDiff)*zDiff+
            c011*(1-xDiff)*yDiff*zDiff+
            c110*xDiff*yDiff*(1-zDiff)+
            c111*xDiff*yDiff*zDiff;
    }
}

/*
template <int sharedSize>
__global__ void trilinearInterpolation(float *input, float *output, int width, int height, int depth, int newWidth, int newHeight, int newDepth) {
    // 定义共享内存数组
    __shared__ float sharedInput[sharedSize];
    __shared__ int beginindex;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int blockWidth = int((blockDim.x + 2 )/ 3);  // 向上取整 4
    int blockheight = int((blockDim.y + 2) / 3); // 4

    if (x < newWidth && y < newHeight && z < newDepth) {
        float xRatio = ((float)(width - 1)) / newWidth;
        float yRatio = ((float)(height - 1)) / newHeight;
        float zRatio = ((float)(depth - 1)) / newDepth;

        int x1 = (int)(x * xRatio);
        int y1 = (int)(y * yRatio);
        int z1 = (int)(z * zRatio);

        float xDiff = (x * xRatio) - x1;
        float yDiff = (y * yRatio) - y1;
        float zDiff = (z * zRatio) - z1;

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            beginindex = z1 * width * height + y1 * width + x1;
        }
        __syncthreads();
        // int index = z1 * width * height + y1 * width + x1;

        // 加载数据到共享内存
        if (threadIdx.x < blockWidth && threadIdx.y < blockheight) {
            // 这里应该加载全局的
            int pos = beginindex + threadIdx.z * width * height +  threadIdx.y * width + threadIdx.x;
            sharedInput[threadIdx.z * blockWidth * blockheight +  threadIdx.y * blockWidth + threadIdx.x] = input[pos];  // 行 列 通道 排布
        }    
        // 同步线程
        __syncthreads();

        // 从共享内存中读取数据
         xRatio = ((float)(blockWidth - 1)) / blockDim.x;
         yRatio = ((float)(blockheight - 1)) / blockDim.y;
         zRatio = ((float)(blockDim.z - 1)) / blockDim.z;

         x1 = (int)(threadIdx.x * xRatio);
         y1 = (int)(threadIdx.y * yRatio);
         z1 = (int)(threadIdx.z * zRatio);

        int index = z1 * blockWidth * blockheight + y1 * blockWidth + x1;
        float c000 = sharedInput[index];
        float c001 = sharedInput[index + blockWidth * blockheight];
        float c010 = sharedInput[index + blockWidth];
        float c011 = sharedInput[index + blockWidth * blockheight + blockWidth];
        float c100 = sharedInput[index + 1];
        float c101 = sharedInput[index + blockWidth * blockheight + 1];
        float c110 = sharedInput[index + blockWidth + 1];
        float c111 = sharedInput[index + blockWidth * blockheight + blockWidth + 1];

        output[z * newWidth * newHeight + y * newWidth + x] =
            c000 * (1 - xDiff) * (1 - yDiff) * (1 - zDiff) +
            c100 * xDiff * (1 - yDiff) * (1 - zDiff) +
            c010 * (1 - xDiff) * yDiff* (1 - zDiff) +
            c001* (1-xDiff)*(1-yDiff)*zDiff+
            c101*xDiff*(1-yDiff)*zDiff+
            c011*(1-xDiff)*yDiff*zDiff+
            c110*xDiff*yDiff*(1-zDiff)+
            c111*xDiff*yDiff*zDiff;
    }
}
*/

void resize3D(std::string dirpath, int width, int height, int depth, int newWidth, int newHeight, int newDepth){
    // 每个流的执行顺序不保证把？？？就是执行引擎不保证执行完毕顺序和提交顺序一致？先实现再说！！
    float* d_input[STREAM_NUM];
    float* d_output[STREAM_NUM];
    float* h_input[STREAM_NUM];
    float* h_output[STREAM_NUM];
    int predictCount = 15;
    int inputSize = width * height * depth;
    int outputsize = newWidth * newHeight * newDepth;
    int nDevices;
    // 查询系统中可用的CUDA设备数量
    cudaGetDeviceCount(&nDevices);
    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++i) {
        CHECK(cudaSetDevice(i % nDevices));
        CHECK(cudaHostAlloc(&h_input[i], inputSize * sizeof(float), cudaHostAllocDefault));  // cudaHostAllocMapped 分配
        // h_input[i] = (float*)malloc(inputSize * sizeof(float));
        CHECK(cudaMalloc(&d_input[i], inputSize * sizeof(float)));
        CHECK(cudaHostAlloc(&h_output[i], outputsize * sizeof(float), cudaHostAllocDefault));
        // h_output[i] = (float*)malloc(outputsize * sizeof(float));
        CHECK(cudaMalloc(&d_output[i], outputsize * sizeof(float)));
        CHECK(cudaStreamCreate(&stream[i]));
    }
    // for (int i = 0; i < STREAM_NUM; ++i) {
    //     cudaHostAlloc(&h_input[i], inputSize * sizeof(float), cudaHostAllocMapped);  // cudaHostAllocMapped 分配
    //     // cudaMalloc(&d_input[i], inputSize * sizeof(float));
    //     cudaHostGetDevicePointer((void**)&d_input[i], (void*)h_input[i], 0);
    //     cudaHostAlloc(&h_output[i], outputsize * sizeof(float), cudaHostAllocMapped);
    //     // cudaMalloc(&d_output[i], outputsize * sizeof(float));
    //     cudaHostGetDevicePointer((void**)&d_output[i], (void*)h_output[i], 0);
    //     cudaStreamCreate(&stream[i]);
    // }
    float elapsedTime;
    cudaEvent_t start, stop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    // 记录 cudaMemcpy 的开始时间
    CHECK(cudaEventRecord(start, 0));  // 绑定在默认流中！！这个和设备相关联把！！多设备如何确定？？

    DIR *dir;
    struct dirent *ent;
    std::string folder = dirpath;
    int count = 0;
    
    if ((dir = opendir(folder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL && count <  STREAM_NUM * 2) {
            std::string filename = ent->d_name;
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".raw") {
                std::string filepath = folder + "/" + filename;
                FILE *file = fopen(filepath.c_str(), "rB");
                if (file == NULL) {
                    std::cout << "Error opening file: " << filepath << std::endl;
                } else {
                    // 得在不同device上进行实现
                    // 或者使用线程池
                    // 两种优化都试试
                    
                    std::cout << "Successfully opened file: " << filepath << std::endl;
                    int streamPos = count % STREAM_NUM;
                    if (count % STREAM_NUM == 0 && count != 0) {
                        for (int i = 0; i < STREAM_NUM; i++) {
                            CHECK(cudaSetDevice(i % nDevices));
                            CHECK(cudaStreamSynchronize(stream[i]));  // 需要访问输出结果，所以需要进行同步
                            cv::Mat rgb = getRgb(h_output[i]);
                            cv::imwrite("resized_image_raw" + std::to_string(count) + std::to_string(i) + ".jpg", rgb);
                        }
                    }
                    // fread之前需要确保之前的数据已经传输完毕了，但是需要用同步来保证把！！！
                    fread(h_input[streamPos], sizeof(float), inputSize, file);
                    CHECK(cudaSetDevice(streamPos % nDevices));
                    CHECK(cudaMemcpyAsync(d_input[streamPos], h_input[streamPos], width * height * depth * sizeof(float), cudaMemcpyHostToDevice, stream[streamPos]));
                    dim3 blockSize(12, 12, 6);
                    dim3 gridSize((newWidth + blockSize.x - 1) / blockSize.x,
                                  (newHeight + blockSize.y - 1) / blockSize.y,
                                  (newDepth + blockSize.z - 1) / blockSize.z);
                    trilinearInterpolation<<<gridSize, blockSize, 0, stream[streamPos]>>>(d_input[streamPos], d_output[streamPos], width, height, depth, newWidth, newHeight, newDepth);
                    CHECK(cudaMemcpyAsync(h_output[streamPos], d_output[streamPos], newWidth * newHeight * newDepth * sizeof(float), cudaMemcpyDeviceToHost, stream[streamPos]));
                    
                    fclose(file);
                    count++;
                }
                
            }
        }
        closedir(dir);
        
    }
    // 最后同步所有的stream 注意在对应的设备上执行！！！
    for (int i = 0; i < STREAM_NUM; i++) {
       CHECK(cudaSetDevice(i % nDevices));
        CHECK(cudaStreamSynchronize(stream[i]));
        cv::Mat rgb = getRgb(h_output[i]);
        cv::imwrite("resized_image_raw" + std::to_string(count) + std::to_string(i) + ".jpg", rgb);
    }
    // cudaMemcpy(d_input, input, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    // 记录 cudaMemcpy 的结束时间
    CHECK(cudaSetDevice(0));  
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    // 计算 cudaMemcpy 的执行时间
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "all time: " << elapsedTime << " ms" << std::endl;
    for (int i = 0; i < STREAM_NUM; i++) {
        CHECK(cudaSetDevice(i % nDevices));
       CHECK(cudaFreeHost(h_input[i]));
        // free(h_input[i]);
        CHECK(cudaFree(d_input[i]));
        CHECK(cudaFreeHost(h_output[i]));
        // free(h_output[i]);
        CHECK(cudaFree(d_output[i]));
        CHECK(cudaStreamDestroy(stream[i]));
    }
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    // // dim3 blockSize(16, 16, 4);
    // dim3 blockSize(12, 12, 6);
    // dim3 gridSize((newWidth + blockSize.x - 1) / blockSize.x,
    //               (newHeight + blockSize.y - 1) / blockSize.y,
    //               (newDepth + blockSize.z - 1) / blockSize.z);
    // cudaEventRecord(start, 0); // int((blockSize.x + 2 / 3) * (blockSize.y + 2 / 3) * blockSize.z)
    // const int sharedsize = 4 * 4 * 6;
    // // 只能使用 extern 进行动态分配了！！
    // trilinearInterpolation<sharedsize><<<gridSize, blockSize>>>(d_input, d_output, width, height, depth, newWidth, newHeight, newDepth);
    // // 记录核函数的结束时间
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // // 计算核函数的执行时间
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "Time to execute kernel: " << elapsedTime << " ms" << std::endl;

    // cudaEventRecord(start, 0);
    // cudaMemcpy(output, d_output, newWidth * newHeight * newDepth * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // // 计算核函数的执行时间
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "Time to transfer data fro device to host:" << elapsedTime << " ms" << std::endl;
    // cudaFree(d_input);
    // cudaFree(d_output);

}

int main() {
    // 输入数据
    const int lines = 1020;     // 行数
    const int samples = 1020;   // 列数
    const int bands = 18;      // 波段数
    const int new_lines = lines * 3;     // 行数
    const int new_samples = samples * 3;   // 列数
    // FILE* fp;
    std::string dirpath = "/public_data1/dataset_18ch/raw_data/";
    // fp = fopen("/public_data1/dataset_18ch/raw_data/20211021150620.raw", "rb");
    // const int size = lines*samples*bands;
    // float* data = new float[size];
    // const int new_size = new_lines*new_samples*bands;
    // float* output_data = new float[new_size];
    // fread(data, sizeof(float), size, fp);
    // 下一步：流水线插值10张？
    resize3D(dirpath, samples,lines, bands, new_samples, new_lines, bands);
    
    // delete[] data;
    // delete[] output_data;
}