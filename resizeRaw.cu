#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include<opencv2/imgproc/types_c.h> 
// #include<opencv2/opencv.hpp>

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

void resize3D(float* input, float* output, int width, int height, int depth, int newWidth, int newHeight, int newDepth){
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, width * height * depth * sizeof(float));
    cudaMalloc(&d_output, newWidth * newHeight * newDepth * sizeof(float));
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录 cudaMemcpy 的开始时间
    cudaEventRecord(start, 0);
    cudaMemcpy(d_input, input, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    // 记录 cudaMemcpy 的结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算 cudaMemcpy 的执行时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to transfer data from host to device: " << elapsedTime << " ms" << std::endl;

    dim3 blockSize(16, 16, 4);
    dim3 gridSize((newWidth + blockSize.x - 1) / blockSize.x,
                  (newHeight + blockSize.y - 1) / blockSize.y,
                  (newDepth + blockSize.z - 1) / blockSize.z);
    cudaEventRecord(start, 0);
    trilinearInterpolation<<<gridSize, blockSize>>>(d_input, d_output, width, height, depth, newWidth, newHeight, newDepth);
    // 记录核函数的结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算核函数的执行时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to execute kernel: " << elapsedTime << " ms" << std::endl;

    cudaEventRecord(start, 0);
    cudaMemcpy(output, d_output, newWidth * newHeight * newDepth * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算核函数的执行时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to transfer data fro device to host:" << elapsedTime << " ms" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 输入数据
    const int lines = 1020;     // 行数
    const int samples = 1020;   // 列数
    const int bands = 18;      // 波段数
    const int new_lines = lines * 3;     // 行数
    const int new_samples = samples * 3;   // 列数
    FILE* fp;
    fp = fopen("/public_data1/dataset_18ch/raw_data/20211021150620.raw", "rb");
    const int size = lines*samples*bands;
    float* data = new float[size];
    const int new_size = new_lines*new_samples*bands;
    float* output_data = new float[new_size];
    fread(data, sizeof(float), size, fp);
    resize3D(data, output_data, samples,lines, bands, new_samples, new_lines, bands);
    cv::Mat rgb = getRgb(output_data);
    cv::imwrite("resized_image_raw.jpg", rgb);
    delete[] data;
    delete[] output_data;
}