#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void resize_kernel(unsigned char* input, unsigned char* output, int width, int height, int new_width, int new_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height) {
        float x_ratio = ((float)(width - 1)) / new_width;
        float y_ratio = ((float)(height - 1)) / new_height;
        float x_lerp = x * x_ratio;
        float y_lerp = y * y_ratio;
        int x_int = (int)x_lerp;
        int y_int = (int)y_lerp;
        float x_frac = x_lerp - x_int;
        float y_frac = y_lerp - y_int;
        int index = (y_int * width + x_int) * 3;

        for (int c = 0; c < 3; c++) {
            unsigned char p1 = input[index + c];
            unsigned char p2 = input[index + 3 + c];
            unsigned char p3 = input[index + width * 3 + c];
            unsigned char p4 = input[index + width * 3 + 3 + c];

            output[(y * new_width + x) * 3 + c] = (unsigned char)(p1 * (1 - x_frac) * (1 - y_frac) + p2 * x_frac * (1 - y_frac) + p3 * (1 - x_frac) * y_frac + p4 * x_frac * y_frac);
        }
    }
}

void resize_image(unsigned char* input, unsigned char* output, int width, int height, int new_width, int new_height) {
    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, width * height * 3);
    cudaMalloc(&d_output, new_width * new_height * 3);

    cudaMemcpy(d_input, input, width * height * 3, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((new_width + block.x - 1) / block.x, (new_height + block.y - 1) / block.y);

    resize_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height);

    cudaMemcpy(output, d_output, new_width * new_height * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat image = cv::imread("/public_data1/spectraldata/water_skin_rgb/20220623144118297.png");
    int width = image.cols;
    int height = image.rows;
    int new_width = width * 3;
    int new_height = height * 3;

    unsigned char* input_image = image.data;
    unsigned char* output_image = new unsigned char[new_width * new_height * 3];

    resize_image(input_image, output_image, width, height, new_width, new_height);

    cv::Mat resized_image(new_height, new_width, CV_8UC3, output_image);
    cv::imwrite("resized_image.jpg", resized_image);

    delete[] output_image;

    return 0;
}
