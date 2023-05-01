#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;

__global__ void add(int *a, const int *b)
{
	int i = blockIdx.x;
	a[i] += 2 * b[i];
}

int main()
{
	const int N = 10; // number of elements
	int *a, *b, *temp, i;
	// malloc HOST memory for temp
	temp = new int[N];
	// malloc DEVICE memory for a, b
	cudaMalloc(&a, N * sizeof(int));
	cudaMalloc(&b, N * sizeof(int));
	// set a's values: a[i] = i
	for (i = 0; i < N; i++)
		temp[i] = i;
	cudaMemcpy(a, temp, N * sizeof(int), cudaMemcpyHostToDevice);
	// set b's values: b[i] = 2*i
	for (i = 0; i < N; i++)
		temp[i] = 2 * i;
	cudaMemcpy(b, temp, N * sizeof(int), cudaMemcpyHostToDevice);
	// calculate a[i] += b[i] in GPU
	add<<<N, 1>>>(a, b);
	// show a's values
	cudaMemcpy(temp, a, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (i = 0; i < N; i++)
	{
		cout << temp[i] << endl;
	}
	// free HOST & DEVICE memory
	delete[] temp;
	cudaFree(a);
	cudaFree(b);

	cv::Mat image = cv::imread("/public_data1/spectraldata/water_skin_rgb/20220623144118297.png");
	if (image.empty())
    {
        std::cerr << "无法读取图像文件" << std::endl;
        return false;
    }
    // 修改通道顺序为BGR
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // 检查图像是否正确读取

    // 保存结果图像
    cv::imwrite("./result.jpg", image);
}
