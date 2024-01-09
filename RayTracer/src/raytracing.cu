#include "raytracing.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>

#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
#define CHECK(result) cuda_assert(result, __FILE__, __LINE__);
	inline void cuda_assert(cudaError_t result, const char* file, int line)
	{
		if (result)
		{
			std::cout << cudaGetErrorString(result) << "" << file << "" << line << '\n';
		}
	}

	__global__ void renderRayKernel(float* buffer, int width, int height)
	{
		int totalPixels{ width * height };

		int strideX = blockDim.x * gridDim.x;
		int strideY = blockDim.y * gridDim.y;
		int totalStride = strideX * strideY;

		int threadX = threadIdx.x + blockIdx.x * blockDim.x;
		int threadY = threadIdx.y + blockIdx.y * blockDim.y;

		for (int pixel = threadX + threadY * strideX; pixel < totalPixels; pixel += totalStride)
		{
			int i = pixel / height;
			int j = pixel % height;

			buffer[3 * pixel] = float(i) / width;
			buffer[3 * pixel + 1] = float(j) / height;
			buffer[3 * pixel + 2] = 0.2;
		}
	}

	std::vector<float> rtw::renderGPU(int width, int height, int channels)
	{
		// Want number of threads per block to be multiple of 32
		constexpr int threadsX{ 8 };
		constexpr int threadsY{ 8 };

		size_t bufferSize = width * height * channels;
		float* buffer{};

		CHECK(cudaMallocManaged(&buffer, bufferSize * sizeof(float)));

		dim3 blockDim(width / threadsX + 1, height / threadsY + 1);
		dim3 threadDim(threadsX, threadsY);

		renderRayKernel << <blockDim, threadDim >> > (buffer, width, height);

		CHECK(cudaGetLastError());

		CHECK(cudaDeviceSynchronize());

		std::vector<float> data{};
		data.assign(buffer, buffer + bufferSize);

		CHECK(cudaFree(buffer));

		return data;
	}
}
