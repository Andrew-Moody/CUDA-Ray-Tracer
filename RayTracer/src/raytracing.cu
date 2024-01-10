#include "raytracing.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>

#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"

#include "hittable.cuh"
#include "sphere.cuh"

#include "scene.cuh"

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

	__global__ void initializeSceneKernel(Scene scene, Camera camera)
	{
		// Only execute once
		if (threadIdx.x != 0 || blockIdx.x != 0 || threadIdx.y != 0 || blockIdx.y != 0)
		{
			return;
		}

		scene.initializeScene(camera);
	}

	__global__ void renderRayKernel(float* buffer, Camera camera, Scene scene)
	{
		int strideX = blockDim.x * gridDim.x;

		for (int pixel = threadIdx.x + blockIdx.x * blockDim.x; pixel < camera.totalPixels(); pixel += strideX)
		{
			Ray ray = camera.getRayFromPixelIndex(pixel);

			Vec3 color{ scene.getColor(ray) };

			// Write color to buffer
			int index = 3 * pixel;
			buffer[index] = color.x();
			buffer[index + 1] = color.y();
			buffer[index + 2] = color.z();
		}
	}

	std::vector<float> rtw::renderGPU(int width, int height, int channels, int numSpheres)
	{
		// Camera
		const Camera camera{ width, height };
		
		// Allocate memory and initialize scene on device
		Sphere* spheres{};
		CHECK(cudaMalloc(&spheres, numSpheres * sizeof(Sphere)));
		Scene scene{ spheres, spheres + numSpheres };
		initializeSceneKernel << <1, 1 >> > (scene, camera);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		
		// Allocate memory for the frame buffer
		float* buffer{};
		const size_t bufferSize{ static_cast<size_t>(width * height * channels) };
		CHECK(cudaMallocManaged(&buffer, bufferSize * sizeof(float)));

		// Want number of threads per block to be multiple of 32
		// will have to experiment with optimal amount
		// also will want to clamp the max num of blocks for high resolutions
		constexpr int threadsPerBlock{ 64 };
		const int blocks{ (width * height) / threadsPerBlock + 1 };

		// Render the frame
		renderRayKernel << <blocks, threadsPerBlock >> > (buffer, camera, scene);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		// Copy data from managed memory to normal memory
		std::vector<float> data{};
		data.assign(buffer, buffer + bufferSize);

		// Release device memory
		CHECK(cudaFree(buffer));
		CHECK(cudaFree(spheres));

		return data;
	}
}
