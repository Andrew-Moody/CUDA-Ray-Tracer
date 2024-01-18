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

	__global__ void renderRayKernel(float* buffer, Camera camera, Scene scene)
	{
		int strideX = blockDim.x * gridDim.x;
		int thread = threadIdx.x + blockIdx.x * blockDim.x;

		for (int pixel = thread; pixel < camera.totalPixels(); pixel += strideX)
		{
			uint32_t pcgState = pixel;

			Vec3 worldPos = camera.getWorldPosFromPixelIndex(pixel);

			Vec3 color{};

			const int nSamples = camera.nSamples();

			for (int i = 0; i < nSamples; ++i)
			{
				// Create a ray from a random point on DOF disk to random point in the pixel
				Ray ray = camera.generateRay(worldPos, pcgState);

				//color += scene.getColor(ray, camera.nBounces(), pcgState);
				
				color += ray.direction();
			}

			//color *= 1.0f / camera.nSamples();

			// Seems to use 14 registers on its own vectorization can sometimes help
			// but might not when register bound
			//// Write color to buffer
			int index = 3 * pixel;
			buffer[index] = color.x();
			buffer[index + 1] = color.y();
			buffer[index + 2] = color.z();
		}
	}

	struct Transform
	{
		float x, y, z, r;
		int index;
	};

	std::vector<float> rtw::renderGPU(const Camera& camera, const Scene& scene)
	{	
		cudaDeviceProp prop;
		CHECK(cudaGetDeviceProperties(&prop, 0));

		// total number of threads that can theoretically be resident
		// 10 mp * 2048 threads per mp
		const int maxThreadsTotal = 10 * 2048;
		//const int maxThreadsPerBlock = 1024;
		//const int maxBlocksTotal = 10 * 32;

		// want to pick multiples of 32 to maximize warp usage
		const int threadsPerBlock{ 256 };
		const int blocks{ maxThreadsTotal / threadsPerBlock };


		// Allocate memory for the frame buffer
		float* buffer{};
		const size_t bufferSize{ static_cast<size_t>(camera.totalPixels() * camera.nChannels()) };
		CHECK(cudaMallocManaged(&buffer, bufferSize * sizeof(float)));


		// Copy spheres and colliders to device and wrap in a scene
		Sphere* spheres{};
		Collider* colliders{};
		int numSpheres = scene.getSphereCount();
		CHECK(cudaMalloc(&spheres, numSpheres * sizeof(Sphere)));
		CHECK(cudaMalloc(&colliders, numSpheres * sizeof(Collider)));
		CHECK(cudaMemcpy(spheres, scene.getSpheres(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(colliders, scene.getColliders(), numSpheres * sizeof(Collider), cudaMemcpyHostToDevice));
		CHECK(cudaDeviceSynchronize());
		Scene sceneDevice{ spheres, colliders, numSpheres };


		// Render the frame
		renderRayKernel<<<blocks, threadsPerBlock>>>(buffer, camera, sceneDevice);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		// Copy data from managed memory to normal memory
		std::vector<float> data{};
		data.assign(buffer, buffer + bufferSize);

		// do this off kernel to remove some register pressure
		for (auto & value : data)
		{
			value /= camera.nSamples();
		}

		// Release device memory
		CHECK(cudaFree(spheres));
		CHECK(cudaFree(colliders));
		CHECK(cudaFree(buffer));

		return data;
	}
}
