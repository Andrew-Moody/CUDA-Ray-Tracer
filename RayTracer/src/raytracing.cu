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

		curandState randState;
		curand_init(1234, 0, 0, &randState);

		scene.initializeScene(camera, &randState);
	}

	__global__ void initRandomKernel(curandState* state, int stateSize, int seed)
	{
		int thread = threadIdx.x + blockIdx.x * blockDim.x;
		if (thread >= stateSize)
		{
			return;
		}

		// set same seed difference sequence for each thread
		curand_init(seed, thread, 0, &state[thread]);
	}

	__global__ void renderRayKernel(float* buffer, Camera camera, Scene scene, curandState* randStates)
	{
		int strideX = blockDim.x * gridDim.x;
		int thread = threadIdx.x + blockIdx.x * blockDim.x;
		curandState* randState = &randStates[thread];

		for (int pixel = thread; pixel < camera.totalPixels(); pixel += strideX)
		{
			Vec3 worldPos = camera.getWorldPosFromPixelIndex(pixel);

			Vec3 color{};

			for (int i = 0; i < camera.nSamples(); ++i)
			{
				// sample defocus disk
				Vec3 rayOrigin = camera.getSampleOrigin(randState);

				// Sample a random location inside the square halfway between the pixel position
				// and the positions of neighboring pixels
				Vec3 samplePos = camera.shiftWorldPos(worldPos, randomFloat(randState) - 0.5f, randomFloat(randState) - 0.5f);

				Ray ray{ rayOrigin, samplePos - rayOrigin };

				color += scene.getColor(ray, camera.nBounces(), randState);
			}

			color *= 1.0f / camera.nSamples();


			/*Ray ray = camera.getRayFromPixelIndex(pixel);

			Vec3 color{ scene.getColor(ray, &randStates[thread]) };*/

			// Write color to buffer
			int index = 3 * pixel;
			buffer[index] = color.x();
			buffer[index + 1] = color.y();
			buffer[index + 2] = color.z();
		}
	}

	std::vector<float> rtw::renderGPU(const Camera& camera, int numSpheres)
	{		
		// Want number of threads per block to be multiple of 32
		// will have to experiment with optimal amount
		// also will want to clamp the max num of blocks for high resolutions
		constexpr int threadsPerBlock{ 1024 };
		const int blocks{ camera.totalPixels() / threadsPerBlock + 1 };

		// Allocate memory for the frame buffer
		float* buffer{};
		const size_t bufferSize{ static_cast<size_t>(camera.totalPixels() * camera.nChannels()) };
		CHECK(cudaMallocManaged(&buffer, bufferSize * sizeof(float)));

		// Allocate memory and initialize scene on device
		Sphere* spheres{};
		CHECK(cudaMalloc(&spheres, numSpheres * sizeof(Sphere)));
		Scene scene{ spheres, spheres + numSpheres };
		initializeSceneKernel << <1, 1 >> > (scene, camera);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		
		// Allocate memory and initialize rand states for each thread
		const int threadCount = threadsPerBlock * blocks;
		curandState* randStates{};
		CHECK(cudaMalloc(&randStates, threadCount * sizeof(curandState)));
		initRandomKernel << <blocks, threadsPerBlock >> > (randStates, threadCount, 1337);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		// Render the frame
		renderRayKernel << <blocks, threadsPerBlock >> > (buffer, camera, scene, randStates);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		// Copy data from managed memory to normal memory
		std::vector<float> data{};
		data.assign(buffer, buffer + bufferSize);

		// Release device memory
		CHECK(cudaFree(spheres));
		CHECK(cudaFree(randStates));
		CHECK(cudaFree(buffer));

		return data;
	}
}
