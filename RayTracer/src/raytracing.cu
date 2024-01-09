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


	__device__ float getDiscriminantGPU(const Ray& ray, const Vec3& center, float radius)
	{
		// Essentially an intersection test between a ray and a sphere

		Vec3 orig = ray.origin();
		Vec3 dir = ray.direction();
		Vec3 diff = orig - center;

		float rsqr = radius * radius;

		float a = dot(dir, dir);
		float b = dot(2.0f * dir, diff);
		float c = dot(diff, diff) - rsqr;

		float discr = b * b - (4 * a * c);

		// discr < 0.0f  0 roots, ray did not hit sphere
		// discr == 0.0f 1 root,  ray intersects the sphere tangent to the surface
		// discr > 0.0f  2 roots, ray intersects through the sphere 

		return discr;
	}


	__device__ Vec3 getDiscriminantColorGPU(const Ray& ray, const Vec3& center, float radius)
	{
		// Can create fun patterns
		// 800 x 600 seems a good middle ground
		// too high resolution and it becomes smooth losing the pattern
		// too low resolution it becomes too grainy

		float discr = getDiscriminantGPU(ray, center, radius);

		// Modulate color by the discriminant
		if (discr < 0.0f)
		{
			// Outside sphere
			return Vec3(discr, 0.2f + discr, 0.1f + discr);
		}
		else if (discr > 0.0f)
		{
			// Inside sphere
			return Vec3(0.0f, discr, 0.0f);
		}

		// Sphere edge color
		return Vec3(0.0f, 0.0f, 1.0f);

		// Original pattern
		/*if (discr < 0.0f)
			return Vec3(discr, 0.0f, 0.0f);
		else if (discr > 0.0f)
			return Vec3(0.0f, discr, 0.0f);
		return Vec3(0.0f, 0.0f, 1.0f);*/
	}



	__device__ Vec3 getColor(float x, float y)
	{
		return Vec3(x, y, 0.2f);
	}


	__device__ float getHitPointGPU(const Ray& ray, const Vec3& center, float radius)
	{
		// Return the distance along the ray where a hit occured
		// Slightly simplified by using half b to reduce the factor of 2
		Vec3 orig = ray.origin();
		Vec3 dir = ray.direction();
		Vec3 diff = orig - center;

		float rsqr = radius * radius;

		float a = dot(dir, dir);
		float half_b = dot(dir, diff);
		float c = diff.length_squared() - rsqr;

		float discr = half_b * half_b - (a * c);

		if (discr < 0.0f)
		{
			return -1.0f;
		}

		// extra work if discr is == 0 but that is rare
		// for now assuming the smallest value is the closest to the origin
		return (-half_b - std::sqrt(discr)) / a;
	}

	__device__ Vec3 getNormalColorGPU(const Ray& ray, const Vec3& center, float radius)
	{
		float t = getHitPointGPU(ray, center, radius);

		if (t < 0.0f)
		{
			return Vec3(0.9f, 0.9f, 1.0f);
		}

		Vec3 hit = ray.at(t);

		Vec3 normal = (hit - center) * (1.0f / radius);
		// Remap (-1, 1) to (0, 1)
		Vec3 color = 0.5f * normal + Vec3(0.5f, 0.5f, 0.5f);

		return color;
	}

	__global__ void renderRayKernel(float* buffer, int width, int height, float viewWidth, float viewHeight, Vec3 viewOrigin, Vec3 camPos, Vec3 spherePos, float radius)
	{
		int totalPixels{ width * height };
		int strideX = blockDim.x * gridDim.x;

		for (int pixel = threadIdx.x + blockIdx.x * blockDim.x; pixel < totalPixels; pixel += strideX)
		{
			// pixel position in normalized screen coordinates
			float px = static_cast<float>(pixel % width) / width;
			float py = static_cast<float>(pixel / width) / height;

			// Take care that world y decreases as screen y increases
			Vec3 pos{ px * viewWidth, -py * viewHeight, 0.0f };
			pos += viewOrigin;

			Ray ray{camPos, pos - camPos};

			//Vec3 color{ pos };

			Vec3 color = getNormalColorGPU(ray, spherePos, radius);

			//Vec3 color = getColor(px, py);

			//Vec3 color = getDiscriminantColorGPU(ray, spherePos, radius);

			int index = 3 * pixel;
			buffer[index] = color.x();
			buffer[index + 1] = color.y();
			buffer[index + 2] = color.z();
		}
	}

	std::vector<float> rtw::renderGPU(int width, int height, int channels)
	{
		const float aspectRatio{ static_cast<float>(width) / height };

		// Size of the viewport (unclear what units, physical?)
		const float viewHeight{ 2.0f };
		const float viewWidth{ viewHeight * aspectRatio };

		// Distance between pixels in viewport units (ideally should be equal)
		const float deltaU{ viewWidth / width };
		const float deltaV{ viewHeight / height };

		// Camera Properties
		Vec3 cameraPos{ 0.0f, 0.0f, 0.0f };
		Vec3 cameraForward{ 0.0f, 0.0f, -1.0f }; // Camera points in negative z direction
		Vec3 cameraUp{ 0.0f, 1.0f, 0.0f };
		Vec3 cameraRight{ 1.0f, 0.0f, 0.0f };
		float focalLength{ 1.0f }; // Distance from Camera to viewport

		// Sphere
		Vec3 center{ 0.5f, 0.5f, -3.0f };
		float radius = 1.0f;

		Vec3 viewCenter(cameraForward * focalLength); // Center of the screen in worldspace
		Vec3 viewOrigin = viewCenter + 0.5f * ((cameraUp * viewHeight) - (cameraRight * viewWidth)); // Topleft of screen in worldspace
		Vec3 pixelOrigin = viewOrigin + 0.5f * Vec3(deltaU, -deltaV, 0.0f); // Offset viewOrigin to the center of the first pixel


		

		// Want number of threads per block to be multiple of 32
		// will have to experiment with optimal amount
		constexpr int threadsPerBlock{ 64 };

		// will want to clamp to max num of blocks for high resolutions
		const int blocks{ (width * height) / threadsPerBlock + 1 };

		float* buffer{};
		const size_t bufferSize{ static_cast<size_t>(width * height * channels) };
		CHECK(cudaMallocManaged(&buffer, bufferSize * sizeof(float)));

		//renderRayKernel(buffer, width, height, viewWidth, viewHeight, pixelOrigin, cameraPos, center, radius);

		renderRayKernel << <blocks, threadsPerBlock >> > (buffer, width, height, viewWidth, viewHeight, pixelOrigin, cameraPos, center, radius);

		CHECK(cudaGetLastError());

		CHECK(cudaDeviceSynchronize());

		std::vector<float> data{};
		data.assign(buffer, buffer + bufferSize);

		CHECK(cudaFree(buffer));

		return data;
	}
}
