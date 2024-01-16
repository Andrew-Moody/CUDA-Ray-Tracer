#pragma once

#include <device_launch_parameters.h>

#include <random>
#include <curand_kernel.h>

namespace rtw
{
	__host__ __device__ inline float randomFloat(curandState* randState)
	{
#ifndef __CUDA_ARCH__

		static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		static std::mt19937 generator(5489); // 5489 traditional default

		return distribution(generator);

#else
		return curand_uniform(randState);
#endif
	}
}