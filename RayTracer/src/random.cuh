#pragma once

#include <device_launch_parameters.h>

#include <random>
#include <curand_kernel.h>

namespace rtw
{
	// can be used on host and device and allows better state locality
	__host__ __device__ inline float randomFloat(uint32_t& state)
	{
		// PCG Hash
		state = state * 747796495u + 2891336453u;
		state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		state = (state >> 22u) ^ state;

		// scale the output between 0 and 1
		float scale = static_cast<float>(0xffffffffu);
		return static_cast<float>(state) / scale;
	}

	__host__ inline float randomFloatMT()
	{
		static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		static std::mt19937 generator(5489); // 5489 traditional default

		return distribution(generator);
	}
}