#pragma once

#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	struct HitResult
	{
		Vec3 point;
		Vec3 normal;
		float t{};

		__host__ __device__ Vec3 getNormalColor() { return 0.5f * normal + Vec3(0.5f, 0.5f, 0.5f); }
	};

	class Hittable
	{
	public:

		__host__ __device__ virtual bool hit(Ray ray, HitResult& outResult) const = 0;

		__host__ __device__ virtual ~Hittable() = 0;
	};

	__host__ __device__ inline Hittable::~Hittable() {}
}