#pragma once

#include "hittable.cuh"
#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	class Sphere// : public Hittable
	{
	public:

		__host__ __device__ Sphere() {}

		__host__ __device__ Sphere(float x, float y, float z, float r)
			: center_{ x, y, z }, radius_{ r } {}

		__host__ __device__ Sphere(const Vec3& center, float radius)
			: center_{ center }, radius_{ radius } {}

		__host__ __device__ bool hit(Ray ray, HitResult& outResult) const;

		__host__ __device__ bool hitInline(Ray ray, HitResult& outResult) const
		{
			// Return the distance along the ray where a hit occured
			// Slightly simplified by using half b to reduce the factor of 2
			Vec3 orig = ray.origin();
			Vec3 dir = ray.direction();
			Vec3 diff = orig - center_;

			float rsqr = radius_ * radius_;

			float a = dot(dir, dir);
			float half_b = dot(dir, diff);
			float c = diff.length_squared() - rsqr;

			float discr = half_b * half_b - (a * c);

			if (discr < 0.0f)
			{
				return false;
			}

			// for now assuming the smallest value is the closest to the origin
			// clamped to positive values
			float t = (-half_b - std::sqrt(discr)) / a;

			if (t < 0.0f)
			{
				return false;
			}

			outResult.t = t;
			outResult.point = ray.at(t);
			outResult.normal = (outResult.point - center_) * (1.0f / radius_);

			return true;
		}

	private:

		Vec3 center_{};
		float radius_{};
	};
}