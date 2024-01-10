#include "sphere.cuh"

#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	__host__ __device__ bool Sphere::hit(Ray ray, HitResult& outResult) const
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
		outResult.point = ray.at((-half_b - std::sqrt(discr)) / a);
		outResult.normal = (outResult.point - center_) * (1.0f / radius_);

		return true;
	}
}