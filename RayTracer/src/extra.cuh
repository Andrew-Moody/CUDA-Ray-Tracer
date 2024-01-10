#pragma once

#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	// Functions that can create fun renders or for testing 

	Vec3 getDiscriminantColor(const Ray& ray, const Vec3& center, float radius)
	{
		// Can create fun patterns
		// 800 x 600 seems a good middle ground
		// too high resolution and it becomes smooth losing the pattern
		// too low resolution it becomes too grainy

		Vec3 orig = ray.origin();
		Vec3 dir = ray.direction();
		Vec3 diff = orig - center;

		float rsqr = radius * radius;

		float a = dot(dir, dir);
		float b = dot(2.0f * dir, diff);
		float c = dot(diff, diff) - rsqr;

		float discr = b * b - (4 * a * c);

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
}