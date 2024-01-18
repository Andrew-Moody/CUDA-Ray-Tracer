#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

namespace rtw
{
	class Collider
	{
	public:

		__host__ __device__ Collider() {};
		__host__ __device__ Collider(const Sphere& sphere)
			: center_{ sphere.center() }, radiusSquared_{ sphere.radiusSquared() } {};

		__host__ __device__ float checkHit(Ray ray) const
		{
			Vec3 diff = center_ - ray.origin();
			float minusHalfB = ray.direction().dot(diff);
			float c = diff.length_squared() - radiusSquared_;

			float discr = minusHalfB * minusHalfB - c;

			if (discr < 0.0f)
			{
				return 0.0f;
			}

			float sqrtDiscr = sqrtf(discr);

			float t = (minusHalfB - sqrtDiscr);

			if (t < 0.001f)
			{
				t = (minusHalfB + sqrtDiscr);

				if (t < 0.001f)
				{
					return 0.0f;
				}
			}

			return t;
		}

	private:

		Vec3 center_{};
		float radiusSquared_{};
	};


	class SphereList
	{
	public:

		__host__ __device__ SphereList() {}

		__host__ __device__ SphereList(Sphere* spheres, Collider* colliders, int count)
			: spheres_{ spheres }, colliders_{ colliders }, count_{ count }, visibleCount_{ count } {}

		__host__ __device__ Sphere& sphereAt(int index) { return spheres_[index]; }
		__host__ __device__ Collider& colliderAt(int index) { return colliders_[index]; }

		__host__ __device__ const Sphere& sphereAt(int index) const { return spheres_[index]; }
		__host__ __device__ const Collider& colliderAt(int index) const { return colliders_[index]; }

		__host__ __device__ int count() const { return count_; }

		__host__ __device__ int visibleCount() const { return visibleCount_; }

		__host__ __device__ void sortVisible()
		{
			// Start at the end of the array
			auto back = count_ - 1;

			// skip to the first visible from the end keeping any
			// non visible where they are at the end
			while (back >= 0 && !spheres_[back].visible)
			{
				--back;
			}

			// Search the rest of the array for non visible spheres
			// and swap with the last visible
			for (auto front = back - 1; front >= 0; --front)
			{
				if (!spheres_[front].visible)
				{
					// Swap spheres and colliders

					Sphere temp{ spheres_[back] };
					spheres_[back] = spheres_[front];
					spheres_[front] = temp;

					Collider tempCollider{ colliders_[back] };
					colliders_[back] = colliders_[front];
					colliders_[front] = tempCollider;
					--back;
				}
			}

			visibleCount_ = back + 1;
		}


		__host__ __device__ void updateColliders()
		{
			for (int i = 0; i < count_; ++i)
			{
				colliders_[i] = Collider{ spheres_[i] };
			}
		}

	private:

		Sphere* spheres_{};
		Collider* colliders_{};
		int count_{};
		int visibleCount_{};
	};
}