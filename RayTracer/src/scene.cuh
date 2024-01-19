#pragma once

#include "collider.cuh"
#include "sphere.cuh"
#include "camera.cuh"

#include <vector>
#include <memory>

namespace rtw
{
	class Scene
	{
	public:

		__host__ __device__ Scene() {};

		__host__ __device__ Scene(const Sphere* spheres, const Collider* colliders, int sphereCount)
			: spheres_{ spheres }, colliders_{ colliders }, count_{ sphereCount } {}

		const Sphere* getSpheres() const { return spheres_; };
		const Collider* getColliders() const { return colliders_; }
		int getSphereCount() const { return count_; }


		__host__ __device__ Vec3 getColor(Ray ray, int nBounces, uint32_t& pcgState) const
		{
			Vec3 attenuation{ 1.0f, 1.0f, 1.0f };

			for (int i = 0; i <= nBounces; ++i)
			{
				// Find the closest hit
				int closestIndex{ -1 };
				float closestT{ 1.0e4f };


				for (int i = 0; i < count_; ++i)
				{
					// t of 0.0f indicates a miss (t must be greater than a tolerance)
					float t = colliders_[i].checkHit(ray);
					if (t > 0.0f && t < closestT)
					{
						closestIndex = i;
						closestT = t;
					}
				}			


				if (closestIndex >= 0)
				{
					// Hit something, apply attenuation and get the next bounce direction
					// Absorb half the light
					//attenuation *= 0.5f;

					//color = sphere.getColor(ray, result, randState);
					BounceResult bResult = spheres_[closestIndex].bounceRay(ray, closestT, pcgState);
					ray = bResult.bounceRay;
					attenuation *= bResult.attenuation;
				}
				else
				{
					// ray failed to hit anything return the accumulated color
					// Could make this a gradient
					//Vec3 ambientColor{ 0.5f, 0.7f, 1.0f };

					// lerp the ambient light depending of the direction of the final ray
					// seems to produce a nice daylight effect
					float lerpT = 0.5f * (ray.direction().y() + 1.0f);

					Vec3 ambientColor = (1.0f - lerpT) * Vec3(1.0f, 1.0f, 1.0f) + lerpT * Vec3(0.5f, 0.7f, 1.0f);

					// Apply cumulative attenuation
					ambientColor *= attenuation;

					return ambientColor;
				}
			}

			// Max number of bounces reached return no contribution
			return Vec3{ 0.0f, 0.0f, 0.0f };
		}

	private:

		const Sphere* spheres_{};
		const Collider* colliders_{};
		int count_{};
	};
}