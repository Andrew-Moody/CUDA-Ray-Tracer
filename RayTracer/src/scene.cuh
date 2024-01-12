#pragma once

#include "sphere.cuh"
#include "camera.cuh"

#include <vector>
#include <memory>

namespace rtw
{

	// Only used to perform operations on a collection for both cpu and gpu
	// does not own the resources involved
	class SceneView
	{
	public:

		__host__ __device__ SceneView() {}

		__host__ __device__ SceneView(Sphere* begin, Sphere* end) : begin_{ begin }, end_{ end } {}

		__host__ __device__ void clear() { begin_ = nullptr; end_ = nullptr; }

		__host__ __device__ Sphere* begin() const { return begin_; }
		__host__ __device__ Sphere* end() const { return end_; }

		__host__ __device__ Sphere& operator[](int index) { return begin_[index]; }

		__host__ __device__ size_t count() const { return end_ - begin_; }

	private:

		Sphere* begin_{};
		Sphere* end_{};
	};


	class Scene
	{
	public:

		__host__ __device__ Scene(Sphere* begin, Sphere* end) : sceneView_{ begin, end } {}

		__host__ __device__ Scene(SceneView sceneView) : sceneView_{ sceneView } {}

		__host__ __device__ void clear() { sceneView_.clear(); }

		__host__ __device__ Vec3 getColor(Ray ray, int nBounces, curandState* randState) const
		{
			HitResult result{};

			Vec3 attenuation{ 1.0f, 1.0f, 1.0f };

			for (int i = 0; i <= nBounces; ++i)
			{
				// Find the closest hit
				const Sphere* closestSphere{ nullptr };
				float closestT{ 1000000.0f };

				for (const auto& sphere : sceneView_)
				{
					if (sphere.hit(ray, result) && result.t < closestT)
					{
						closestSphere = &sphere;
						closestT = result.t;
					}
				}

				if (closestSphere)
				{
					// Hit something, apply attenuation and get the next bounce direction
					// Absorb half the light
					//attenuation *= 0.5f;

					//color = sphere.getColor(ray, result, randState);
					BounceResult bResult = closestSphere->bounceRay(ray, closestT, randState);
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
					float lerpT = 0.5f * (normalized(ray.direction()).y() + 1.0f);

					Vec3 ambientColor = (1.0f - lerpT) * Vec3(1.0f, 1.0f, 1.0f) + lerpT * Vec3( 0.5f, 0.7f, 1.0f );

					// Apply cumulative attenuation
					ambientColor *= attenuation;

					return ambientColor;
				}
			}

			// Max number of bounces reached return no contribution
			return Vec3{0.0f, 0.0f, 0.0f};
		}

		__host__ __device__ void initializeScene(Camera camera, curandState* randState)
		{
			//int reserved{ 1 };

			auto sphere = sceneView_.begin();

			//Material material{ Vec3{ 0.9f, 0.5f, 0.5f }, 0.0f };

			// Ground
			*sphere = Sphere{ 0.0f, -1000.0f, 0.0f, 1000.0f, Material{ Vec3{ 0.5f, 0.5f, 0.5f }, BlendMode::diffuse } };
			sphere++;

			// Final render
			*sphere = Sphere{ 0.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			sphere++;
			*sphere = Sphere{ -4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.4f, 0.2f, 0.1f }, BlendMode::diffuse} };
			sphere++;
			*sphere = Sphere{ 4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.7f, 0.6f, 0.5f }, BlendMode::metallic, 0.0f } };
			sphere++;

			int count{};

			for (int x = -11; x < 11; ++x)
			{
				for (int z = -11; z < 11; ++z)
				{
					float radius = 0.2f;
					Vec3 center{ x + 0.9f * randomFloat(randState), radius, z + 0.9f * randomFloat(randState) };
					
					if ((center - Vec3{ 4.0f, 0.2, 0 }).length() > 0.9f)
					{
						float randomMat = randomFloat(randState);

						if (randomMat < 0.75f)
						{
							// Diffuse
							Vec3 albedo = (randomVec(randState) *= randomVec(randState));
							*sphere = Sphere{ center, radius, Material{ albedo, BlendMode::diffuse } };
						}
						else if (randomMat < 0.90f)
						{
							// Metallic
							Vec3 albedo = randomVec(randState) * 0.5f + Vec3{ 0.5f, 0.5f, 0.5f};
							*sphere = Sphere{ center, radius, Material{ albedo, BlendMode::metallic, randomFloat(randState) * 0.5f } };
						}
						else
						{
							// Glass
							*sphere = Sphere{ center, radius, Material{ { 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
						}
					}
					else
					{
						// hide rather deal with dynamic memory for now
						*sphere = Sphere{};
						sphere->visible = false;
					}

					sphere++;
					count++;
				}
			}

			sphere++;

			// Chapter 13 scene
			//*sphere = Sphere{ 0.0f, -100.5f, -1.0f, 100.0f, Material{ Vec3{ 0.8f, 0.8f, 0.0f }, BlendMode::diffuse, 1.0f } };
			//sphere++;
			//*sphere = Sphere{ 0.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.1f, 0.2f, 0.5f }, BlendMode::diffuse} };
			//sphere++;
			//*sphere = Sphere{ 1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.8f, 0.6f, 0.2f }, BlendMode::metallic, 0.0f } };
			//sphere++;

			//// Hollow glass
			//*sphere = Sphere{ -1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			//sphere++;
			//*sphere = Sphere{ -1.0f, 0.0f, -1.0f, -0.4f, Material{ Vec3{ 0.9f, 0.9f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			//sphere++;

			//// Extra
			//*sphere = Sphere{ -0.35f, -0.3f, -0.5f, 0.1f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			//sphere++;
			//*sphere = Sphere{ 0.35f, -0.3f, -0.5f, 0.1f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::metallic, 0.3f } };
		}

		__host__ __device__ Vec3 randomVec(curandState* randstate)
		{
			return { randomFloat(randstate), randomFloat(randstate), randomFloat(randstate) };
		}

	private:

		SceneView sceneView_{};
	};
}