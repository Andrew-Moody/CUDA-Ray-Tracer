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
					float lerpT = 0.5f * (ray.direction().y() + 1.0f);

					Vec3 ambientColor = (1.0f - lerpT) * Vec3(1.0f, 1.0f, 1.0f) + lerpT * Vec3( 0.5f, 0.7f, 1.0f );

					// Apply cumulative attenuation
					ambientColor *= attenuation;

					return ambientColor;
				}
			}

			// Max number of bounces reached return no contribution
			return Vec3{0.0f, 0.0f, 0.0f};
		}

		__host__ __device__ void initializeScene(Camera camera)
		{
			//int reserved{ 1 };

			//auto sphere = sceneView_.begin();

			//Material material{ Vec3{ 0.9f, 0.5f, 0.5f }, 0.0f };

			//// "Ground"
			//*sphere = Sphere{ 0.0f, -150.1f, -13.0f, 150.0f, Material{ Vec3{ 0.4f, 0.6f, 0.4f }, 1.0f } };
			//sphere++;

			//// distribute remaining spheres across the screen
			//float scaleFactor = 1.0f / (1 + sceneView_.end() - sphere);
			//float r = 0.5f / (sceneView_.end() - sphere);

			//for (int i = 0; sphere < sceneView_.end(); ++sphere, ++i)
			//{
			//	float x = static_cast<float>(i + 1) * scaleFactor;
			//	float y = static_cast<float>(i + 1) * scaleFactor;

			//	Vec3 center = camera.normalizedScreenToWorld(x, y);

			//	sceneView_[i + reserved] = Sphere{ center, r, material };

			//	material.roughness = 1.0f;
			//}

			auto sphere = sceneView_.begin();
			*sphere = Sphere{ 0.0f, -100.5f, -1.0f, 100.0f, Material{ Vec3{ 0.8f, 0.8f, 0.0f }, BlendMode::diffuse, 1.0f } };
			sphere++;
			*sphere = Sphere{ 0.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.1f, 0.2f, 0.5f }, BlendMode::diffuse} };
			sphere++;
			*sphere = Sphere{ 1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.8f, 0.6f, 0.2f }, BlendMode::metallic, 0.0f } };
			sphere++;
			*sphere = Sphere{ 0.35f, -0.3f, -0.5f, 0.1f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::metallic, 0.3f } };
			sphere++;

			// Hollow glass
			*sphere = Sphere{ -1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			sphere++;
			*sphere = Sphere{ -1.0f, 0.0f, -1.0f, -0.4f, Material{ Vec3{ 0.9f, 0.9f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			sphere++;

			*sphere = Sphere{ -0.35f, -0.3f, -0.5f, 0.1f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
		}

	private:

		SceneView sceneView_{};
	};
}