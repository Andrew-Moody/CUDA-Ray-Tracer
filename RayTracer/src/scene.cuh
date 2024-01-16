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

		__host__ __device__ void clear() { begin_ = nullptr; end_ = nullptr; visibleEnd_ = nullptr; }

		__host__ __device__ Sphere* begin() const { return begin_; }
		//__host__ __device__ Sphere* end() const { return end_; }
		__host__ __device__ Sphere* end() const { return visibleEnd_; }
		__host__ __device__ Sphere* visibleEnd() const { return visibleEnd_; }

		__host__ __device__ Sphere& operator[](int index) { return begin_[index]; }

		//__host__ __device__ size_t count() const { return end_ - begin_; }
		//__host__ __device__ size_t visibleCount() const { return visibleEnd_ - begin_; }

		__host__ __device__ void sortVisible()
		{
			// back points to last visible
			auto back = end_ - 1;

			while (back >= begin_ && !back->visible)
			{
				--back;
			}

			for (auto front = back - 1; front >= begin_; --front)
			{
				if (!front->visible)
				{
					Sphere temp{ *back };
					*back = *front;
					*front = temp;
					--back;
				}
			}

			visibleEnd_ = back + 1; 
		}

	private:

		Sphere* begin_{};
		Sphere* end_{};
		Sphere* visibleEnd_{begin_};
	};


	class Scene
	{
	public:

		__host__ __device__ Scene() {};

		__host__ __device__ Scene(Sphere* begin, Sphere* end) : sceneView_{ begin, end } {}

		__host__ __device__ Scene(SceneView sceneView) : sceneView_{ sceneView } {}

		__host__ __device__ void clear() { sceneView_.clear(); }

		__host__ __device__ void sortVisible() { sceneView_.sortVisible(); }

		__host__ __device__ Vec3 getColor(Ray ray, int nBounces, curandState* randState) const
		{

//#ifndef __CUDA_ARCH__
//			Sphere::sampleCount++;
//#endif
			HitResult result{};

			Vec3 attenuation{ 1.0f, 1.0f, 1.0f };

			for (int i = 0; i <= nBounces; ++i)
			{
				// Find the closest hit
				const Sphere* closestSphere{ nullptr };
				float closestT{ 1000000.0f };

				for (const auto& sphere : sceneView_)
				{
					// t of 0.0f indicates a miss (t must be greater than a tolerance)
					float t = sphere.checkHit(ray);
					if (t > 0.0f && t < closestT)
					{
						closestSphere = &sphere;
						closestT = t;
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
//#ifndef __CUDA_ARCH__
//					Sphere::bounceCount+= i;
//
//					double error{ i - Sphere::mean };
//					Sphere::accSqrError += error * error;
//#endif

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

//#ifndef __CUDA_ARCH__
//			Sphere::bounceCount += nBounces;
//			Sphere::maxBounceCount++;
//
//			double error{ nBounces - Sphere::mean };
//			Sphere::accSqrError += error * error;
//#endif

			// Max number of bounces reached return no contribution
			return Vec3{0.0f, 0.0f, 0.0f};
		}

		__host__ __device__ void initializeScene(curandState* randState)//(Camera camera, curandState* randState)
		{
			auto sphere = sceneView_.begin();

			// Final render

			// Ground
			*sphere = Sphere{ 0.0f, -1000.0f, 0.0f, 1000.0f, Material{ Vec3{ 0.5f, 0.5f, 0.5f }, BlendMode::diffuse } };
			sphere++;

			*sphere = Sphere{ 0.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			sphere++;
			*sphere = Sphere{ -4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.4f, 0.2f, 0.1f }, BlendMode::diffuse} };
			sphere++;
			*sphere = Sphere{ 4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.7f, 0.6f, 0.5f }, BlendMode::metallic, 0.0f } };
			sphere++;

			for (int x = -11; x < 11; ++x)
			{
				for (int z = -11; z < 11; ++z)
				{
					float radius = 0.2f;
					Vec3 center{ x + 0.9f * randomFloat(randState), radius, z + 0.9f * randomFloat(randState) };
					
					if ((center - Vec3{ 4.0f, 0.2f, 0.0f }).length() > 0.9f)
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
						// hide rather than deal with dynamic memory for now
						*sphere = Sphere{};
						sphere->visible = false;
					}

					sphere++;
				}
			}

			//// Chapter 13 scene
			//*sphere = Sphere{ 0.0f, -100.5f, -1.0f, 100.0f, Material{ Vec3{ 0.8f, 0.8f, 0.4f }, BlendMode::diffuse, 1.0f } };
			//sphere++;
			//*sphere = Sphere{ 0.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.1f, 0.2f, 0.5f }, BlendMode::diffuse} };
			//sphere++;
			//*sphere = Sphere{ 1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 0.8f, 0.6f, 0.2f }, BlendMode::metallic, 0.0f } };
			//sphere++;

			//// Hollow glass
			//*sphere = Sphere{ -1.0f, 0.0f, -1.0f, 0.5f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			//sphere++;
			//*sphere = Sphere{ -1.0f, 0.0f, -1.0f, -0.4f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } };
			//sphere++;
			//

			//// Extras for custom example
			//// Diamond
			//*sphere = Sphere{ -0.1f, 0.25f, -0.25f, 0.35f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 2.4f } };
			//sphere++;
			//// Copper
			//*sphere = Sphere{ -0.5f, 0.5f, -2.0f, 0.35f, Material{ Vec3{ 1.0f, 0.4f, 0.2f }, BlendMode::metallic, 0.0f } };
			//sphere++;
			//// Ruby
			//*sphere = Sphere{ 1.0f, -.35f, 0.0f, 0.15f, Material{ Vec3{ 1.0f, 0.5f, 0.5f }, BlendMode::translucent, 0.0f, 1.8f } };


			sortVisible();
		}

		__host__ __device__ Vec3 randomVec(curandState* randstate)
		{
			return { randomFloat(randstate), randomFloat(randstate), randomFloat(randstate) };
		}

	private:

		SceneView sceneView_{};
	};
}