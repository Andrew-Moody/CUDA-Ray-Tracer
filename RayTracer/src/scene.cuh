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

		__host__ __device__ Vec3 getColor(Ray ray) const
		{
			HitResult result{};
			float closestT{ 1000000.0f };

			Vec3 color{ 0.9f, 0.9f, 1.0f };

			for (const auto& sphere : sceneView_)
			{
				if (sphere.hitInline(ray, result) && result.t < closestT)
				{
					color = result.getNormalColor();
					closestT = result.t;
				}
			}

			return color;
		}

		__host__ __device__ void initializeScene(Camera camera)
		{
			int reserved{ 1 };

			auto sphere = sceneView_.begin();

			// "Ground"
			*sphere = Sphere{ 0.0f, -150.0f, -20.0f, 150.0f };
			sphere++;

			// distribute remaining spheres across the screen
			float scaleFactor = 1.0f / (1 + sceneView_.end() - sphere);
			float r = 0.5f / (sceneView_.end() - sphere);

			for (int i = 0; sphere < sceneView_.end(); ++sphere, ++i)
			{
				float x = static_cast<float>(i + 1) * scaleFactor;
				float y = static_cast<float>(i + 1) * scaleFactor;

				Vec3 center = camera.normalizedScreenToWorld(x, y);

				sceneView_[i + reserved] = Sphere{ center, r };
			}
		}

	private:

		SceneView sceneView_{};
	};
}