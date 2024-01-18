#pragma once

#include <vector>

#include "scene.cuh"
#include "sphere.cuh"
#include "collider.cuh"

namespace rtw
{
	class SceneGenerator
	{
	public:

		Scene generateScene()
		{
			// Final render

			// Ground
			spheres_.push_back(Sphere{ 0.0f, -1000.0f, 0.0f, 1000.0f, Material{ Vec3{ 0.5f, 0.5f, 0.5f }, BlendMode::diffuse } });

			// Large Spheres
			spheres_.push_back(Sphere{ 0.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } });
			
			spheres_.push_back(Sphere{ -4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.4f, 0.2f, 0.1f }, BlendMode::diffuse} });
			
			spheres_.push_back(Sphere{ 4.0f, 1.0f, 0.0f, 1.0f, Material{ Vec3{ 0.7f, 0.6f, 0.5f }, BlendMode::metallic, 0.0f } });

			// Add random small spheres
			for (int x = -11; x < 11; ++x)
			{
				for (int z = -11; z < 11; ++z)
				{
					float radius = 0.2f;
					Vec3 center{ x + 0.9f * randomFloatMT(), radius, z + 0.9f * randomFloatMT() };

					if ((center - Vec3{ 4.0f, 0.2f, 0.0f }).length() > 0.9f)
					{
						float randomMat = randomFloatMT();

						if (randomMat < 0.75f)
						{
							// Diffuse
							Vec3 albedo = randomVecMT() *= randomVecMT();
							spheres_.push_back(Sphere{ center, radius, Material{ albedo, BlendMode::diffuse } });
						}
						else if (randomMat < 0.90f)
						{
							// Metallic
							Vec3 albedo = randomVecMT() * 0.5f + Vec3{ 0.5f, 0.5f, 0.5f };
							spheres_.push_back(Sphere{ center, radius, Material{ albedo, BlendMode::metallic, randomFloatMT() * 0.5f } });
						}
						else
						{
							// Glass
							spheres_.push_back(Sphere{ center, radius, Material{ { 1.0f, 1.0f, 1.0f }, BlendMode::translucent, 0.0f, 1.5f } });
						}
					}
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

			generateColliders();

			return Scene{ spheres_.data(), colliders_.data(), static_cast<int>(spheres_.size()) };
		}

	private:

		void generateColliders()
		{
			colliders_.reserve(spheres_.size());

			for (int i = 0; i < spheres_.size(); ++i)
			{
				colliders_.push_back(Collider{ spheres_[i] });
			}
		}

		Vec3 randomVecMT() { return { randomFloatMT(), randomFloatMT(), randomFloatMT() }; }

		std::vector<Sphere> spheres_;
		std::vector<Collider> colliders_;
	};
}