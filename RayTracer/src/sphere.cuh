#pragma once

#include "hittable.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "random.cuh"

namespace rtw
{
	struct BounceResult
	{
		Ray bounceRay;
		Vec3 attenuation;
	};

	enum class BlendMode
	{
		diffuse,
		metallic,
		translucent
	};

	struct Material
	{
		Vec3 albedo{};
		BlendMode mode{};
		float roughness{};
		float refractIndex{};
	};


	class Sphere
	{
	public:
		bool visible{ true };

		__host__ __device__ Sphere() {}

		__host__ __device__ Sphere(float x, float y, float z, float radius, Material material)
			: center_{ x, y, z }, radius_{ radius }, material_{ material } {}

		__host__ __device__ Sphere(const Vec3& center, float radius, Material material)
			: center_{ center }, radius_{ radius }, material_{material} {}

		__host__ __device__ Vec3 center() const { return center_; }
		__host__ __device__ float radius() const { return radius_; }
		__host__ __device__ float radiusSquared() const { return radius_ * radius_; }


		__host__ __device__ bool cantRefract(float cosTheta, float refractRatio, float threshold) const
		{
			if (refractRatio * sqrtf(1.0f - cosTheta * cosTheta) > 1.0f)
			{
				return true;
			}

			//Schlick approximation
			float r0 = (refractRatio - 1.0f) / (refractRatio + 1.0f);
			r0 *= r0;
			//Changing pow to powf or 5 to 5.0f reduces registers/thread from 56 to 46
			//r0 = r0 + ((1.0f - r0) * static_cast<float>(pow((1.0f - cosTheta), 5)));
			r0 = r0 + ((1.0f - r0) * powf((1.0f - cosTheta), 5.0f));

			return r0 > threshold;

			//return false;
		}

		__host__ __device__ Vec3 refract(Vec3 unitVector, Vec3 normal, float refractRatio, float threshold) const
		{
			//float cosTheta = dot(-unitVector, normal);
			float cosTheta = -dot(unitVector, normal);

			// book has fmin around the result. is that to prevent errors
			// if the input vectors are not unit length?
			/*if (cosTheta > 1.0f)
			{
				cosTheta = 1.0f;
			}*/

			// means the ray came from inside the volume
			// the book does not keep normals always pointing outwards meaning it skips this step
			// but consistent normals is helpful in other places
			if (cosTheta < 0.0f)
			{
				// flip everything around to pretend you hit an air sphere in a world of glass
				cosTheta = -cosTheta;
				normal = -normal;
				refractRatio = 1.0f / refractRatio;
			}

			if (cantRefract(cosTheta, refractRatio, threshold))
			{
				// Reflect
				return unitVector - (2.0f * dot(unitVector, normal) * normal);
			}
			
			// Refract
			Vec3 perpendicular = refractRatio * (unitVector + (cosTheta * normal));
			Vec3 parallel = normal * -std::sqrt(std::fabs(1.0f - perpendicular.length_squared()));

			return perpendicular + parallel;
		}


		__host__ __device__ BounceResult bounceRay(Ray ray, float hitT, uint32_t& pcgState) const
		{
			Vec3 point = ray.at(hitT);
			Vec3 normal = (point - center_) * (1.0f / radius_);

			if (material_.mode == BlendMode::translucent)
			{
				// Normalizing ray direction here should be extraneous (Somehow increases register count by 4?)
				// Seems decoupling collider from sphere fixed the issue
				//Vec3 bounceDir = refract(normalized(ray.direction()), normal, 1.0f / material_.refractIndex, randomFloat(pcgState));
				Vec3 bounceDir = refract(Vec3(ray.direction()), normal, 1.0f / material_.refractIndex, randomFloat(pcgState));

				return BounceResult{ Ray{ point, bounceDir }, material_.albedo };
			}

			
			// Generate a random vector with components in the range [-1.0f, 1.0f]
			// reject if the vector falls outside the unit circle (square length > 1)
			// this prevents the distribution from skewing towards the corners of the bounding cube
			Vec3 randUnitDir{};
			for (;;)
			{
				randUnitDir[0] = 2.0f * randomFloat(pcgState) - 1.0f;
				randUnitDir[1] = 2.0f * randomFloat(pcgState) - 1.0f;
				randUnitDir[2] = 2.0f * randomFloat(pcgState) - 1.0f;

				if (randUnitDir.length_squared() <= 1.0f)
				{
					randUnitDir.normalize();
					break;
				}
			}		


			if (material_.mode == BlendMode::diffuse)
			{
				// Convert uniform distribution to lambertian distribution
				Vec3 diffuseDir = randUnitDir + normal;

				// reduce issues with near zero vectors
				if (diffuseDir.length_squared() < 1.0e-8f)
				{
					diffuseDir = normal;
				}

				diffuseDir.normalize();

				return BounceResult{ Ray{ point, diffuseDir }, material_.albedo };
			}
			


			Vec3 reflectDir = ray.direction() - (2.0f * dot(ray.direction(), normal) * normal);

			// What is the correct name for the "fuzz" parameter
			Vec3 fuzzReflectDir = material_.roughness * randUnitDir + reflectDir;

			// reduce issues with near zero vectors
			if (fuzzReflectDir.length_squared() < 1.0e-8f)
			{
				fuzzReflectDir = reflectDir;
			}

			fuzzReflectDir.normalize();

			return BounceResult{ Ray{ point, fuzzReflectDir }, material_.albedo };
		}

	private:

		Vec3 center_{};
		float radius_{};
		Material material_{};
	};
}