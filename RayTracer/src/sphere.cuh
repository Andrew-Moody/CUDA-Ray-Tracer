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


	class Sphere// : public Hittable
	{
	public:

		__host__ __device__ Sphere() {}

		__host__ __device__ Sphere(float x, float y, float z, float radius, Material material)
			: center_{ x, y, z }, radius_{ radius }, material_{ material } {}

		__host__ __device__ Sphere(const Vec3& center, float radius, Material material)
			: center_{ center }, radius_{ radius }, material_{material} {}

		__host__ __device__ bool hit(Ray ray, HitResult& outResult) const
		{
			// Return the distance along the ray where a hit occured
			// Slightly simplified by using half b to reduce the factor of 2
			Vec3 orig = ray.origin();
			Vec3 dir = ray.direction();
			//Vec3 dir = normalized(ray.direction());
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

			// first assume the smallest value is the closest to the origin
			// not checking the other value means you wont get refracted rays detecting hits from inside
			// transparent materials
			float t = (-half_b - std::sqrt(discr)) / a;

			// If it is negative it means the intersection along the ray in the opposite direction it was cast

			// Also at very small or zero t values floating point error may make the origin of a bounced
			// ray slightly bellow the surface leading to an false positive
			// limiting the smallest value can help
			// (haven't seen a visibly difference so far)
			if (t < 0.001f)
			{
				// Try the other possible value
				t = (-half_b + std::sqrt(discr)) / a;

				if (t < 0.001f)
				{
					return false;
				}
			}

			outResult.t = t;
			outResult.point = ray.at(t);
			outResult.normal = (outResult.point - center_) * (1.0f / radius_);

			return true;
		}

		__host__ __device__ Vec3 getColor(Ray ray, HitResult hitResult, curandState* randState) const
		{
			return hitResult.getNormalColor();
		}


		__host__ __device__ bool cantRefract(float cosTheta, float refractRatio, float threshold) const
		{
			//float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
			if (refractRatio * std::sqrt(1.0f - cosTheta * cosTheta) > 1.0f)
			{
				return true;
			}

			//Schlick approximation
			float r0 = (refractRatio - 1.0f) / (refractRatio + 1.0f);
			r0 *= r0;
			r0 = r0 + ((1.0f - r0) * static_cast<float>(pow((1.0f - cosTheta), 5)));

			return r0 > threshold;

			//return false;
		}

		__host__ __device__ Vec3 refract(Vec3 unitVector, Vec3 normal, float refractRatio, curandState* randState) const
		{
			float cosTheta = dot(-unitVector, normal);

			// book has fmin around the result. is that to prevent errors
			// if the input vectors are not unit length?
			/*if (cosTheta > 1.0f)
			{
				cosTheta = 1.0f;
			}*/

			// means the ray came from inside the volume
			// (the book does not keep normals always pointing outwards meaning this doesn't have to be done)
			if (cosTheta < 0.0f)
			{
				// flip everything around to pretend you hit an air sphere in a world of glass
				cosTheta = -cosTheta;
				normal = -normal;
				refractRatio = 1.0f / refractRatio;
			}


			float threshold{ randomFloat(randState) };
			//float threshold{ 0.0f };

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


		__host__ __device__ BounceResult bounceRay(Ray ray, float hitT, curandState* randState) const
		{
			Vec3 point = ray.at(hitT);
			Vec3 normal = (point - center_) * (1.0f / radius_);

			if (material_.mode == BlendMode::translucent)
			{
				Vec3 bounceDir = refract(normalized(ray.direction()), normal, 1.0f / material_.refractIndex, randState);

				return BounceResult{ Ray{ point, bounceDir }, material_.albedo };
			}

			
			// Generate a random vector with components in the range [-1.0f, 1.0f]
			// reject if the vector falls outside the unit circle (square length > 1)
			// this prevents the distribution from skewing towards the corners of the bounding cube
			Vec3 randUnitDir{};
			for (;;)
			{
				randUnitDir[0] = 2.0f * randomFloat(randState) - 1.0f;
				randUnitDir[1] = 2.0f * randomFloat(randState) - 1.0f;
				randUnitDir[2] = 2.0f * randomFloat(randState) - 1.0f;

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