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
		bool visible{ true };

		__host__ __device__ Sphere() {}

		__host__ __device__ Sphere(float x, float y, float z, float radius, Material material)
			: center_{ x, y, z }, radius_{ radius }, material_{ material } {}

		__host__ __device__ Sphere(const Vec3& center, float radius, Material material)
			: center_{ center }, radius_{ radius }, material_{material} {}

		__host__ __device__ Vec3 center() const { return center_; }
		__host__ __device__ float radius() const { return radius_; }
		__host__ __device__ float radiusSquared() const { return radius_ * radius_; }


		__host__ __device__ float checkHit(Ray ray) const
		{
			/*if (!visible)
			{
				return 0.0f;
			}*/

//#ifndef __CUDA_ARCH__
//			Sphere::checkCount++;
//#endif
			

			// Reversed the order since we want negative b/2 eventually
			Vec3 diff = center_ - ray.origin();
			//float a = dot(ray.direction(), ray.direction()); // direction is unit length
			float minusHalfB = ray.direction().dot(diff);

			// check if the sphere center is "behind" the ray origin
//			if (minusHalfB < 0.0f)
//			{
//				// the only time a sphere center can be behind the origin and still intersect at
//				// a positive t is if the ray originates inside the sphere (but not on the surface due to refraction)
//				// the only time this should happen is if two spheres are more than half overlapped
//				// or a sphere is overlapping the camera
//				// if a glass sphere greatly overlaps any other sphere you would need to 
//				// also check if the distance from origin to center is less than the radius (with some tolerance
//				// to exclude the surface of the sphere the ray is bouncing from)
//#ifndef __CUDA_ARCH__
//				Sphere::behindCount++;
//#endif
//				return 0.0f;
//			}

			//float c = diff.length_squared() - radiusSquared_;

			float c = diff.length_squared() - radius_ * radius_;

			float discr = minusHalfB * minusHalfB - c;


			if (discr < 0.0f)
			{
				return 0.0f;
			}

			//float sqrtDiscr = std::sqrt(discr);

			float sqrtDiscr = sqrtf(discr);

			float t = (minusHalfB - sqrtDiscr);

			// If t is negative it most likely means the sphere is behind the ray origin
			// if t is close to zero it is detecting the point of bounce or refraction
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

		//__host__ __device__ float fastHit(Ray ray) const
		//{
		//	// t1 might be a small +- number when checked for the refracting from sphere
		//	// t2 might be a small +- number when checked for the diffuse/reflecting from sphere
		//	// either will have to be greater than a tolerance to qualify as a hit

		//	// Since we want the closest sphere, a final value for t greater or equal to a max distance
		//	// will be considered a miss. The mac distance will have to be picked to limit precision issues
		//	// for distance objects (10,000 meters for a first guess)
		//	// MAXHALFB is the square of this max distance since it is the square length
		//	// MAXB is 2 x MAXHALFB so that potentially large numbers (up to MAXHALFB) subtracted from MAXB
		//	// still yield >= MAXHALFB

		//	const float TOLERANCE{ 0.001f };
		//	const float MAXHALFB{ 1.0e8f };
		//	const float MAXB{ 2.0f * MAXHALFB };

		//	// Reversed the order since we want negative b/2 eventually
		//	Vec3 diff = center_ - ray.origin();
		//	float minusHalfB = ray.direction().dot(diff);
		//	float c = diff.length_squared() - radiusSquared_;
		//	float discr = minusHalfB * minusHalfB - c;

		//	discr = discr >= 0.0f ? discr : 1.0e16f;

		//	discr = std::sqrt(discr);

		//	//float sqrtDiscr = std::sqrt(discr);

		//	// Get the sqrt only if discr is positive else return MAX
		//	//float sqrtDiscr = discr >= 0.0f ? std::sqrt(discr) : MAXB;

		//	//float sqrtDiscr = std::sqrt(discr);
		//	//sqrtDiscr = fminf(sqrtDiscr, MAXB); // returns MAXB if discr is NAN, but is it safe?
		//	
		//	// The two potential intersection distances
		//	//float t1 = minusHalfB - sqrtDiscr; // t will also be negative if discr was < 0
		//	//float t2 = minusHalfB + sqrtDiscr; // at minusHalfB = -MAXHALFB t2 >= MAXHALFB even if discr was < 0

		//	// t1 needs to be positive to be in front of the ray but also needs be greater
		//	// than a tolerance to prevent a refracting from sphere causing a false positive 
		//	//float t = t1 > TOLERANCE ? t1 : minusHalfB + sqrtDiscr; // t = t2 if discr < 0 or t < 0.001f
		//	
		//	// t now equals either a valid t1, or t2 which may be +, -, or >= MAXHALFB if discr was < 0
		//	// t2 is only valid during refraction and needs to be greater than a tolerance or a sphere being
		//	// bounced from will cause a false positive
		//	//t = t > TOLERANCE ? t : MAXB;

		//	// If the final t >= MAXHALFB it will be considered a miss

		//	float t1 = minusHalfB - discr; // t will also be negative if discr was < 0
		//	//float t2 = minusHalfB + sqrtDiscr; // at minusHalfB = -MAXHALFB t2 >= MAXHALFB even if discr was < 0

		//	// t1 needs to be positive to be in front of the ray but also needs be greater
		//	// than a tolerance to prevent a refracting from sphere causing a false positive 
		//	float t = t1 > TOLERANCE ? t1 : minusHalfB + discr; // t = t2 if discr < 0 or t < 0.001f

		//	// t now equals either a valid t1, or t2 which may be +, -, or >= MAXHALFB if discr was < 0
		//	// t2 is only valid during refraction and needs to be greater than a tolerance or a sphere being
		//	// bounced from will cause a false positive
		//	t = t > TOLERANCE ? t : MAXB;


		//	return t;
		//}



		__host__ __device__ Vec3 getColor(Ray ray, HitResult hitResult, curandState* randState) const
		{
			return hitResult.getNormalColor();
		}


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
				// Normalizing ray direction here should be extraneous (Somehow increases register count by 4?)
				Vec3 bounceDir = refract(normalized(ray.direction()), normal, 1.0f / material_.refractIndex, randState);
				//Vec3 bounceDir = refract(Vec3(ray.direction()), normal, 1.0f / material_.refractIndex, randState);

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

		/*static uint64_t checkCount;
		static uint64_t behindCount;
		static uint64_t sampleCount;
		static uint64_t bounceCount;
		static uint64_t maxBounceCount;
		static double mean;
		static double accSqrError;*/

	private:

		Vec3 center_{};
		float radius_{};
		//float radiusSquared_{radius_ * radius_};
		Material material_{};
	};

	/*class SphereCollider
	{
	public:

		Vec3 center_{};
		float radius_{};


	};*/
}