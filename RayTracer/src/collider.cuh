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


		__host__ __device__ float fastHit(Ray ray) const
		{
			// t1 might be a small +- number when checked for the refracting from sphere
			// t2 might be a small +- number when checked for the diffuse/reflecting from sphere
			// either will have to be greater than a tolerance to qualify as a hit

			// Since we want the closest sphere, a final value for t greater or equal to a max distance
			// will be considered a miss. The mac distance will have to be picked to limit precision issues
			// for distance objects (10,000 meters for a first guess)
			// MAXHALFB is the square of this max distance since it is the square length
			// MAXB is 2 x MAXHALFB so that potentially large numbers (up to MAXHALFB) subtracted from MAXB
			// still yield >= MAXHALFB

			/*const float TOLERANCE{ 0.001f };
			const float MAXHALFB{ 1.0e8f };
			const float MAXB{ 2.0f * MAXHALFB };*/

			// Reversed the order since we want negative b/2 eventually
			Vec3 diff = center_ - ray.origin();
			float minusHalfB = ray.direction().dot(diff);
			float c = diff.length_squared() - radiusSquared_;
			float discr = minusHalfB * minusHalfB - c;

			// Early return actually seems to help when the the majority of
			// cases will return such that all threads in a warp can return early
			
			//discr = discr >= 0.0f ? discr : 1.0e16f;

			if (discr < 0.0f)
			{
				return 1.0e8f;
			}

			discr = std::sqrt(discr);

			//float sqrtDiscr = std::sqrt(discr);

			// Get the sqrt only if discr is positive else return MAX
			//float sqrtDiscr = discr >= 0.0f ? std::sqrt(discr) : MAXB;

			//float sqrtDiscr = std::sqrt(discr);
			//sqrtDiscr = fminf(sqrtDiscr, MAXB); // returns MAXB if discr is NAN, but is it safe?
			
			// The two potential intersection distances
			//float t1 = minusHalfB - sqrtDiscr; // t will also be negative if discr was < 0
			//float t2 = minusHalfB + sqrtDiscr; // at minusHalfB = -MAXHALFB t2 >= MAXHALFB even if discr was < 0

			// t1 needs to be positive to be in front of the ray but also needs be greater
			// than a tolerance to prevent a refracting from sphere causing a false positive 
			//float t = t1 > TOLERANCE ? t1 : minusHalfB + sqrtDiscr; // t = t2 if discr < 0 or t < 0.001f
			
			// t now equals either a valid t1, or t2 which may be +, -, or >= MAXHALFB if discr was < 0
			// t2 is only valid during refraction and needs to be greater than a tolerance or a sphere being
			// bounced from will cause a false positive
			//t = t > TOLERANCE ? t : MAXB;

			// If the final t >= MAXHALFB it will be considered a miss

			//float t1 = minusHalfB - discr; // t will also be negative if discr was < 0
			//float t2 = minusHalfB + sqrtDiscr; // at minusHalfB = -MAXHALFB t2 >= MAXHALFB even if discr was < 0

			// t1 needs to be positive to be in front of the ray but also needs be greater
			// than a tolerance to prevent a refracting from sphere causing a false positive 
			//float t = t1 > TOLERANCE ? t1 : minusHalfB + discr; // t = t2 if discr < 0 or t < 0.001f

			// t now equals either a valid t1, or t2 which may be +, -, or >= MAXHALFB if discr was < 0
			// t2 is only valid during refraction and needs to be greater than a tolerance or a sphere being
			// bounced from will cause a false positive
			//t = t > TOLERANCE ? t : MAXB;

			float t = minusHalfB - discr; // t will also be negative if discr was < 0
			//float t2 = minusHalfB + sqrtDiscr;

			if (t < 0.001f)
			{
				t = minusHalfB + discr;

				if (t < 0.001f)
				{
					return 1.0e5f;
				}
			}

			return t;
		}

	private:

		Vec3 center_{};
		float radiusSquared_{};
	};
}
