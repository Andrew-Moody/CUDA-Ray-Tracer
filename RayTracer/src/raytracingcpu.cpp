#include "raytracingcpu.h"

#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	float getDiscriminant(const Ray& ray, const Vec3& center, float radius)
	{
		// Essentially an intersection test between a ray and a sphere

		Vec3 orig = ray.origin();
		Vec3 dir = ray.direction();
		Vec3 diff = orig - center;

		float rsqr = radius * radius;

		float a = dot(dir, dir);
		float b = dot(2.0f * dir, diff);
		float c = dot(diff, diff) - rsqr;

		float discr = b * b - (4 * a * c);

		// discr < 0.0f  0 roots, ray did not hit sphere
		// discr == 0.0f 1 root,  ray intersects the sphere tangent to the surface
		// discr > 0.0f  2 roots, ray intersects through the sphere 

		return discr;
	}

	float getHitPoint(const Ray& ray, const Vec3& center, float radius)
	{
		// Return the distance along the ray where a hit occured
		// Slightly simplified by using half b to reduce the factor of 2
		Vec3 orig = ray.origin();
		Vec3 dir = ray.direction();
		Vec3 diff = orig - center;

		float rsqr = radius * radius;

		float a = dot(dir, dir);
		float half_b = dot(dir, diff);
		float c = diff.length_squared() - rsqr;

		float discr = half_b * half_b - (a * c);

		if (discr < 0.0f)
		{
			return -1.0f;
		}

		// extra work if discr is == 0 but that is rare
		// for now assuming the smallest value is the closest to the origin
		return (-half_b - std::sqrt(discr)) / a;
	}


	Vec3 getNormalColor(const Ray& ray, const Vec3& center, float radius)
	{
		float t = getHitPoint(ray, center, radius);

		if (t < 0.0f)
		{
			return Vec3(0.9f, 0.9f, 1.0f);
		}

		Vec3 hit = ray.at(t);

		Vec3 normal = (hit - center) * (1.0f / radius);
		// Remap (-1, 1) to (0, 1)
		Vec3 color = 0.5f * normal + Vec3(0.5f, 0.5f, 0.5f);

		return color;
	}


	Vec3 getFlatShadeColor(const Ray& ray, const Vec3& center, float radius)
	{
		if (getDiscriminant(ray, center, radius) >= 0.0f)
		{
			return Vec3(1.0f, 0.0f, 0.0f);
		}

		return Vec3(0.9f, 0.9f, 1.0f);
	}

	Vec3 getDiscriminantColor(const Ray& ray, const Vec3& center, float radius)
	{
		// Can create fun patterns
		// 800 x 600 seems a good middle ground
		// too high resolution and it becomes smooth losing the pattern
		// too low resolution it becomes too grainy

		float discr = getDiscriminant(ray, center, radius);

		// Modulate color by the discriminant
		if (discr < 0.0f)
		{
			// Outside sphere
			return Vec3(discr, 0.2f + discr, 0.1f + discr);
		}
		else if (discr > 0.0f)
		{
			// Inside sphere
			return Vec3(0.0f, discr, 0.0f);
		}

		// Sphere edge color
		return Vec3(0.0f, 0.0f, 1.0f);

		// Original pattern
		/*if (discr < 0.0f)
			return Vec3(discr, 0.0f, 0.0f);
		else if (discr > 0.0f)
			return Vec3(0.0f, discr, 0.0f);
		return Vec3(0.0f, 0.0f, 1.0f);*/
	}


	std::vector<float> rtw::renderCPU(int width, int height, int channels)
	{
		const float aspectRatio{ static_cast<float>(width) / height };

		// Size of the viewport (unclear what units, physical?)
		const float viewHeight{ 2.0f };
		const float viewWidth{ viewHeight * aspectRatio };

		// Distance between pixels in viewport units (ideally should be equal)
		const float deltaU{ viewWidth / width };
		const float deltaV{ viewHeight / height };

		// Camera Properties
		Vec3 cameraPos{ 0.0f, 0.0f, 0.0f };
		Vec3 cameraForward{ 0.0f, 0.0f, -1.0f }; // Camera points in negative z direction
		Vec3 cameraUp{ 0.0f, 1.0f, 0.0f };
		Vec3 cameraRight{ 1.0f, 0.0f, 0.0f };
		float focalLength{ 1.0f }; // Distance from Camera to viewport

		// Sphere
		Vec3 center{ 0.5f, 0.5f, -3.0f };
		float radius = 1.0f;

		Vec3 viewCenter(cameraForward * focalLength); // Center of the screen in worldspace
		Vec3 viewOrigin = viewCenter + 0.5f * ((cameraUp * viewHeight) - (cameraRight * viewWidth)); // Topleft of screen in worldspace
		Vec3 pixelOrigin = viewOrigin + 0.5f * Vec3(deltaU, -deltaV, 0.0f); // Offset viewOrigin to the center of the first pixel

		// frame buffer
		const int size{ width * height * channels };
		std::vector<float> imageData(size);

		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				// pixel pos in worldspace
				Vec3 pixelPos{ pixelOrigin + Vec3(i * deltaU, -j * deltaV, 0.0f) };

				Ray ray(cameraPos, pixelPos - cameraPos);

				//Vec3 color = getDiscriminantColor(ray, center, radius);
				Vec3 color = getNormalColor(ray, center, radius);

				int pixelIdx = 3 * (j * width + i);
				imageData[pixelIdx] = color[0];
				imageData[pixelIdx + 1] = color[1];
				imageData[pixelIdx + 2] = color[2];
			}
		}

		return imageData;
	}
}
