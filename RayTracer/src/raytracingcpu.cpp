#include "raytracingcpu.h"

#include <vector>

#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "hittable.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "random.cuh"

namespace rtw
{

	std::vector<float> rtw::renderCPU(const Camera& camera, int numSpheres)
	{
		// Create a scene for cpu
		/*std::unique_ptr<Sphere[]> spheres{ new Sphere[numSpheres] };
		Scene scene{ spheres.get(), spheres.get() + numSpheres };*/

		std::unique_ptr<Sphere[]> spheres{ new Sphere[numSpheres] };
		std::unique_ptr<Collider[]> colliders{ new Collider[numSpheres] };
		Scene scene{ spheres.get(), colliders.get(), numSpheres };
		scene.initializeScene(nullptr);

		// frame buffer
		const int size{ camera.totalPixels() * camera.nChannels() };
		std::vector<float> imageData(size);
		

		// Flattened to better simulate eventual gpu code
		for (int pixel = 0; pixel < camera.totalPixels(); ++pixel)
		{
			Vec3 worldPos = camera.getWorldPosFromPixelIndex(pixel);

			Vec3 color{};

			for (int i = 0; i < camera.nSamples(); ++i)
			{
				// sample defocus disk
				Vec3 rayOrigin = camera.getSampleOrigin(nullptr);

				// Sample a random location inside the square halfway between the pixel position
				// and the positions of neighboring pixels
				Vec3 samplePos = camera.shiftWorldPos(worldPos, randomFloat(nullptr) - 0.5f, randomFloat(nullptr) - 0.5f);

				Ray ray{ rayOrigin, (samplePos - rayOrigin).normalize() };

				color += scene.getColor(ray, camera.nBounces(), nullptr);
			}
			
			color *= 1.0f / camera.nSamples();

			int pixelIdx = 3 * pixel;
			imageData[pixelIdx] = color[0];
			imageData[pixelIdx + 1] = color[1];
			imageData[pixelIdx + 2] = color[2];
		}

		//double percent{ Sphere::behindCount / static_cast<double>(Sphere::checkCount) };
		//std::cout << "Total: " << Sphere::checkCount << ", Behind: " << Sphere::behindCount << ", Percent: " << percent << '\n';

		/*double stdDev = std::sqrt(Sphere::accSqrError / (Sphere::sampleCount - 1));

		double bouncesPSample = Sphere::bounceCount / static_cast<double>(Sphere::sampleCount);

		double maxBounceFrac = Sphere::maxBounceCount / static_cast<double>(Sphere::sampleCount);

		std::cout << "Color Checks: " << Sphere::sampleCount << ", Bounces: " << Sphere::bounceCount << ", Average Bounces: " << bouncesPSample <<
			", Max Bounce Fraction: " << maxBounceFrac << ", Std Deviation: " << stdDev << '\n';*/

		return imageData;
	}

	/*double Sphere::mean{ 1.74153 };

	uint64_t Sphere::checkCount{};
	uint64_t Sphere::behindCount{};
	uint64_t Sphere::sampleCount{};
	uint64_t Sphere::bounceCount{};
	uint64_t Sphere::maxBounceCount{};
	double Sphere::accSqrError{};*/
}
