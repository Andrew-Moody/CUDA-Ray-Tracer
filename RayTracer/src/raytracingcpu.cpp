#include "raytracingcpu.h"

#include <vector>

#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "hittable.cuh"
#include "scenegenerator.h"
#include "scene.cuh"
#include "sphere.cuh"
#include "random.cuh"

namespace rtw
{

	std::vector<float> rtw::renderCPU(const Camera& camera, const Scene& scene)
	{
		// frame buffer
		const int size{ camera.totalPixels() * camera.nChannels() };
		std::vector<float> imageData(size);
		

		// Flattened to better simulate eventual gpu code
		for (int pixel = 0; pixel < camera.totalPixels(); ++pixel)
		{
			Vec3 worldPos = camera.getWorldPosFromPixelIndex(pixel);

			Vec3 color{};

			uint32_t pcgState = pixel;

			for (int i = 0; i < camera.nSamples(); ++i)
			{
				Ray ray = camera.generateRay(worldPos, pcgState);

				color += scene.getColor(ray, camera.nBounces(), pcgState);
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
