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
		std::unique_ptr<Sphere[]> spheres{ new Sphere[numSpheres] };
		Scene scene{ spheres.get(), spheres.get() + numSpheres };
		scene.initializeScene(camera);

		// frame buffer
		const int size{ camera.totalPixels() * camera.nChannels() };
		std::vector<float> imageData(size);
		

		// Flattened to better simulate eventual gpu code
		for (int pixel = 0; pixel < camera.totalPixels(); ++pixel)
		{
			Vec3 worldPos = camera.getWorldPosFromPixelIndex(pixel);

			Vec3 cameraPos = camera.cameraPosition();
			float deltaU = camera.deltaU();
			float deltaV = camera.deltaV();

			Vec3 color{};

			for (int i = 0; i < camera.nSamples(); ++i)
			{
				// Sample a random location inside the square halfway between the pixel position
				// and the positions of neighboring pixels
				Vec3 samplePos{ worldPos };
				samplePos[0] += deltaU * ( randomFloat(nullptr) - 0.5f );
				samplePos[1] += deltaV * ( randomFloat(nullptr) - 0.5f );

				Ray ray{ camera.cameraPosition(), samplePos - camera.cameraPosition() };

				color += scene.getColor(ray, camera.nBounces(), nullptr);
			}
			
			color *= 1.0f / camera.nSamples();

			int pixelIdx = 3 * pixel;
			imageData[pixelIdx] = color[0];
			imageData[pixelIdx + 1] = color[1];
			imageData[pixelIdx + 2] = color[2];
		}

		return imageData;
	}
}
