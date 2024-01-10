#include "raytracingcpu.h"

#include <vector>

#include "vec3.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "hittable.cuh"
#include "scene.cuh"
#include "sphere.cuh"

namespace rtw
{

	std::vector<float> rtw::renderCPU(int width, int height, int channels, int numSpheres)
	{
		const Camera camera{width, height};

		// Create a scene for cpu
		std::unique_ptr<Sphere[]> spheres{ new Sphere[numSpheres] };
		Scene scene{ spheres.get(), spheres.get() + numSpheres };
		scene.initializeScene(camera);

		// frame buffer
		const int size{ width * height * channels };
		std::vector<float> imageData(size);
		

		// Flattened to better simulate eventual gpu code
		for (int pixel = 0; pixel < camera.totalPixels(); ++pixel)
		{
			Ray ray = camera.getRayFromPixelIndex(pixel);

			Vec3 color{ scene.getColor(ray) };

			int pixelIdx = 3 * pixel;
			imageData[pixelIdx] = color[0];
			imageData[pixelIdx + 1] = color[1];
			imageData[pixelIdx + 2] = color[2];
		}

		return imageData;
	}
}
