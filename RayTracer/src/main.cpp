#include <vector>

#include "image.h"
#include "raytracingcpu.h"
#include "raytracing.cuh"

int main()
{
	constexpr int width{ 800 }; // image width in pixels
	constexpr int height{ 600 }; // image height in pixels
	constexpr int channels{ 3 }; // floats per pixel 3 for rgb, 4 for rgba

	constexpr int numSpheres{ 5 };

	std::vector<float> cpuImage = rtw::renderCPU(width, height, channels, numSpheres);

	demo::storePNG("Renders/renderCPU.png", cpuImage, width, height, channels);

	std::vector<float> gpuImage = rtw::renderGPU(width, height, channels, numSpheres);

	demo::storePNG("Renders/renderGPU.png", gpuImage, width, height, channels);
} 
