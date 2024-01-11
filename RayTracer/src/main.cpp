#include <vector>

#include "image.h"
#include "camera.cuh"
#include "raytracingcpu.h"
#include "raytracing.cuh"

int main()
{
	constexpr int width{ 800 }; // image width in pixels
	constexpr int height{ 480 }; // image height in pixels
	constexpr int nChannels{ 3 }; // floats per pixel 3 for rgb, 4 for rgba
	constexpr int nSamples{ 500 }; // number of samples per pixel for anti-aliasing
	constexpr int nBounces{ 5 }; // number of bounces before discarding a rays contribution

	const rtw::Camera camera{ width, height, nChannels, nSamples, nBounces };

	constexpr int numSpheres{ 7 };

	//std::vector<float> cpuImage = rtw::renderCPU(camera, numSpheres);

	//demo::storePNG("Renders/renderCPU.png", cpuImage, width, height, nChannels);

	std::vector<float> gpuImage = rtw::renderGPU(camera, numSpheres);

	demo::storePNG("Renders/renderGPU.png", gpuImage, width, height, nChannels);
} 
