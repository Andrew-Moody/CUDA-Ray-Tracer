#include <vector>

#include "image.h"
#include "camera.cuh"
#include "raytracingcpu.h"
#include "raytracing.cuh"

int main()
{
	constexpr int width{ 800 }; // image width in pixels
	constexpr int height{ 600 }; // image height in pixels
	constexpr int nChannels{ 3 }; // floats per pixel 3 for rgb, 4 for rgba
	constexpr int nSamples{ 5 }; // number of samples per pixel for anti-aliasing
	constexpr int nBounces{ 5 }; // number of bounces before discarding a rays contribution

	rtw::Camera camera{ width, height, nChannels, nSamples, nBounces };

	//Settings for Chapter 13 render (depth of field)
	//camera.setTransform(-2.0f, 2.0f, 1.0f, 0.0f, 0.0f, -1.0f);
	//camera.setFocus(20.0f, 3.4640f, 5.0f);

	// Settings for final Render
	camera.setTransform(13.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f);
	camera.setFocus(20.0f, 10.0f, 100000.9097f);

	constexpr int numSpheres{ 488 };

	//std::vector<float> cpuImage = rtw::renderCPU(camera, numSpheres);

	//demo::storePNG("Renders/renderCPU.png", cpuImage, width, height, nChannels);

	std::vector<float> gpuImage = rtw::renderGPU(camera, numSpheres);

	demo::storePNG("Renders/renderGPU.png", gpuImage, width, height, nChannels);

	int test{};
	//std::cin >> test;

	return 0;
} 
