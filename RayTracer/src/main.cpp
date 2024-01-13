#include <vector>

#include "image.h"
#include "camera.cuh"
#include "raytracingcpu.h"
#include "raytracing.cuh"

int main()
{
	constexpr int width{ 800 }; // image width in pixels
	constexpr int height{ 450 }; // image height in pixels
	constexpr int nChannels{ 3 }; // floats per pixel 3 for rgb, 4 for rgba
	constexpr int nSamples{ 100 }; // number of samples per pixel for anti-aliasing
	constexpr int nBounces{ 50 }; // number of bounces before discarding a rays contribution

	rtw::Camera camera{ width, height, nChannels, nSamples, nBounces };

	// Settings for final Render
	camera.setTransform(13.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f);
	camera.setFocus(20.0f, 10.0f, 100.0);
	constexpr int numSpheres{ 488 };

	//Settings for Chapter 13 render (depth of field)
	/*camera.setTransform(-2.0f, 2.0f, 1.0f, 0.0f, 0.0f, -1.0f);
	camera.setFocus(20.0f, 3.4640f, 5.0f);
	constexpr int numSpheres{ 5 };*/

	//Settings for custom example highlighting refraction
	/*camera.setTransform(-2.0f, 2.0f, 1.0f, 0.0f, 0.0f, -1.0f);
	camera.setFocus(35.0f, 3.4640f, 1500.0f);
	constexpr int numSpheres{ 8 };*/

	/*std::vector<float> cpuImage = rtw::renderCPU(camera, numSpheres);

	demo::storePNG("Renders/renderCPU.png", cpuImage, width, height, nChannels);*/

	std::vector<float> gpuImage = rtw::renderGPU(camera, numSpheres);

	demo::storePNG("Renders/renderGPU.png", gpuImage, width, height, nChannels);

	return 0;
} 
