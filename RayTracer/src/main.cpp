#include <vector>
#include <chrono>
#include "image.h"
#include "scenegenerator.h"
#include "camera.cuh"
#include "raytracingcpu.h"
#include "raytracing.cuh"


int main()
{
	constexpr int width{ 1200 }; // image width in pixels
	constexpr int height{ 675 }; // image height in pixels
	constexpr int nChannels{ 3 }; // floats per pixel 3 for rgb, 4 for rgba
	constexpr int nSamples{ 5 }; // number of samples per pixel for anti-aliasing
	constexpr int nBounces{ 50 }; // number of bounces before discarding a rays contribution

	rtw::Camera camera{ width, height, nChannels, nSamples, nBounces };

	// Settings for final Render
	camera.setTransform(13.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f);
	camera.setFocus(20.0f, 10.0f, 100.0);

	////Settings for Chapter 13 render (depth of field)
	//camera.setTransform(-2.0f, 2.0f, 1.0f, 0.0f, 0.0f, -1.0f);
	//camera.setFocus(20.0f, 3.4640f, 5.0f);

	////Settings for custom example highlighting refraction
	//camera.setTransform(-2.0f, 2.0f, 1.0f, 0.0f, 0.0f, -1.0f);
	//camera.setFocus(35.0f, 3.4640f, 1500.0f);

	rtw::SceneGenerator sceneGenerator{};
	rtw::Scene scene{ sceneGenerator.generateScene() };


	auto start = std::chrono::steady_clock::now();

	//std::vector<float> cpuImage = rtw::renderCPU(camera, scene);

	//auto end = std::chrono::steady_clock::now();

	//demo::storePNG("Renders/renderCPU.png", cpuImage, width, height, nChannels);

	std::vector<float> gpuImage = rtw::renderGPU(camera, scene);

	auto end = std::chrono::steady_clock::now();

	demo::storePNG("Renders/renderGPU.png", gpuImage, width, height, nChannels);

	

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Total Time: " << duration.count() / 1000.0f << "s" << std::endl;

	return 0;
} 
