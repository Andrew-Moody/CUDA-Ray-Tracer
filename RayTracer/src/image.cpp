#include "image.h"

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STBI_MSC_SECURE_CRT // Can also disable warnings in external includes
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace demo;

std::vector<float> demo::loadPNG(const std::string& filepath, int& outWidth, int& outHeight, int& outChannels, int desiredChannels)
{
	std::unique_ptr<stbi_uc[], StbiImageDeleter> data{ stbi_load(filepath.c_str(), &outWidth, &outHeight, &outChannels, desiredChannels) };

	// Convert image data to from uint (0, 255) to float (0, 1)
	// Have to make a copy anyway to get a vector or std::array so it makes sense to convert at the same time

	size_t size = outWidth * outHeight * outChannels;

	std::vector<float> floatData(size);

	for (int i = 0; i < size; ++i)
	{
		floatData[i] = data[i] / 255.0f;
	}

	return floatData;
}

void demo::storePNG(const std::string& filepath, const std::vector<float>& data, int width, int height, int channels)
{
	// Convert float to stbi_uc(unsigned char) and save using stb_image_write.h

	std::vector<stbi_uc> uintData(data.size());

	int i = 0;
	for (float value : data)
	{
		// The square root serves to convert from linear space to gamma space
		uintData[i++] = static_cast<stbi_uc>(255 * std::sqrt(value));
	}

	if (!stbi_write_png(filepath.c_str(), width, height, channels, uintData.data(), channels * width))
	{
		std::cout << "stbi_write_png Failed to write file" << '\n';
	}
}

void demo::storePPM(const std::string& filepath, const std::vector<float>& data, int width, int height, int channels)
{
	// Save image as a .ppm file. Can be done without a library since it has a simple header
	// and can be encoded for ASCII. See Ray Tracing in One Weekend for explanation

	std::ofstream image(filepath);

	if (!image.is_open())
	{
		std::cout << "Failed to open output file";
		return;
	}

	image << "P3\n" << width << ' ' << height << "\n255\n";

	int stride{ 3 * width };

	for (int h = 0; h < height; ++h)
	{
		int rowOffset = 3 * width * h;

		for (int w = 0; w < stride; w += 3)
		{
			int pixel = rowOffset + w;

			int r{ static_cast<int>(255 * data[pixel]) };
			int b{ static_cast<int>(255 * data[pixel + 1]) };
			int g{ static_cast<int>(255 * data[pixel + 2]) };

			image << r << ' ' << b << ' ' << g << '\n';
		}
	}

	image.close();
}

std::vector<float> demo::renderTestCPU(int width, int height, int channels)
{
	size_t bufferSize = width * height * channels;

	std::vector<float> data(bufferSize);

	for (int pIdx = 0; pIdx < width * height; ++pIdx)
	{
		int px = pIdx % width;
		int py = pIdx / width;

		data[3 * pIdx] = float(px) / width;
		data[3 * pIdx + 1] = float(py) / height;
		data[3 * pIdx + 2] = 0.2f;
	}

	return data;
}

void demo::StbiImageDeleter::operator()(void* image)
{
	stbi_image_free(image);
}
