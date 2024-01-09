#pragma once

#include <vector>
#include <string>

namespace demo
{
	// Added png loading as well for completeness
	std::vector<float> loadPNG(const std::string& filepath, int& outWidth, int& outHeight, int& outChannels, int desiredChannels);

	// Added support to save as png since that works better for viewing and editing with the tools I use
	void storePNG(const std::string& filepath, const std::vector<float>& data, int width, int height, int channels);

	// The tutorial uses ppm to save images without needing a library (can be viewed/edited with GIMP on windows)
	void storePPM(const std::string& filepath, const std::vector<float>& data, int width, int height, int channels);

	// Creates image data for a simple color gradient to help debug save/load
	std::vector<float> renderTestCPU(int width, int height, int channels);

	// Used to allow a unique_ptr to correctly deallocate memory allocated by stbi_load (stbi_image_free)
	// since allocation and deallocation can be customized
	struct StbiImageDeleter { void operator()(void* image); };
}
