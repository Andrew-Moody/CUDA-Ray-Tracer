#pragma once

#include <vector>

namespace rtw
{
	class Camera;

	std::vector<float> renderGPU(const Camera& camera, int numSpheres);
}
