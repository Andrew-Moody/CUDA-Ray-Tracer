#pragma once

#include <vector>

namespace rtw
{
	class Camera;

	std::vector<float> renderCPU(const Camera& camera, int numSpheres);
}
