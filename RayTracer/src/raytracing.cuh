#pragma once

#include <vector>

namespace rtw
{
	class Camera;
	class Scene;

	std::vector<float> renderGPU(const Camera& camera, const Scene& scene);
}
