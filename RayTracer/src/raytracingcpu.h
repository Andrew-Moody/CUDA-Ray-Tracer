#pragma once

#include <vector>

namespace rtw
{
	class Camera;
	class Scene;

	std::vector<float> renderCPU(const Camera& camera, const Scene& scene);
}
