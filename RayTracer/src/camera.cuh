#pragma once


#include "vec3.cuh"
#include "ray.cuh"

namespace rtw
{
	class Camera
	{
	public:

		__host__ __device__ Camera() {}

		__host__ __device__ Camera(int screenWidth, int screenHeight, int nChannels, int nSamples, int nBounces)
			: screenWidth_{ screenWidth }, screenHeight_{ screenHeight }, nChannels_{ nChannels },
			nSamples_{ nSamples }, nBounces_{ nBounces } {}

		__host__ __device__ int totalPixels() const { return totalPixels_; }

		__host__ __device__ int nChannels() const { return nChannels_; }

		__host__ __device__ int nSamples() const { return nSamples_; }

		__host__ __device__ int nBounces() const { return nBounces_; }

		__host__ __device__ Vec3 cameraPosition() const { return cameraPos_; }
		__host__ __device__ Vec3 viewOrigin() const { return pixelOrigin_; }

		__host__ __device__ float deltaU() const { return deltaU_; }
		__host__ __device__ float deltaV() const { return deltaV_; }

		__host__ __device__ Vec3 normalizedScreenToWorld(float x, float y) const
		{
			return Vec3{ x * viewWidth_ + viewOrigin_.x(), y * -viewHeight_ + viewOrigin_.y(), viewOrigin_.z() };
		}

		// Convenient if iterating over a buffer with two loops
		__host__ __device__ Ray getRayFromScreenPos(int x, int y) const
		{
			Vec3 pixelPos{ pixelOrigin_ + Vec3(x * deltaU_, -y * deltaV_, 0.0f) };
			Ray ray(cameraPos_, pixelPos - cameraPos_);
			return ray;
		}

		// Best for iterating over a buffer with a single loop (or striding on gpu)
		__host__ __device__ Ray getRayFromPixelIndex(int index) const
		{
			// Direct method to get screen coordinates from pixel index 
			//int x = index % screenWidth_;
			//int y = index / screenWidth_;

			//// pixel position in normalized screen coordinates
			//float px = x / screenWidth_;
			//float py = y / screenHeight_;

			//// Convert to world space coordinates
			//// Take care that world y decreases as screen y increases
			//Vec3 pixelPos{ px * viewWidth_, -py * viewHeight_, 0.0f };
			//pixelPos += pixelOrigin_;

			// pixel offsets in world units (avoiding extra divisions)
			// float px = static_cast<float>(index % screenWidth_) * deltaU_;
			// float py = static_cast<float>(index / screenWidth_) * deltaV_;

			// may want to try potential optimizations to avoid division/modulo
			// and/or experiment with math intrinsics
			// cache rw = 1/width, rh = 1/height before rendering a frame ( and sx = viewWidth/width, sy = viewHeight/height )
			// get back the screen space coords from the pixel index
			// y = "floor"( index * rw ), x = index - y * w
			// index will not be negative so floor and static_cast<int>() should both work
			// with casting allegedly faster for cpu but no idea for gpu with intrinsics
			// scale from screen space to world space with
			// px = x * sx, py = y * sy

			// pixel offsets in world units (avoiding division entirely)
			int y = static_cast<int>(index * recipWidth_);
			float px = (index - y * screenWidth_) * deltaU_;
			float py = y * deltaV_;

			// Get the world space postition of the pixel
			Vec3 pixelPos{ px, -py, 0.0f };
			pixelPos += pixelOrigin_;
			

			Ray ray(cameraPos_, pixelPos - cameraPos_);
			return ray;
		}

		__host__ __device__ Vec3 getWorldPosFromPixelIndex(int index) const
		{
			// pixel offsets in world units (avoiding division entirely)
			int y = static_cast<int>(index * recipWidth_);
			float px = (index - y * screenWidth_) * deltaU_;
			float py = y * deltaV_;

			// Get the world space postition of the pixel
			return Vec3{ pixelOrigin_.x() + px, pixelOrigin_.y() -py, pixelOrigin_.z() };
		}

	private:

		// Screen Properties
		int screenWidth_{ 800 };
		int screenHeight_{ 600 };
		int totalPixels_{ screenWidth_ * screenHeight_ };
		float recipWidth_{ 1.0f / screenWidth_ };

		// Sample Properties
		int nChannels_{ 3 };
		int nSamples_{ 1 };
		int nBounces_{ 5 };

		// Camera Properties
		Vec3 cameraPos_{ 0.0f, 0.0f, 0.0f };
		Vec3 cameraForward_{ 0.0f, 0.0f, -1.0f }; // Camera points in negative z direction
		Vec3 cameraUp_{ 0.0f, 1.0f, 0.0f };
		Vec3 cameraRight_{ 1.0f, 0.0f, 0.0f };

		float aspectRatio_{ static_cast<float>(screenWidth_) / screenHeight_ };
		float focalLength_{ 1.0f }; // Distance from Camera to viewport

		// Viewport Properties

		// Size of the viewport (unclear what units, for now assumed to be world units)
		float viewHeight_{ 2.0f };
		float viewWidth_{ viewHeight_ * aspectRatio_ };

		// Distance between pixels in viewport units (ideally should be equal)
		// Also acts as a scale factor between screen space and view space (and world space if viewport dimensions are world space)
		float deltaU_{ viewWidth_ / screenWidth_ };
		float deltaV_{ viewHeight_ / screenHeight_ };

		Vec3 viewCenter_{ cameraForward_ * focalLength_ }; // Center of the screen in worldspace
		Vec3 viewOrigin_{ viewCenter_ + 0.5f * ((cameraUp_ * viewHeight_) - (cameraRight_ * viewWidth_)) }; // Topleft of screen in worldspace
		Vec3 pixelOrigin_{ viewOrigin_ + 0.5f * Vec3(deltaU_, -deltaV_, 0.0f) }; // Offset viewOrigin to the center of the first pixel
	};
}