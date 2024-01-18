#pragma once


#include "vec3.cuh"
#include "ray.cuh"
#include "random.cuh"

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

		/*__host__ __device__ float deltaU() const { return deltaU_; }
		__host__ __device__ float deltaV() const { return deltaV_; }*/

		__host__ void setFocus(float fov, float focalLength, float fNumber = 1000000.0f)
		{
			fov_ = fov;
			focalLength_ = focalLength; // Distance from Camera to viewport
			fNumber_ = fNumber;
			lensBasisU_ = cameraRight_ * (focalLength * 0.5f / fNumber_);
			lensBasisV_ = cameraUp_ * (focalLength * 0.5f / fNumber_);

			updateView();
		}

		__host__ void setTransform(float x, float y, float z, float lx, float ly, float lz)
		{
			/*cameraPos_ = { x, y, z };
			cameraForward_ = { lx, ly, lz };
			cameraForward_.normalize();*/

			cameraPos_ = { x, y, z };
			Vec3 lookat{ lx, ly, lz };

			Vec3 lookVec = lookat - cameraPos_;
			focalLength_ = lookVec.length();
			cameraForward_ = normalized(lookVec);

			// Assume up direction is near global up
			// if forward is near vertical use +/- global forward instead
			Vec3 upGuess{ 0.0f, 1.0f, 0.0f };

			float d = dot(cameraForward_, upGuess);

			if (d > 0.9f)
			{
				upGuess = { 0.0f, 0.0f, -1.0f };
			}
			else if (d < -0.9f)
			{
				upGuess = { 0.0f, 0.0f, 1.0f };
			}

			// Get unit vector orthogonal to forward and up
			cameraRight_ = cross(cameraForward_, upGuess);
			cameraRight_.normalize();

			// Get the true up direction orthogonal to forward and right
			cameraUp_ = cross(cameraRight_, cameraForward_);

			//// Get unit vector orthogonal to forward and up
			//cameraRight_ = cross(upGuess, cameraForward_);
			//cameraRight_.normalize();

			//// Get the true up direction orthogonal to forward and right
			//cameraUp_ = cross(cameraForward_, cameraRight_);

			updateView();
		}

		__host__ void updateView()
		{
			viewHeight_ = 2.0f * focalLength_ * std::tan(fov_ * static_cast<float>(std::_Pi) / 360.0f);
			viewWidth_ = viewHeight_ * aspectRatio_;

			/*deltaU_ = viewWidth_ / screenWidth_;
			deltaV_ = viewHeight_ / screenHeight_;

			viewBasisU_ = deltaU_ * cameraRight_;
			viewBasisV_ = deltaV_ * cameraUp_;*/

			viewBasisU_ = viewWidth_ / screenWidth_ * cameraRight_;
			viewBasisV_ = viewHeight_ / screenHeight_ * cameraUp_;

			viewCenter_ = cameraPos_ + cameraForward_ * focalLength_;
			viewOrigin_ = viewCenter_ + 0.5f * ((cameraUp_ * viewHeight_) - (cameraRight_ * viewWidth_));
			pixelOrigin_ = viewOrigin_ + 0.5f * (viewBasisU_ - viewBasisV_);
		}

		__host__ __device__ Vec3 normalizedScreenToWorld(float x, float y) const
		{
			return Vec3{ x * viewWidth_ + viewOrigin_.x(), y * -viewHeight_ + viewOrigin_.y(), viewOrigin_.z() };
		}

		//// Convenient if iterating over a buffer with two loops
		//__host__ __device__ Ray getRayFromScreenPos(int x, int y) const
		//{
		//	Vec3 pixelPos{ pixelOrigin_ + Vec3(x * deltaU_, -y * deltaV_, 0.0f) };
		//	Ray ray(cameraPos_, pixelPos - cameraPos_);
		//	return ray;
		//}

		// Best for iterating over a buffer with a single loop (or striding on gpu)
		//__host__ __device__ Ray getRayFromPixelIndex(int index) const
		//{
		//	// Direct method to get screen coordinates from pixel index 
		//	//int x = index % screenWidth_;
		//	//int y = index / screenWidth_;

		//	//// pixel position in normalized screen coordinates
		//	//float px = x / screenWidth_;
		//	//float py = y / screenHeight_;

		//	//// Convert to world space coordinates
		//	//// Take care that world y decreases as screen y increases
		//	//Vec3 pixelPos{ px * viewWidth_, -py * viewHeight_, 0.0f };
		//	//pixelPos += pixelOrigin_;

		//	// pixel offsets in world units (avoiding extra divisions)
		//	// float px = static_cast<float>(index % screenWidth_) * deltaU_;
		//	// float py = static_cast<float>(index / screenWidth_) * deltaV_;

		//	// may want to try potential optimizations to avoid division/modulo
		//	// and/or experiment with math intrinsics
		//	// cache rw = 1/width, rh = 1/height before rendering a frame ( and sx = viewWidth/width, sy = viewHeight/height )
		//	// get back the screen space coords from the pixel index
		//	// y = "floor"( index * rw ), x = index - y * w
		//	// index will not be negative so floor and static_cast<int>() should both work
		//	// with casting allegedly faster for cpu but no idea for gpu with intrinsics
		//	// scale from screen space to world space with
		//	// px = x * sx, py = y * sy

		//	// pixel offsets in world units (avoiding division entirely)
		//	int y = static_cast<int>(index * recipWidth_);
		//	float px = (index - y * screenWidth_) * deltaU_;
		//	float py = y * deltaV_;

		//	// Get the world space postition of the pixel
		//	Vec3 pixelPos{ px, -py, 0.0f };
		//	pixelPos += pixelOrigin_;
		//	

		//	Ray ray(cameraPos_, pixelPos - cameraPos_);
		//	return ray;
		//}

		__host__ __device__ Vec3 getWorldPosFromPixelIndex(int index) const
		{
			// Truncate the result of "dividing" index by number of pixels in a screen row
			float y{ static_cast<float>(static_cast<int>(index * recipWidth_)) };
			float x{ index - y * screenWidth_ };
			return pixelOrigin_ + (x * viewBasisU_) - (y * viewBasisV_);
		}


		//__host__ __device__ Point2D getPixelCoordFromIndex(int index) const
		//{
		//	// Truncate the result of "dividing" index by number of pixels in a screen row
		//	float y{ static_cast<float>(static_cast<int>(index * recipWidth_)) };
		//	float x{ index - y * screenWidth_ };
		//	return Point2D{ x, y };
		//}

		// Add a screen space shift to a world position (for multisampling)
		__host__ __device__ Vec3 shiftWorldPos(Vec3 worldPos, float shiftX, float shiftY) const
		{
			// +/- should not matter for random sampling purposes
			//return worldPos + (shiftX * viewBasisU_) - (shiftY * viewBasisV_);
			return worldPos + (shiftX * viewBasisU_) + (shiftY * viewBasisV_);
		}

		__host__ __device__ Vec3 getSampleOrigin(uint32_t& pcgState) const
		{
			float shiftX{};
			float shiftY{};

			for (;;)
			{
				shiftX = 2.0f * randomFloat(pcgState) - 1.0f;
				shiftY = 2.0f * randomFloat(pcgState) - 1.0f;

				if (shiftX * shiftX + shiftY * shiftY <= 1.0f)
				{
					// dont normalize want inside unit disk not on edge
					//shift.normalize();
					break;
				}
			}			

			return cameraPos_ + (shiftX * lensBasisU_) + (shiftY * lensBasisV_);
		}

		__host__ __device__ Ray generateRay(Vec3 worldPos, uint32_t& pcgState) const
		{
			Vec3 origin = getSampleOrigin(pcgState);

			Vec3 samplePos = shiftWorldPos(worldPos, randomFloat(pcgState) - 0.5f, randomFloat(pcgState) - 0.5f);

			samplePos -= origin;
			samplePos.normalize();


			return { origin, samplePos };
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
		float fov_{ 90.0f };
		float focalLength_{ 1.0f }; // Distance from Camera to viewport
		float fNumber_{ 1000000.0f }; // book uses angle but this avoids atan and is a common camera parameter (f-number)
		Vec3 lensBasisU_{ cameraRight_ * (2.0f / fNumber_) };
		Vec3 lensBasisV_{ cameraUp_ * (2.0f / fNumber_) };

		// Viewport Properties
		
		// Size of the viewport (unclear what units, for now assumed to be world units)
		float viewHeight_{ 2.0f * focalLength_ * std::tan(fov_ * static_cast<float>(std::_Pi) / 360.0f) };
		float viewWidth_{ viewHeight_ * aspectRatio_ };

		// Distance between pixels in viewport units (ideally should be equal)
		// Also acts as a scale factor between screen space and view space (and world space if viewport dimensions are world space)
		//float deltaU_{ viewWidth_ / screenWidth_ };
		//float deltaV_{ viewHeight_ / screenHeight_ };

		//// pre multiply camera basis vectors by pixel deltas
		//Vec3 viewBasisU_{ deltaU_ * cameraRight_ };
		//Vec3 viewBasisV_{ deltaV_ * cameraUp_ };

		Vec3 viewBasisU_{ viewWidth_ / screenWidth_ * cameraRight_ };
		Vec3 viewBasisV_{ viewHeight_ / screenHeight_ * cameraUp_ };

		Vec3 viewCenter_{ cameraPos_ + cameraForward_ * focalLength_ }; // Center of the screen in worldspace
		Vec3 viewOrigin_{ viewCenter_ + 0.5f * ((cameraUp_ * viewHeight_) - (cameraRight_ * viewWidth_)) }; // Topleft of screen in worldspace
		Vec3 pixelOrigin_{ viewOrigin_ + 0.5f * (viewBasisU_ - viewBasisV_) };// Offset viewOrigin to the center of the first pixel
	};
}