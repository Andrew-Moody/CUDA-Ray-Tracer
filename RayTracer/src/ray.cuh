#pragma once

#include <device_launch_parameters.h>
#include "Vec3.cuh"

class Ray
{
public:

	__host__ __device__ Ray() {}
	__host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin_{ origin }, direction_{ direction } {}

	__host__ __device__ Vec3 origin() const { return origin_; }
	__host__ __device__ Vec3 direction() const { return direction_; }
	__host__ __device__ Vec3 at(float t) const { return origin_ + direction_ * t; };

private:

	Vec3 origin_;
	Vec3 direction_;
};