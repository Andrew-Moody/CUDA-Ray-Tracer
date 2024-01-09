#pragma once

#include <iostream>
#include <cmath>

#include <device_launch_parameters.h>

class Vec3
{
public:

	__host__ __device__ Vec3() : values{0.0f, 0.0f, 0.0f } {}

	__host__ __device__ Vec3(float x, float y, float z) : values{x, y, z } {}

	__host__ __device__ float x() const { return values[0]; }
	__host__ __device__ float y() const { return values[1]; }
	__host__ __device__ float z() const { return values[2]; }
	__host__ __device__ float operator[](int i) const { return values[i]; }
	__host__ __device__ float& operator[](int i) { return values[i]; }


	__host__ __device__ Vec3 operator-() const
	{
		return Vec3(-values[0], -values[1], -values[2]);
	}

	__host__ __device__ Vec3& operator+=(const Vec3& other)
	{
		values[0] += other.values[0];
		values[1] += other.values[1];
		values[2] += other.values[2];
		return *this;
	}

	__host__ __device__ Vec3& operator*=(float t)
	{
		values[0] *= t;
		values[1] *= t;
		values[2] *= t;
		return *this;
	}


	__host__ __device__ float length_squared() const
	{
		return (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
	}

	__host__ __device__ float length() const
	{
		return std::sqrt(length_squared());
	}

	friend float dot(const Vec3& a, const Vec3& b);

	friend Vec3 cross(const Vec3& a, const Vec3& b);

private:

	float values[3];
};


__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b)
{ 
	Vec3 result{ a };
	return result += b;
}

inline Vec3 operator-(const Vec3& a, const Vec3& b)
{
	Vec3 result{ -b };
	result += a;
	return result;
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t)
{
	Vec3 result{ v };
	return result *= t;
}

inline Vec3 operator*(float t, const Vec3& v)
{
	return v * t;
}

inline Vec3 normalized(const Vec3& v)
{
	Vec3 vec{ v };
	return vec *= (1.0f / v.length());
}

inline float dot(const Vec3& a, const Vec3& b)
{
	return (a.values[0] * b.values[0] + a.values[1] * b.values[1] + a.values[2] * b.values[2]);
}

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
	return Vec3(
		a.values[1] * b.values[2] - a.values[2] * b.values[1],
		a.values[2] * b.values[0] - a.values[0] * b.values[2],
		a.values[0] * b.values[1] - a.values[1] * b.values[0]
	);
}

inline std::ostream& operator<<(std::ostream& ostr, const Vec3& v)
{
	ostr << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
	return ostr;
}