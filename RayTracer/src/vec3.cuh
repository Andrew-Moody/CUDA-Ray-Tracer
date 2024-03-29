#pragma once

#include <iostream>
#include <cmath>
#include <device_launch_parameters.h>

namespace rtw
{
	struct Point2D
	{
		float x;
		float y;
	};

	class Vec3
	{
	public:

		__host__ __device__ Vec3() : values{ 0.0f, 0.0f, 0.0f } {}

		__host__ __device__ Vec3(float x, float y, float z) : values{ x, y, z } {}

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

		__host__ __device__ Vec3& operator-=(const Vec3& other)
		{
			values[0] -= other.values[0];
			values[1] -= other.values[1];
			values[2] -= other.values[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(float t)
		{
			values[0] *= t;
			values[1] *= t;
			values[2] *= t;
			return *this;
		}

		__host__ __device__ Vec3& operator/=(float t)
		{
			float d = 1.0f / t;
			values[0] *= d;
			values[1] *= d;
			values[2] *= d;
			return *this;
		}

		// element wise multiplication (nice for color attenuation)
		__host__ __device__ Vec3& operator*=(const Vec3& other)
		{
			values[0] *= other.values[0];
			values[1] *= other.values[1];
			values[2] *= other.values[2];
			return *this;
		}

		__host__ __device__ float length_squared() const
		{
			return (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
		}

		__host__ __device__ float length() const
		{
			return sqrtf(length_squared());
		}

		__host__ __device__ Vec3& normalize()
		{
			*this *= (1.0f / sqrtf(length_squared()));
			return *this;
		}


		__host__ __device__ friend Vec3 operator-(const Vec3& a, const Vec3& b);

		__host__ __device__ friend float dot(const Vec3& a, const Vec3& b);

		__host__ __device__ friend Vec3 cross(const Vec3& a, const Vec3& b);

		__host__ __device__ friend Vec3 at(const Vec3& a, const Vec3& b, float t);

		/*__host__ __device__ friend float dot(Vec3 a, Vec3 b);

		__host__ __device__ friend Vec3 cross(Vec3 a, Vec3 b);*/

		/*__host__ __device__ float dot(const Vec3& b)
		{
			return (values[0] * b.values[0] + values[1] * b.values[1] + values[2] * b.values[2]);
		}*/

		__host__ __device__ float dot(Vec3 b)
		{
			return (values[0] * b.values[0] + values[1] * b.values[1] + values[2] * b.values[2]);
		}

	private:

		float values[3];
	};

	__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b)
	{
		Vec3 result{ a };
		return result += b;
	}

	__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b)
	{
		Vec3 result{ a };
		return result -= b;

		//return Vec3{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, float t)
	{
		Vec3 result{ v };
		return result *= t;
	}

	__host__ __device__ inline Vec3 operator*(float t, const Vec3& v)
	{
		return v * t;
	}

	__host__ __device__ inline Vec3 normalized(const Vec3& v)
	{
		Vec3 vec{ v };
		return vec *= (1.0f / v.length());
	}

	__host__ __device__ inline float dot(const Vec3& a, const Vec3& b)
	{
		return (a.values[0] * b.values[0] + a.values[1] * b.values[1] + a.values[2] * b.values[2]);
	}

	__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b)
	{
		return Vec3(
			a.values[1] * b.values[2] - a.values[2] * b.values[1],
			a.values[2] * b.values[0] - a.values[0] * b.values[2],
			a.values[0] * b.values[1] - a.values[1] * b.values[0]
		);
	}

	__host__ __device__ inline Vec3 at(const Vec3& a, const Vec3& b, float t)
	{
		return Vec3{ a[0] + b[0] * t, a[1] + b[1] * t, a[2] + b[2] * t };
	}


	inline std::ostream& operator<<(std::ostream& ostr, const Vec3& v)
	{
		ostr << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
		return ostr;
	}


	// By value

	/*__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b)
	{
		Vec3 result{ a };
		return result += b;
	}

	__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b)
	{
		Vec3 result{ a };
		return result -= b;
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, float t)
	{
		Vec3 result{ v };
		return result *= t;
	}

	__host__ __device__ inline Vec3 operator*(float t, const Vec3& v)
	{
		return v * t;
	}

	__host__ __device__ inline Vec3 normalized(const Vec3& v)
	{
		Vec3 vec{ v };
		return vec *= (1.0f / v.length());
	}

	__host__ __device__ inline float dot(Vec3 a, Vec3 b)
	{
		return (a.values[0] * b.values[0] + a.values[1] * b.values[1] + a.values[2] * b.values[2]);
	}

	__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b)
	{
		return Vec3(
			a.values[1] * b.values[2] - a.values[2] * b.values[1],
			a.values[2] * b.values[0] - a.values[0] * b.values[2],
			a.values[0] * b.values[1] - a.values[1] * b.values[0]
		);
	}*/
}
