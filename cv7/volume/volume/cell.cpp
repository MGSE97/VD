#include "stdafx.h"

#include "cell.h"
#include "matrix3x3.h"
#include "volume.h"

Cell::Cell( const DATA_TYPE * rhos, const Vector3 & A, const Vector3 & B )
{
	memcpy( rhos_, rhos, sizeof( rhos_ ) );
	//rhos_ = rhos;

	bounds_ = AABB( A, B );
}

Cell::~Cell()
{

}

float cubicInterpolate(const float p[4], float x) {
	return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

float bicubicInterpolate(const float p[4][4], float x, float y) {
	float arr[4];
	arr[0] = cubicInterpolate(p[0], y);
	arr[1] = cubicInterpolate(p[1], y);
	arr[2] = cubicInterpolate(p[2], y);
	arr[3] = cubicInterpolate(p[3], y);
	return cubicInterpolate(arr, x);
}

float tricubicInterpolate(const float p[4][4][4], float x, float y, float z) {
	float arr[4];
	arr[0] = bicubicInterpolate(p[0], y, z);
	arr[1] = bicubicInterpolate(p[1], y, z);
	arr[2] = bicubicInterpolate(p[2], y, z);
	arr[3] = bicubicInterpolate(p[3], y, z);
	return cubicInterpolate(arr, x);
}

//float GammaTL(const Vector3& uvw) const
//{
//	//return ( rho_A() + rho_B() + rho_C() + rho_D() + rho_E() + rho_F() + rho_G() + rho_H() ) / 8.0f;
//
//	const float& u_t = uvw.x;
//	const float& v_t = uvw.y;
//	const float& w_t = uvw.z;
//
//	// trilinear interpolation
//	const COMP_TYPE alpha_AB = rho_A() * (1 - u_t) + rho_B() * u_t;
//	const COMP_TYPE alpha_DC = rho_D() * (1 - u_t) + rho_C() * u_t;
//	const COMP_TYPE alpha_EF = rho_E() * (1 - u_t) + rho_F() * u_t;
//	const COMP_TYPE alpha_HG = rho_H() * (1 - u_t) + rho_G() * u_t;
//
//	const COMP_TYPE beta_0 = alpha_AB * (1 - v_t) + alpha_DC * v_t;
//	const COMP_TYPE beta_1 = alpha_EF * (1 - v_t) + alpha_HG * v_t;
//
//	return static_cast<DATA_TYPE>(beta_0 * (1 - w_t) + beta_1 * w_t);
//}

float Cell::Gamma(const Vector3& uvw) const
{

	// TODO try to add tricubic interpolation
	return tricubicInterpolate(rhos_, uvw.x, uvw.y, uvw.z);
}

Matrix3x3 Hp{
	1, 2, 1,
	2, 4, 2,
	1, 2, 1
},
Hz{
	0,0,0,
	0,0,0,
	0,0,0
},
Hn{
	-1, -2, -1,
	-2, -4, -2,
	-1, -2, -1
};

Vector3 Cell::GradGamma(const Vector3& uvw) const
{
	// TODO compute the gradient of the scalar field here (use finite central differences or Sobel–Feldman operator)
	// Sobel–Feldman operator
	float Gx = 0, Gy = 0, Gz = 0;
	for(int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++)
			for (int z = -1; z <= 1; z++)
			{
				auto v = uvw + Vector3(x, y, z);
				auto g = Gamma(v);
				Gx += x * Hp.get(y, z) * g;
				Gy += y * Hp.get(x, z) * g;
				Gz += z * Hp.get(x, y) * g;
			}

	return Vector3(Gx, Gy, Gz);
}

float Cell::Integrate(Ray& ray, const float t0, const float t1) const
{
	// TODO approximate computation of an integral of scalar values along the given segment of the ray
	auto ints = 0.0f;
	auto step = 0.1f;

	for (auto t = t0; t <= t1; t += step)
	{
		ints += Gamma(u(ray.eval(t))) * step;
	}

	return ints;
}

float Cell::FindIsoSurface(Ray& ray, const float t0, const float t1, const float iso_value) const
{
	// TODO find the parametric distance of the iso surface of the certain iso value along the given segment of the ray
	auto dst = -1.f;
	auto step = 0.1f;

	for (auto t = t0; t <= t1; t += step)
	{
		if (Gamma(u(ray.eval(t))) >= iso_value) {
			dst = t;
			break;
		}
	}

	return dst;
}

bool Cell::FrontToBack(Ray& ray, const float t0, const float t1, Vector3* sample_color, float* sample_alpha)
{
	auto step = 0.1f;
	auto flesh = 0.3f, bone = 0.50f;
	auto gin = sample_color->x, ain = *sample_alpha;
	bool stop = false;
	for (auto t = t0; t <= t1 && !stop; t += step)
	{
		auto g = Gamma(u(ray.eval(t)));
		auto a = 0.0035f;

		if (g < flesh) continue;
		else if (g > bone) {
			a = 0.3f;
		}

		gin = gin + (1 - ain) * a * g;
		ain = ain + (1 - ain) * a;

		stop = ain >= 1;
	}
	*sample_color = Vector3(gin, gin, gin);
	*sample_alpha = ain;
	return stop;
}

Vector3 Cell::u( const Vector3 & p ) const
{
	Vector3 uvw = ( p - A() ) / ( G() - A() ); // gives the reference coordinates of the world space point p inside this cell (uvw is in the range <0,1>^3)

	return uvw;
}

Vector3 Cell::A() const
{
	return bounds_.lower_bound();
}

Vector3 Cell::G() const
{
	return bounds_.upper_bound();
}

//float Cell::rho_A() const
//{
//	return rhos_[0];
//}
//
//float Cell::rho_B() const
//{
//	return rhos_[1];
//}
//
//float Cell::rho_C() const
//{
//	return rhos_[2];
//}
//
//float Cell::rho_D() const
//{
//	return rhos_[3];
//}
//
//float Cell::rho_E() const
//{
//	return rhos_[4];
//}
//
//float Cell::rho_F() const
//{
//	return rhos_[5];
//}
//
//float Cell::rho_G() const
//{
//	return rhos_[6];
//}
//
//float Cell::rho_H() const
//{
//	return rhos_[7];
//}
