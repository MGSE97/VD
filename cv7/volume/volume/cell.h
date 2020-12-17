#ifndef CELL_H_
#define CELL_H_

#include "vector3.h"
#include "ray.h"
#include "aabb.h"

#define DATA_TYPE float
#define COMP_TYPE double

/*! \class Cell
\brief Implementation of a single hexahedral cell.

\author Tomáš Fabián
\version 1.0
\date 2011-2020
*/
class Cell
{
public:
	Cell( const DATA_TYPE * rhos, const Vector3 & A, const Vector3 & B );
	~Cell();
		
	float Gamma(const Vector3 & uvw) const;
	Vector3 GradGamma( const Vector3 & uvw ) const;
	
	float Integrate( Ray & ray, const float t0, const float t1 ) const;		
	float FindIsoSurface( Ray & ray, const float t0, const float t1, const float iso_value ) const;

	bool Cell::FrontToBack(Ray& ray, const float t0, const float t1, Vector3* sample_color, float* sample_alpha);

	Vector3 u( const Vector3 & p ) const;

	/*DATA_TYPE rho_A() const;
	DATA_TYPE rho_B() const;
	DATA_TYPE rho_C() const;
	DATA_TYPE rho_D() const;

	DATA_TYPE rho_E() const;
	DATA_TYPE rho_F() const;
	DATA_TYPE rho_G() const;
	DATA_TYPE rho_H() const;*/

	Vector3 A() const;
	Vector3 G() const;	

private:
	DATA_TYPE rhos_[4][4][4];	
	AABB bounds_;
};

#endif
