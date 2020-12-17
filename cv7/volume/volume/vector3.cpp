#include "stdafx.h"

#include "vector3.h"

float Vector3::L2Norm() const
{
	return sqrt( SQR( x ) + SQR( y ) + SQR( z ) );
}

float Vector3::SqrL2Norm() const
{
	return SQR( x ) + SQR( y ) + SQR( z );
}

void Vector3::Normalize()
{
	const float norm = SqrL2Norm();

	if ( norm != 0 )
	{
		const float rn = 1 / sqrt( norm );

		x *= rn;
		y *= rn;
		z *= rn;
	}
}

Vector3 Vector3::CrossProduct( const Vector3 & v )
{
	return Vector3(
		y * v.z - z * v.y,
		z * v.x - x * v.z,
		x * v.y - y * v.x );
}

float Vector3::DotProduct( const Vector3 & v ) const
{
	return x * v.x + y * v.y + z * v.z;
}

char Vector3::LargestComponent( const bool absolute_value )
{
	const Vector3 d = ( absolute_value )? Vector3( abs( x ), abs( y ), abs( z ) ) : *this;

	if ( d.x > d.y )
	{
		if ( d.x > d.z )
		{			
			return 0 ;
		}
		else
		{
			return 2;
		}
	}
	else
	{
		if ( d.y > d.z )
		{
			return 1;
		}
		else
		{
			return 2;
		}
	}

	return -1;
}

char Vector3::SmallestComponent( const bool absolute_value )
{
	Vector3 d = ( absolute_value )? Vector3( abs( x ), abs( y ), abs( z ) ) : *this;

	if ( std::isnan( d.x ) ) d.x = REAL_MAX;
	if ( std::isnan( d.y ) ) d.y = REAL_MAX;
	if ( std::isnan( d.z ) ) d.z = REAL_MAX;

	if ( d.x <= d.y )
	{
		if ( d.x <= d.z )
		{			
			return 0 ;
		}
		else
		{
			return 2;
		}
	}
	else
	{
		if ( d.y <= d.z )
		{
			return 1;
		}
		else
		{
			return 2;
		}
	}

	return -1;
}

Vector3 Vector3::Abs() const
{
	return Vector3( fabs( x ), fabs( y ), fabs( z ) );
}

void Vector3::Print()
{
	printf( "(%0.3f, %0.3f, %0.3f)\n", x, y, z );
}

// --- operátory ------

Vector3 operator-( const Vector3 & v )
{
	return Vector3( -v.x, -v.y, -v.z );
}

Vector3 operator+( const Vector3 & u, const Vector3 & v )
{
	return Vector3( u.x + v.x, u.y + v.y, u.z + v.z );
}

Vector3 operator-( const Vector3 & u, const Vector3 & v )
{
	return Vector3( u.x - v.x, u.y - v.y, u.z - v.z );
}

Vector3 operator*( const Vector3 & v, const float a )
{
	return Vector3( a * v.x, a * v.y, a * v.z );
}

Vector3 operator*( const float a, const Vector3 & v )
{
	return Vector3( a * v.x, a * v.y, a * v.z  ); 		
}

Vector3 operator*( const Vector3 & u, const Vector3 & v )
{
	return Vector3( u.x * v.x, u.y * v.y, u.z * v.z );
}

Vector3 operator/( const Vector3 & v, const float a )
{
	return v * ( 1 / a );
}

Vector3 operator/( const Vector3 & u, const Vector3 & v )
{
	return Vector3( u.x / v.x, u.y / v.y, u.z / v.z );
}

Vector3 operator/( const float a, const Vector3 & u )
{
	return Vector3( a / u.x, a / u.y, a / u.z );
}

void operator+=( Vector3 & u, const Vector3 & v )
{
	u.x += v.x;
	u.y += v.y;	
	u.z += v.z;	
}

void operator-=( Vector3 & u, const Vector3 & v )
{
	u.x -= v.x;
	u.y -= v.y;
	u.z -= v.z;
}

void operator*=( Vector3 & v, const float a )
{
	v.x *= a;
	v.y *= a;
	v.z *= a;
}

void operator/=( Vector3 & v, const float a )
{
	const float r = 1 / a;

	v.x *= r;
	v.y *= r;
	v.z *= r;
}
