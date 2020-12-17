#pragma once

#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <array>
#include <random>
#include <functional>

// opencv
#include <opencv2/opencv.hpp>

// omp
#include <omp.h>

#define REAL_MAX static_cast<float>( FLT_MAX )
#define REAL_MIN static_cast<float>( -FLT_MAX )
#define EPSILON static_cast<float>( 1e-5 )

#define DEG2RAD( x ) ( ( x ) * static_cast<float>( M_PI / 180 ) )
#define RAD2DEG( x ) ( ( x ) * static_cast<float>( 180 / M_PI ) )
#define SQR( x ) ( ( x ) * ( x ) )

//NaN-safe min/max functions that will return NaN only if the second argument b is NaN
#ifndef MIN
#define MIN( a, b ) ( ( a < b )? ( a ) : ( b ) )
#endif

#ifndef MAX
#define MAX( a, b ) ( ( a > b )? ( a ) : ( b ) )
#endif

#define _1_MPI 1.f / M_PI

#define SAFE_DELETE( p ) { \
	if ( p != NULL ) \
		{ \
	delete p; \
	p = NULL; \
	} \
}

#define SAFE_DELETE_ARRAY( p ) { \
	if ( p != NULL ) \
		{ \
		delete [] p; \
		p = NULL; \
		} \
}
