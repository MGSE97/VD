#include "stdafx.h"

#include "camera.h"
#include "volume.h"

int main( int argc, char * argv[] )
{
	printf( "VD, (c)2015-2020 Tomas Fabian\n\n" );

	// Test 1
	Volume volume( 512, 512, 150, Vector3( 1, 1, 1 ) ); // width x height x #slices
	volume.Load( std::string( "../../data/female_ankle/slice%03d.png" ), 1, 150 ); // 0, #slices-1

	// Test 2
	/*const int n = 512;
	Volume volume( n, n, n, Vector3( 500.0f / n, 500.0f / n, 500.0f / n ) );
	volume.Generate();*/

	Camera camera = Camera( Vector3( 500.0f, 400.0f, 1300.0f ) * 0.3f, Vector3( 0.0f, 0.0f, 0.0f ), 640, 480, DEG2RAD( 40.0f ) );

	volume.Raycast( camera, 1); // with super sampling

	return 0;
}
