/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: main.c,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */
#define _CRT_SECURE_NO_WARNINGS

#include "def.h"
#include "oa.h"
#include "matrix.h"
#include "file.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define ERRACT	fprintf(stderr, \
"USAGE: %s [-v] [-d dimensions] filename\n", pgname) 

#define TRUE 1
#define FALSE 0

int main(int argc, char** argv)
{
	MATRIX* mat;
	FILE* fp;
	OA* a;
	//int i;
	//char* fname = (char*)"../../../data/ts.large";
	char* fname = (char*)"D:/Documents/Projekty/Škola/VD/cv9/plotly/commodities.csv";


	char * pgname = argv[0];
	int verbose = TRUE; /* default */
	int numdims = 2; /* default */

#ifdef RELEASE
	verbose = FALSE;
	while ( argc > 1 && argv[1][0] == '-' ) {
		switch ( argv[1][1] ) {
		case 'd':	/* set dimensions */
			if ( argc > 2 ) {
				numdims = ( int )atoi( argv[2] );
				assert( numdims > 0 );
				argc--; argv++;
			}
			else {
				ERRACT;
				exit( 1 );
			}
			break;
		case 'v':	/* set verbose flag */
			verbose = TRUE;
			break;
		default:
			fprintf( stderr, "%s: unknown flag %s - ignored\n",
				pgname, argv[1] );
		}
		argc--;
		argv++;
	}
	if ( argc != 2 ) {
		ERRACT;
		exit( 1 );
	}

	fname = argv[1];
#endif

	if ( verbose ) {
		printf( "working on %s \n", fname );
	}

	fp = fopen( fname, "r" );
	assert( fp != 0 );

	/* check for shorter/longer lines */
	if ( checkfile( fp ) == FALSE ) {
		fprintf( stderr, "*** illegal file - exiting\n" );
		exit( 0 );
	}

	a = file2oa( fp );

	if ( verbose ) {
		printf( "\n*********original matrix\n" );
		oaprint( a );
	}

	mat = fastmap( a, numdims );

	if ( verbose ) { printf( "\n*******FASTMAP\n" ); }

	matrixprint( mat );

	if ( verbose ) {
		printf( "\nTESTING DISTANCES - original\n" );
		printdist2( a );

		printf( "\nTESTING DISTANCES - after fastmap\n" );
		printvdist2( mat );
	}

	matrixfree( mat );
	oafree( a );

	exit( 0 ); /* to avoid complains from the makefile */
}
