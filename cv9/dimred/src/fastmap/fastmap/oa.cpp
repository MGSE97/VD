/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: oa.c,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#include "oa.h"
#include <math.h>

OA *oaalloc( int n){
    OA *poa;
    //int i;

    assert( n > 0 );
    poa = (OA *) malloc( sizeof(OA) );
    assert( poa != NULL );
    poa-> list = (OBJECT **) malloc ( sizeof(OBJECT *) * n );
    assert( (poa->list) != NULL );

    poa->count = n;

    /* ******** DO NOT allocate space for objects ****
    for(i=0; i<count; i++){
        (poa->list)[i] = oballoc(m);
    }
    ******** */

    return(poa);
}

void oafree( OA *poa){
    int i;
    for(i=0; i< (poa->count); i++){
	obfree( (poa->list)[i] );
    }
    free( (char *) poa->list);
    poa->count =0;
    free( (char *) poa);
}

/* copies an object to the given position */
void oaput( OA *poa, int pos, OBJECT *pob){
    assert( pos >=0) ;
    assert( pos < (poa->count)  );
    (poa->list)[pos] = obcopy(pob);
}

OBJECT *oaget( OA *poa, int pos){
    assert( pos >=0) ;
    assert( pos < (poa->count)  );
    return( (poa->list)[pos] );
}

void oaprint( OA *poa){
    int i, n;
    assert( poa->count > 0 );
    n = poa->count;
    for(i=0; i<n; i++){
	obprint( oaget( poa, i ) );
    }
}

/********************************************
   given an array of objects `*poa'
   and a desirable dimensionality `k'
   it returns an n*k matrix with coordinates
   of points, whose pairwise distances approximate
   the distances of the objects of `*poa'
 ***********************************************/

MATRIX *fastmap( OA *poa, int k){
   int n = poa->count;
   int i, a1, a2;
   int dims;
   NUMBER d12sq, d1xsq, d2xsq, x;
   MATRIX *pmat;
   VECTOR *pvec;

   pmat = matrixalloc(n);

   /* blank out *pmat */
   pvec = vecalloc( k );
   for( i=0; i<k; i++){
       /* blank out vector */
       vecput( pvec, i, 0.0);
   }
   for(i=0; i<n; i++){
       /* fill up matrix */
       matrixput( pmat, i, pvec);
   }
   vecfree(pvec);
   /* SHOULD MAKE THE ABOVE INTO  A FUNCTION */

   for( dims=0; dims<k; dims++){

       /* pick two remote objects */
       pickAnchors( poa, &a1, &a2, dims, pmat);

       /* compute projections of each object
	  on the axis defined by the two anchors */

       d12sq = mydist2( poa, a1, a2, dims, pmat);
       if( d12sq == 0.0 ) { return(pmat); }
	   /* we've exceeded the dimensionality of the data space */

       for( i=0; i<n; i++){
	   d1xsq = mydist2( poa, i, a1, dims, pmat);
	   d2xsq = mydist2( poa, i, a2, dims, pmat);
	   x = (d1xsq + d12sq - d2xsq) / ( 2.0 * sqrt( (double) d12sq ) );
	   matrix2put( pmat, i, dims, x);
       } /* end for i */
   } /* end for dims */

   return( pmat );
}

/* choose to far-away objects */
void pickAnchors( OA * poa, int *pa1, int *pa2, int dims, MATRIX *pmat){
    int i,j;

    i=0;
    j = Furthest( poa, i, dims, pmat);
    i = Furthest( poa, j, dims, pmat);

    *pa1 = i;
    *pa2 = j;
    return;
}

/* for a given object 'i', returns the index of the furthest one */
int Furthest( OA *poa, int icurrent, int dims, MATRIX *pmat){
    NUMBER max = 0.0;
    int furthest= icurrent;
    int n = poa->count;
    int j;

    for(j=0; j<n; j++){
	if (mydist2( poa, icurrent, j, dims, pmat) > max){
	   max = mydist2( poa, icurrent, j, dims, pmat);
	   furthest = j;
        }
    }
    return(furthest);
}

/* squared distance of two objects, after dims-1 projections */
NUMBER mydist2( OA *poa, int o1, int o2, int dims, MATRIX *pmat){

    OBJECT *p1 = oaget( poa, o1);
    OBJECT *p2 = oaget( poa, o2);
    NUMBER res2 = obdist2( p1, p2);

    /* decrease it, by the projections */
    int i;
    for(i=0; i< dims; i++){
	NUMBER temp = matrix2get(pmat, o1, i) - matrix2get(pmat, o2, i);
	res2 -=  temp*temp;
    }
    return( res2);

}

/* prints the  squared distances of all pairs of elements */
void printdist2(OA *poa){
    int i, j, n;
    n = poa->count;
    for(i=0; i< n; i++){
	for(j=0; j<n; j++){
	    printf("%g\t", obdist2( oaget(poa, i), oaget(poa, j) ));
	}
	printf("\n");
    }
}

/* reads in a file, into an object array */
OA *file2oa(FILE *fp){
    OA *poa;
    OBJECT *pob;
    int i, n;
    char *line;

    myrewind(fp); 
    n = countlines(fp);

    assert( n>0); /* no empty file */

    poa = oaalloc(n);
    assert( poa != NULL);

    myrewind(fp);
    i=0;
    line = getline(fp);
    while( !feof(fp) ){
	pob = line2ob(line);
	oaput( poa, i, pob);
	obfree(pob);

	i++;
	line = getline(fp);
    }

    assert( i == n);
    for(i=0; i<n; i++){
	/*check the lengths of the objects for equality */
    }

    return(poa);
}
