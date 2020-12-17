/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: matrix.c,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#include "matrix.h"

MATRIX *matrixalloc( int n){
    MATRIX *pmatrix;
    //int i;

    assert( n > 0 );
    pmatrix = (MATRIX *) malloc( sizeof(MATRIX) );
    assert( pmatrix != NULL );
    pmatrix-> list = (VECTOR **) malloc ( sizeof(VECTOR *) * n );
    assert( (pmatrix->list) != NULL );

    pmatrix->count = n;

    /* ******** DO NOT allocate space for vec ****
    for(i=0; i<count; i++){
        (pmatrix->list)[i] = vecalloc(m);
    }
    ******** */

    return(pmatrix);
}

void matrixfree( MATRIX *pmatrix){
    int i;
    for(i=0; i< (pmatrix->count); i++){
	vecfree( (pmatrix->list)[i] );
    }
    free( (char *) pmatrix->list);
    pmatrix->count =0;
    free( (char *) pmatrix);
}

void matrixput( MATRIX *pmatrix, int pos, VECTOR *pvec){
    assert( pos >=0) ;
    assert( pos < (pmatrix->count)  );
    (pmatrix->list)[pos] = veccopy(pvec);
}

void matrix2put( MATRIX *pmatrix, int row, int column, NUMBER val){
    VECTOR *pvec;
    assert( row >=0);
    assert( row < (pmatrix->count)  );
    pvec = (pmatrix->list)[row];
    vecput( pvec, column, val);
}

VECTOR *matrixget( MATRIX *pmatrix, int pos){
    assert( pos >=0) ;
    assert( pos < (pmatrix->count)  );
    return( (pmatrix->list)[pos] );
}

NUMBER matrix2get( MATRIX *pmatrix, int row, int column){
    VECTOR *pvec;
    NUMBER val;
    assert( row >=0);
    assert( row < (pmatrix->count)  );
    pvec = (pmatrix->list)[row];
    val = vecget( pvec, column);
    return(val);
}

void matrixprint( MATRIX *pmatrix){
    int i, n;
    if( pmatrix == NULL ){
	printf("empty matrix\n");
	return;
    }
    assert( pmatrix->count >0 );
    n = pmatrix->count;
    for(i=0; i<n; i++){
	vecprint( matrixget( pmatrix, i) );
    }
}

/* prints pairwise squared distances */
void printvdist2(MATRIX *pmat){
    int i, j, n;

    n = pmat->count;
    for(i=0; i<n; i++){
	for(j=0; j<n; j++){
	    printf("%g\t", vecdist2( matrixget(pmat, i),
				     matrixget(pmat, j) ) );
	}
	printf("\n");
    }
}
