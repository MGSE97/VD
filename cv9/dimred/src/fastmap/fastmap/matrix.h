/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: matrix.h,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#ifndef __matrix_h
    #define __matrix_h

    #include <malloc.h>
    #include <assert.h>
    
    #include "def.h"
    #include "vector.h"
    
       /* matrix */
    typedef struct Matrix {
       int count;
       VECTOR **list;
    } MATRIX;
    
    MATRIX *matrixalloc(int n);
    
    void matrixfree( MATRIX *);
    
    void matrixput( MATRIX *pmat, int pos, VECTOR *pvec);
    void matrix2put( MATRIX *pmatrix, int row, int column, NUMBER val);
    
    VECTOR *matrixget( MATRIX *pmat, int pos );
    NUMBER matrix2get( MATRIX *pmatrix, int row, int column);
    
    void matrixprint(MATRIX *pmatrix);

    void printvdist2(MATRIX *pmatrix);

#endif
