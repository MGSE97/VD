/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: oa.h,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#ifndef __oa_h
    #define __oa_h

    #include <malloc.h>
    #include <assert.h>

    #include "def.h"
    #include "object.h"
    #include "matrix.h"
    #include "file.h"

       /* object array */
    typedef struct Oa {
       int count;
       OBJECT **list;
    } OA;


    OA *oaalloc(int n);
    
    void oafree( OA *);
    
    void oaput( OA *poa, int pos, OBJECT *pob);

    void oaprint( OA *poa);
    
    OBJECT *oaget( OA *poa, int pos );
    
    MATRIX *fastmap( OA *poa, int k);

    void pickAnchors(OA *poa, int *pa1, int *pa2, int dims, 
		     MATRIX *pmat);

    int Furthest( OA *poa, int i, int dims, MATRIX *pmat);

    NUMBER mydist2( OA *poa, int o1, int o2, int dims, MATRIX *pmat);

    //void printdist( OA *poa);
    void printdist2( OA * poa );

    OA *file2oa(FILE *);

#endif
