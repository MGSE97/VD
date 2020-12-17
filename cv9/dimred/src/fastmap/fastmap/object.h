/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: object.h,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */


#ifndef __object_h
    #define __object_h
#define _CRT_SECURE_NO_WARNINGS
    #include <malloc.h>
    #include <assert.h>
    #include <string.h>
    
    #include "def.h"
    
    
    /* 1-d vector, with size */
    typedef struct Object { 
       int len;
       NUMBER *vec;
    } OBJECT;
    
    
    
    OBJECT *oballoc(int n);
    
    void obfree( OBJECT *);
    
    NUMBER obdist2( OBJECT *, OBJECT *); /* squared Eucl. distance */
    
    void obput( OBJECT *pob, int pos, NUMBER val);
    
    NUMBER obget( OBJECT *pob, int pos );
    
    OBJECT *obcopy( OBJECT *p);
    
    void obprint( OBJECT *p);

    OBJECT *line2ob(char *line);

#endif
