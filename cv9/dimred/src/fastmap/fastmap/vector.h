/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: vector.h,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#ifndef __vector_h
  #define __vector_h
#define _CRT_SECURE_NO_WARNINGS
  #include <malloc.h>
  #include <assert.h>
  #include <stdio.h>
  #include "def.h"
  
  
  /* 1-d vector, with size */
  typedef struct Vector { 
     int len;
     NUMBER *vec;
  } VECTOR;
  
  
  
  VECTOR *vecalloc(int n);
  
  void vecfree( VECTOR *);
  
  NUMBER vecdist2( VECTOR *, VECTOR *); /* squared Eucl. distance */
  
  void vecput( VECTOR *pvec, int pos, NUMBER val);
  
  NUMBER vecget( VECTOR *pvec, int pos );
  
  VECTOR *veccopy( VECTOR *p);
  
  void vecprint( VECTOR *p);

#endif
