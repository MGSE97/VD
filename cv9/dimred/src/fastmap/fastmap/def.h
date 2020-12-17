/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: def.h,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#ifndef __def
   #define __def
   #define NULL 0

   #ifndef __NUMBER
      typedef double NUMBER;
   #endif
   #ifndef __BOOLEAN
      typedef int BOOLEAN;
      #define TRUE 1
      #define FALSE 0
   #endif
#endif
