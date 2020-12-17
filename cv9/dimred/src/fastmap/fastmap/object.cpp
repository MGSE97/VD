/* $Revision: 7.3 $ */
/* $Author: christos $ */
/* $Id: object.c,v 7.3 1995/03/17 00:20:17 christos Exp christos $ */

#include "object.h"
#include "file.h"
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h> 
#include <string.h> 

OBJECT *oballoc( int n){
    OBJECT *pob;

    assert( n > 0 );
    pob = (OBJECT *) malloc( sizeof(OBJECT) );
    assert( pob != NULL );
    pob-> vec = (NUMBER *) malloc ( n * sizeof(NUMBER) );
    assert( (pob->vec) != NULL );
    pob->len = n;

    return(pob);
}

void obfree( OBJECT *pob){
    free( (char *) pob->vec);
    pob->len =0;
    free( (char *) pob);
}

NUMBER obdist2( OBJECT *p1, OBJECT *p2){
    NUMBER res2; /* squared result */
    NUMBER v1, v2;
    int pos, length;

    assert( (p1->len) == (p2->len) );
    length = p1->len;
    res2=0.0;
    for(pos=0; pos< length; pos++){
	v1 = (p1->vec)[pos];
	v2 = (p2->vec)[pos];
	res2 += (v1-v2)*(v1-v2);
    }

    return( res2);
}

void obput( OBJECT *pob, int pos, NUMBER val){
    assert( pos >=0) ;
    assert( pos < (pob->len)  );
    (pob->vec)[pos] =  val;
}

NUMBER obget( OBJECT *pob, int pos){
    assert( pos >=0) ;
    assert( pos < (pob->len)  );
    return( (pob->vec)[pos] );

}

OBJECT *obcopy( OBJECT *psource){
    OBJECT *ptarget;
    int pos, length;

    assert( (psource->len) >0 );
    length = psource->len;
    ptarget = oballoc(length);
    for(pos=0; pos<length; pos++){
	(ptarget->vec)[pos] = (psource->vec)[pos];
    }
    return( ptarget);
}

void obprint( OBJECT *p){
    int i, length;

    /*  assert( (p->len) > 0 ); */
    length = p->len;

    for(i=0; i< length; i++){
	printf("%g\t", obget(p, i));
    }
    printf("\n");
}

OBJECT *line2ob(char *line){
    char *tmpline;  /* temp copy */
    OBJECT *pob;
    char *token;
    int ntokens;
    NUMBER val;
    int i;

    ntokens = counttokens(line);
    pob = oballoc(ntokens);

    /* copy, because strtok() messed up tmpline */
    tmpline=_strdup(line);

    assert(ntokens>0); /* no empty lines */
    i=0;
    token = strtok(tmpline, BLANKS);
    /* atof did not work! */
    /* val = (NUMBER) atof( token); */
    sscanf( token, "%lf", &val);
    obput(pob, i, val);
    for(i=1; i< ntokens; i++){
        token = strtok( (char *)NULL, BLANKS);
    	sscanf( token, "%lf", &val);
	obput(pob, i,val);
    }
    free(tmpline);
    return(pob);
}
