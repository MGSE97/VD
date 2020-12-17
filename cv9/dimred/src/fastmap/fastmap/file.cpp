#include "file.h"
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <stdlib.h>

void myrewind(FILE *fp){
    if( ferror(fp) ){ clearerr(fp);}
    rewind(fp);
    return;
}

int countlines(FILE *fp){
    int nlcount = 0;
    int ichar;

    myrewind(fp);
    while( (ichar=fgetc(fp)) != EOF ){
	if( ichar == '\n' ){ nlcount ++ ;}
    }
    myrewind(fp);

    return(nlcount);
}

/* measures the number of columns of the first line */
int countcolumns(FILE *fp){
    char *line;

    myrewind(fp);

    line= getline(fp);
    return( counttokens( line) );


}

void checklength(char *line){
    /*
     strchr() is part of the C standard library. index() is a now deprecated POSIX function. The POSIX specification recommends implementing index() as a macro that expands to a call to strchar().
     Since index() is deprecated in POSIX and not part of the C standard library, you should use strchr().
    */
    if( strchr(line, '\n') == NULL ){ /* too long */
	fprintf(stderr, "too long line - stopping \n");
	exit(-1);
    }
}

int counttokens(char *line){
    char *templine;
    int res = 0 ;

    /* create a copy, because strtok() messes its arg */
    templine = _strdup(line);

    if( strtok(templine, BLANKS) != NULL){
	res ++ ;
    }
    while( strtok( (char *) NULL, BLANKS ) != NULL){
	res ++ ;
    }
    free( templine);
    return(res);
}

/* reads a line from fp, and checks its length */
char *getline(FILE *fp){
    static char line[MAXLINE];

    if( fgets(line, MAXLINE, fp) != NULL){ /* EOF */
        checklength(line);
    }
    return(line);
}

/* checks to see if all the lines of the file have the same
   number of tokens */

BOOLEAN checkfile(FILE *fp){
    char *line;
    int nold, nnew;
    int linecount;
    BOOLEAN kosher = TRUE;

    myrewind(fp);

    /* read in the first line */
    line= getline(fp);
    linecount =1;
    if( feof(fp) ){
       return TRUE; /*empty file */
    }else{
	nold = counttokens(line);
#ifdef DEBUG
	/* printf("%d:\t%s", nold, line); */
#endif
    }

    line = getline(fp);
    while( !feof(fp) ){
	linecount ++;
        nnew = counttokens(line);
#ifdef DEBUG
	//printf("%d:\t%s", nnew, line);
#endif
	if( nnew != nold){ 
	   kosher = FALSE;
	   fprintf(stderr, "line %d has %d tokens; line %d has %d\n",
	      1, nold, linecount, nnew);
	}
	/* assert( nnew == nold ); */
	line = getline(fp);
    }

    myrewind(fp);
    return (kosher);
}
