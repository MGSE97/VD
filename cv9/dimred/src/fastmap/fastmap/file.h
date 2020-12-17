#ifndef __file_h
    #define __file_h
#define _CRT_SECURE_NO_WARNINGS
    #include <stdio.h>
    #include <string.h>
    #include <string.h>
    #include <assert.h>

    #include "def.h"

    #define MAXLINE 1000000
    #define BLANKS " \t\n"

    int countlines(FILE *fp);
    int countcolumns(FILE *fp);
    void checklength(char *line);
    int counttokens(char *line);
    char *getline(FILE *fp);
    BOOLEAN checkfile(FILE *fp);
    /* void checkfile(FILE *fp); */
    void myrewind(FILE *fp);

#endif
