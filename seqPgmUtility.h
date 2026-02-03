//
//  seqPgmUtility.h
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#ifndef cscd439pgm_seqPgmUtility_h
#define cscd439pgm_seqPgmUtility_h

#include <math.h>

#define rowsInHeader 4      // number of rows in image header
#define maxSizeHeadRow 200  // maximal number characters in one row in the header

int * seqPgmRead( char **header, int *numRows, int *numCols, FILE *in  );

int seqPgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header );

int seqPgmDrawEdge( int *pixels, int numRows, int numCols, int edgeWidth, char **header );

int seqPgmDrawLine( int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col );

int seqPgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out );

#endif
